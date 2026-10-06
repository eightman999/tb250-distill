"""Wikipedia 言語モデル事前学習（独立実験。意味表現蒸留とは無関係）。

Common-S と同じ embedding(128) + GRU(192 x 2) を次トークン予測で学習し、embedding と GRU の重みを Student の初期値として引き継ぐ。
出力層は Linear(H, V=8192)（キー lm.w (H,V) / lm.b (V)）。softmax CE は (B*T, V) のロジットを行 chunk に分けて計算し、
chunk ごとにその場で dlogits を作って backward する（ロジット全体をメモリに持たない）。
手書き forward/backward は既存 backend の op（gemm, gru_fwd/gru_bwd, gather_rows, scatter_add_rows, colsum, adamw）に
新規 op `softmax_ce`（np/cl 両 backend に追加）を足して構成する。

  python -m tb250distill.student.pretrain_lm --backend cl --device "WX 2100" --data data/wiki/wikiA \
      --run-dir runs/wikiA/pretrain --config common_s --batch-size 128 --epochs 1 [--resume]

  # 事前学習した embedding/GRU を Student の --init に使える npz へ（head は既存 init の値を使う）
  python -m tb250distill.student.pretrain_lm export --ckpt runs/wikiA/pretrain/ckpt/last.npz \
      --base-init runs/common/init.npz --out runs/wikiA/init.npz

データ: `tb250distill.data.wiki build` の出力（uint16 (rows, T)）。1 行 = 連続 T トークン。入力 = row[:-1]、教師 = row[1:]、
各行は h0=0 から読む。val は記事単位の held-out。学習順は (seed, epoch) だけから決まる（デバイス・backend 非依存）。
metrics.csv: step, epoch, train_loss, train_ppl, val_loss, val_ppl, lr, gnorm, samples_per_s, tokens_per_s, step_ms, vram_mb,
temp_c, power_w, gpu_busy_pct, sclk_mhz, wall_s, paused_s（AMD テレメトリは train.SmiMonitor を再利用）。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time

import numpy as np

from . import model as M
from . import train as T_

LM_FORMAT = "tb250distill-lm-v1"
CSV_COLS = ["step", "epoch", "train_loss", "train_ppl", "val_loss", "val_ppl", "lr", "gnorm", "samples_per_s",
            "tokens_per_s", "step_ms", "vram_mb", "temp_c", "power_w", "gpu_busy_pct", "sclk_mhz", "wall_s", "paused_s"]

log = logging.getLogger("tb250.pretrain_lm")


# --------------------------------------------------------------------------------------
# パラメータ
# --------------------------------------------------------------------------------------

def lm_param_specs(cfg):
    """Student の emb / GRU（名前・形・並びは model.param_specs と同一）+ LM 出力層 lm.w (H,V), lm.b (V)。head.* / sem.* は持たない。"""
    specs = [(n, s) for n, s in M.param_specs(cfg) if not n.startswith(("head.", "sem."))]
    specs += [("lm.w", (cfg.hidden, cfg.vocab)), ("lm.b", (cfg.vocab,))]
    return specs


def init_lm_params(cfg, seed, base=None):
    """emb/GRU は model.init_params(cfg, seed)（seed 0 なら runs/common/init.npz と同一）、base（dict）があればそれで上書き。
    lm.w は別乱数 stream の U(-1/sqrt(H), 1/sqrt(H))、lm.b は 0。"""
    p = M.init_params(cfg, seed)
    if base:
        for k in p:
            if k in base:
                p[k] = np.asarray(base[k], np.float32)
    out = {n: p[n] for n, _ in lm_param_specs(cfg) if n in p}
    rng = np.random.default_rng([int(seed), 0x4C4D])
    bound = 1.0 / np.sqrt(cfg.hidden)
    out["lm.w"] = rng.uniform(-bound, bound, size=(cfg.hidden, cfg.vocab)).astype(np.float32)
    out["lm.b"] = np.zeros((cfg.vocab,), np.float32)
    return out


def lm_param_count(cfg):
    return int(sum(int(np.prod(s)) for _, s in lm_param_specs(cfg)))


def save_lm_ckpt(path_base, model, meta, with_opt=True):
    params = model.get_params()
    extra = {}
    if with_opt:
        m_, v_ = model.get_opt()
        extra = {"opt_m": m_, "opt_v": v_}
    meta = dict(meta)
    meta["format"] = LM_FORMAT
    tmp = path_base + ".tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, __meta__=np.array(json.dumps(meta)), **extra, **params)
    os.replace(tmp, path_base + ".npz")
    tmpj = path_base + ".tmp.json"
    with open(tmpj, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmpj, path_base + ".json")


def load_lm_ckpt(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    cfg = M.Config.from_dict(meta["config"])
    params = {n: z[n] for n, _ in lm_param_specs(cfg)}
    opt = (z["opt_m"], z["opt_v"]) if "opt_m" in z.files else None
    return cfg, params, opt, meta


# --------------------------------------------------------------------------------------
# モデル
# --------------------------------------------------------------------------------------

class LMModel:
    """次トークン予測の GRU 言語モデル（backend 非依存）。入力 tokens (b, T+1) -> 入力 row[:, :T]、教師 row[:, 1:]。"""

    def __init__(self, be, cfg, max_batch, seq_in, chunk_rows=2048):
        self.be, self.cfg = be, cfg
        self.Bm, self.T = int(max_batch), int(seq_in)
        self.chunk = int(chunk_rows)
        H, E, L, V = cfg.hidden, cfg.emb, cfg.layers, cfg.vocab
        Bm, T = self.Bm, self.T
        self.specs = lm_param_specs(cfg)
        self.n_params = int(sum(int(np.prod(s)) for _, s in self.specs))
        self.P, self.G = be.alloc((self.n_params,)), be.alloc((self.n_params,))
        self.M, self.V = be.alloc((self.n_params,)), be.alloc((self.n_params,))
        self.p, self.g = {}, {}
        off = 0
        for name, shape in self.specs:
            n = int(np.prod(shape))
            self.p[name] = be.view(self.P, off, shape)
            self.g[name] = be.view(self.G, off, shape)
            off += n
        self.ibuf = be.alloc((2 * T * Bm + Bm,), "i")          # [tok (T*b) | tgt (T*b) | lens (b)]
        self.X0 = be.alloc((T * Bm * E,))
        self.gi = [be.alloc((T * Bm * 3 * H,)) for _ in range(L)]
        self.gh = [be.alloc((T * Bm * 3 * H,)) for _ in range(L)]
        self.hs = [be.alloc(((T + 1) * Bm * H,)) for _ in range(L)]
        self.logits = be.alloc((min(self.chunk, T * Bm) * V,))
        self.losses = be.alloc((T * Bm,))
        self.ss = be.alloc((1,))
        D = max(E, H)
        self.dX = be.alloc((T * Bm * H,))
        self.dxp = [be.alloc((T * Bm * D,)) for _ in range(2)]
        self.dgi = be.alloc((T * Bm * 3 * H,))
        self.dgh = be.alloc((T * Bm * 3 * H,))
        self.dh = be.alloc((Bm * H,))

    # ---- パラメータ入出力 ----
    def _flat_to_dict(self, flat):
        out, off = {}, 0
        for name, shape in self.specs:
            n = int(np.prod(shape))
            out[name] = flat[off:off + n].reshape(shape).copy()
            off += n
        return out

    def set_params(self, params):
        flat = np.concatenate([np.asarray(params[n]).reshape(-1) for n, _ in self.specs])
        self.be.upload(self.P, flat.astype(self.be.dtype))

    def get_params(self):
        return self._flat_to_dict(self.be.download(self.P).reshape(-1))

    def get_grads(self):
        return self._flat_to_dict(self.be.download(self.G).reshape(-1))

    def get_opt(self):
        return self.be.download(self.M).reshape(-1), self.be.download(self.V).reshape(-1)

    def set_opt(self, m, v):
        self.be.upload(self.M, np.asarray(m, dtype=self.be.dtype))
        self.be.upload(self.V, np.asarray(v, dtype=self.be.dtype))

    # ---- forward / loss / backward ----
    def _load(self, rows):
        rows = np.asarray(rows)
        b = int(rows.shape[0])
        assert b <= self.Bm and rows.shape[1] == self.T + 1, (rows.shape, self.Bm, self.T)
        x = rows[:, :-1].astype(np.int32)
        y = rows[:, 1:].astype(np.int32)
        ints = np.concatenate([np.ascontiguousarray(x.T).reshape(-1), np.ascontiguousarray(y.T).reshape(-1),
                               np.full(b, self.T, np.int32)])
        self.be.upload(self.ibuf, ints)
        v = self.be.view
        return b, v(self.ibuf, 0, (self.T * b,)), v(self.ibuf, self.T * b, (self.T * b,)), v(self.ibuf, 2 * self.T * b, (b,))

    def _forward(self, b, tok, lens):
        be, cfg = self.be, self.cfg
        H, E, L, T = cfg.hidden, cfg.emb, cfg.layers, self.T
        v = be.view
        Xin = v(self.X0, 0, (T * b, E))
        be.gather_rows(Xin, self.p["emb"], tok)
        for l in range(L):
            wi, wh = self.p[f"l{l}.wi"], self.p[f"l{l}.wh"]
            bi, bh = self.p[f"l{l}.bi"], self.p[f"l{l}.bh"]
            be.gemm(Xin, wi, v(self.gi[l], 0, (T * b, 3 * H)))
            hs = self.hs[l]
            be.zero(v(hs, 0, (b, H)))     # h0 = 0（b が変わると前バッチの残骸が block0 に残るので毎回消す）
            for t in range(T):
                hp = v(hs, t * b * H, (b, H))
                ght = v(self.gh[l], t * b * 3 * H, (b, 3 * H))
                be.gemm(hp, wh, ght)
                be.gru_fwd(v(self.gi[l], t * b * 3 * H, (b, 3 * H)), ght, bi, bh, hp,
                           v(hs, (t + 1) * b * H, (b, H)), lens, t)
            Xin = v(hs, b * H, (T * b, H))

    def _head(self, b, tgt, want_grads):
        """行 chunk ごとに logits -> softmax CE（その場で dlogits）-> (勾配なら) lm.w/lm.b の勾配と dX。"""
        be, cfg = self.be, self.cfg
        H, V, L, T = cfg.hidden, cfg.vocab, cfg.layers, self.T
        v = be.view
        N = T * b
        inv_n = 1.0 / N
        top = self.hs[L - 1]
        for r0 in range(0, N, self.chunk):
            c = min(self.chunk, N - r0)
            Xc = v(top, b * H + r0 * H, (c, H))
            lg = v(self.logits, 0, (c, V))
            be.gemm(Xc, self.p["lm.w"], lg)
            be.bias_act(lg, self.p["lm.b"], 0)
            be.softmax_ce(lg, v(tgt, r0, (c,)), v(self.losses, r0, (c,)), c, V, inv_n)
            if want_grads:
                be.gemm(Xc, lg, self.g["lm.w"], transA=True, beta=1.0)
                be.colsum(self.g["lm.b"], lg)
                be.gemm(lg, self.p["lm.w"], v(self.dX, r0 * H, (c, H)), transB=True)

    def _backward(self, b, tok, lens):
        be, cfg = self.be, self.cfg
        H, E, L, T = cfg.hidden, cfg.emb, cfg.layers, self.T
        v, p, g = be.view, self.p, self.g
        cur = self.dX
        for l in reversed(range(L)):
            din = E if l == 0 else H
            wi, wh = p[f"l{l}.wi"], p[f"l{l}.wh"]
            bi, bh = p[f"l{l}.bi"], p[f"l{l}.bh"]
            dh = v(self.dh, 0, (b, H))
            be.zero(dh)
            dgi_all = v(self.dgi, 0, (T * b, 3 * H))
            dgh_all = v(self.dgh, 0, (T * b, 3 * H))
            for t in reversed(range(T)):
                dgh_t = v(self.dgh, t * b * 3 * H, (b, 3 * H))
                be.gru_bwd(v(self.gi[l], t * b * 3 * H, (b, 3 * H)), v(self.gh[l], t * b * 3 * H, (b, 3 * H)), bi, bh,
                           v(self.hs[l], t * b * H, (b, H)), dh, v(cur, t * b * H, (b, H)),
                           v(self.dgi, t * b * 3 * H, (b, 3 * H)), dgh_t, lens, t)
                be.gemm(dgh_t, wh, dh, transB=True, beta=1.0)
            Xl = v(self.X0, 0, (T * b, E)) if l == 0 else v(self.hs[l - 1], b * H, (T * b, H))
            be.gemm(Xl, dgi_all, g[f"l{l}.wi"], transA=True, beta=1.0)
            be.gemm(v(self.hs[l], 0, (T * b, H)), dgh_all, g[f"l{l}.wh"], transA=True, beta=1.0)
            be.colsum(g[f"l{l}.bi"], dgi_all)
            be.colsum(g[f"l{l}.bh"], dgh_all)
            nxt = self.dxp[0] if cur is not self.dxp[0] else self.dxp[1]
            dx = v(nxt, 0, (T * b, din))
            be.gemm(dgi_all, wi, dx, transB=True)
            if l == 0:
                be.scatter_add_rows(g["emb"], dx, tok)
            cur = nxt

    def loss_grads(self, rows, want_grads=True):
        """rows (b, T+1) -> dict(loss=平均 nats/token, ntok)。勾配は self.G に（先に zero して）累積。"""
        be = self.be
        b, tok, tgt, lens = self._load(rows)
        if want_grads:
            be.zero(self.G)
        self._forward(b, tok, lens)
        self._head(b, tgt, want_grads)
        if want_grads:
            self._backward(b, tok, lens)
        N = self.T * b
        ls = be.download(be.view(self.losses, 0, (N,))).astype(np.float64)
        return {"loss": float(ls.mean()), "ntok": N}

    def train_step(self, rows, lr, step, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, clip=1.0):
        be = self.be
        st = self.loss_grads(rows, want_grads=True)
        be.sumsq(self.G, self.ss)
        be.adamw(self.P, self.G, self.M, self.V, self.ss, lr, beta1, beta2, eps, wd, step, clip)
        st["gnorm"] = float(np.sqrt(float(be.download(self.ss).reshape(-1)[0])))
        return st

    def eval_loss(self, rows_all, batch):
        """rows_all (n, T+1) 全行の平均 nats/token（forward のみ）。部分バッチも扱う。"""
        tot, cnt = 0.0, 0
        for i in range(0, len(rows_all), batch):
            r = np.asarray(rows_all[i:i + batch])
            st = self.loss_grads(r, want_grads=False)
            tot += st["loss"] * st["ntok"]
            cnt += st["ntok"]
        return tot / max(1, cnt)


# --------------------------------------------------------------------------------------
# データ
# --------------------------------------------------------------------------------------

def load_split(data_dir, split):
    from tb250distill.data import wiki
    parts = wiki.load_rows(data_dir, split, mmap=False)
    if not parts:
        return None
    return np.concatenate(parts, axis=0) if len(parts) > 1 else np.ascontiguousarray(parts[0])


def epoch_order(seed, epoch, n):
    return np.random.default_rng([seed, epoch, 11]).permutation(n)


def lr_at(step, total, lr, warmup, min_frac):
    """step は 1 始まり。線形 warmup -> cosine（min_frac*lr まで）。"""
    if warmup > 0 and step <= warmup:
        return lr * step / warmup
    if total <= warmup:
        return lr
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return lr * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * prog)))


# --------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------

def export_student_init(ckpt, base_init, out):
    """LM ckpt の emb / GRU と base_init の head.*（sem.* は含めない）を合わせた Student 用 init npz を書く。
    Config は base_init のもの（emb/GRU の形が一致すること）。LM 出力層は捨てる。"""
    cfg_b, base, _ = M.load_params_npz(base_init)
    cfg_l, lm, _opt, meta = load_lm_ckpt(ckpt)
    for f in ("vocab", "emb", "hidden", "layers"):
        if getattr(cfg_b, f) != getattr(cfg_l, f):
            raise SystemExit(f"構造が違う: {f} base={getattr(cfg_b, f)} lm={getattr(cfg_l, f)}")
    params = dict(base)
    for n, _s in M.param_specs(cfg_b):
        if n in lm and not n.startswith("head."):
            params[n] = np.asarray(lm[n], np.float32)
    n_repl = sum(1 for n in params if n in lm)
    M.save_params_npz(out, cfg_b, params, {"source": "wikipedia LM pretrain", "lm_ckpt": os.path.abspath(ckpt),
                                           "lm_step": meta.get("step"), "head_from": os.path.abspath(base_init),
                                           "replaced_params": n_repl})
    cfg2, p2, _ = M.load_params_npz(out)     # train.py --init と同じ読み方で検証
    assert cfg2.to_dict() == cfg_b.to_dict()
    return {"out": out, "n_params": M.param_count(cfg2), "replaced": n_repl, "lm_step": meta.get("step")}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Wikipedia LM 事前学習", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backend", default="np", choices=["np", "cl"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", required=True, help="data/wiki/<name>（wiki.py build の出力）")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="common_s")
    ap.add_argument("--init", default=None, help="emb/GRU の初期値 npz（Student 形式。runs/common/init.npz 等）。無ければ seed から生成")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="総 step・lr 計画はそのままに、この step で一旦止める（last を保存して終了。--resume で続きを回す）。テスト・時間制限用")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--chunk-rows", type=int, default=2048, help="softmax の行 chunk（logits バッファ = chunk x V floats）")
    ap.add_argument("--limit-rows", type=int, default=None, help="train の先頭 N 行だけ使う（速度実測・スモーク用）")
    ap.add_argument("--val-rows", type=int, default=2048, help="validation に使う行数の上限（先頭から）")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smi-interval", type=float, default=20.0)
    ap.add_argument("--temp-pause", type=float, default=88.0)
    ap.add_argument("--temp-resume", type=float, default=78.0)
    ap.add_argument("--no-temp-guard", action="store_true")
    ap.add_argument("--np-dtype", default="float32", choices=["float32", "float64"])
    return ap.parse_args(argv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export":
        ap = argparse.ArgumentParser(prog="pretrain_lm export")
        ap.add_argument("--ckpt", required=True)
        ap.add_argument("--base-init", required=True, help="head の初期値を借りる Student init（runs/common/init.npz）")
        ap.add_argument("--out", required=True)
        a = ap.parse_args(argv[1:])
        print(json.dumps(export_student_init(a.ckpt, a.base_init, a.out)))
        return 0
    args = parse_args(argv)
    run_dir = args.run_dir
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
    last_path = os.path.join(run_dir, "ckpt", "last")
    resuming = args.resume and os.path.exists(last_path + ".npz")
    if (not args.resume) and os.path.exists(last_path + ".npz"):
        raise SystemExit(f"{run_dir} には既存 checkpoint がある。--resume するか別の --run-dir を使う")
    handlers = [logging.FileHandler(os.path.join(run_dir, "train.log"), mode="a"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers, force=True)

    tr = load_split(args.data, "train")
    va = load_split(args.data, "val")
    if tr is None:
        raise SystemExit(f"train shard が無い: {args.data}")
    if args.limit_rows:
        tr = tr[:args.limit_rows]
    if va is not None:
        va = va[:args.val_rows]
    T1 = int(tr.shape[1])                     # 行長 = seq_len（入力 T1-1 ステップ）
    meta_path = os.path.join(args.data, "meta.json")
    data_meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}

    if resuming:
        cfg, params, opt, rmeta = load_lm_ckpt(last_path + ".npz")
    else:
        cfg = M.get_config(args.config)
        base = None
        if args.init:
            cfg_i, base, _ = M.load_params_npz(args.init)
            if (cfg_i.vocab, cfg_i.emb, cfg_i.hidden, cfg_i.layers) != (cfg.vocab, cfg.emb, cfg.hidden, cfg.layers):
                raise SystemExit("--init の構造が --config と違う")
        params = init_lm_params(cfg, args.seed, base)
    if int(tr.max()) >= cfg.vocab:
        raise SystemExit(f"shard の最大 token id {int(tr.max())} >= vocab {cfg.vocab}")
    cfg.lp = T1 - 1

    be = M.make_backend(args.backend, args.device, args.np_dtype)
    model = LMModel(be, cfg, args.batch_size, T1 - 1, chunk_rows=args.chunk_rows)
    model.set_params(params)

    n = int(tr.shape[0])
    spe = n // args.batch_size                   # 端数バッチは捨てる（B 固定）
    total = int(round(args.epochs * spe))
    if args.max_steps:
        total = min(total, args.max_steps)
    total = max(total, 1)
    eval_every = args.eval_every or spe
    ck_pcts = {int(round(total * f)): int(round(f * 100)) for f in (0.25, 0.5, 0.75, 1.0)}

    git = T_.git_info()
    step, wall_prev, paused = 0, 0.0, {"s": 0.0, "n": 0}
    samples_done, tokens_done = 0, 0
    power_prev = (0.0, 0.0)
    best = {"val_loss": float("inf"), "step": None}
    if resuming:
        step = int(rmeta["step"])
        wall_prev = float(rmeta.get("wall_s", 0.0))
        paused = {"s": float(rmeta.get("paused_s", 0.0)), "n": int(rmeta.get("pauses", 0))}
        samples_done = int(rmeta.get("samples_done", step * args.batch_size))
        tokens_done = int(rmeta.get("tokens_done", samples_done * (T1 - 1)))
        power_prev = (rmeta.get("power_integral_j", 0.0), rmeta.get("power_time_s", 0.0))
        best = rmeta.get("best", best)
        model.set_opt(*opt)
        log.info("RESUME from step %d (wall %.1fs) best=%s", step, wall_prev, best)

    gpu = T_.SmiMonitor(be, args.smi_interval)
    if resuming:
        gpu.restore(*power_prev)
    gpu_info = gpu.info()
    log.info("gpu monitor: %s", json.dumps(gpu_info, ensure_ascii=False))
    if not resuming:
        status = T_.collect_environment(run_dir, be, args, cfg)
        T_._add_to_hardware_json(run_dir, "student_gpu_monitor", gpu_info)
        cfgd = {"kind": "wikipedia LM pretrain (independent of sem distillation)", "args": vars(args), "model": cfg.to_dict(),
                "n_params_lm": lm_param_count(cfg),
                "n_params_student_part": lm_param_count(cfg) - cfg.hidden * cfg.vocab - cfg.vocab,
                "backend": args.backend, "device": getattr(be, "device_name", None),
                "data": {"dir": args.data, "n_train_rows": n, "n_val_rows": 0 if va is None else int(va.shape[0]),
                         "row_len": T1, "input_len": T1 - 1, "meta": data_meta},
                "plan": {"steps_per_epoch": spe, "total_steps": total, "eval_every": eval_every,
                         "tokens_per_epoch_pred": spe * args.batch_size * (T1 - 1),
                         "lr_schedule": f"linear warmup {args.warmup} -> cosine to {args.min_lr_frac}*lr"},
                "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": args.wd,
                              "clip_grad_norm": args.clip},
                "loss": "mean softmax cross-entropy (nats/token) over next-token targets; chunked logits (rows=%d)" % args.chunk_rows,
                "git_commit": git["commit"], "init": args.init or "seed-init", **status}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfgd, f, indent=1, ensure_ascii=False)
        if git["diff"]:
            with open(os.path.join(run_dir, "git.diff"), "w") as f:
                f.write(git["diff"])
    log.info("backend=%s device=%s params(lm)=%d steps=%d (spe=%d) rows=%d T=%d B=%d git=%s", args.backend,
             getattr(be, "device_name", None), lm_param_count(cfg), total, spe, n, T1 - 1, args.batch_size,
             (git["commit"] or "none")[:12])

    csv_path = os.path.join(run_dir, "metrics.csv")
    new_csv = not (resuming and os.path.exists(csv_path))
    csv_f = open(csv_path, "a" if not new_csv else "w", newline="")
    cw = csv.writer(csv_f)
    if new_csv:
        cw.writerow(CSV_COLS)
    t_start = time.perf_counter()

    def wall():
        return wall_prev + (time.perf_counter() - t_start)

    def meta_now(pct=None):
        return {"config": cfg.to_dict(), "step": step, "epoch": step / spe, "steps_per_epoch": spe, "total_steps": total,
                "pct": pct, "seed": args.seed, "wall_s": wall(), "paused_s": paused["s"], "pauses": paused["n"],
                "best": best, "samples_done": samples_done, "tokens_done": tokens_done,
                "power_integral_j": gpu.power_integral_j, "power_time_s": gpu.power_time_s,
                "dataset_position": {"epoch": step // spe, "batch_in_epoch": step % spe,
                                     "order": "default_rng([seed,epoch,11]).permutation(rows)"},
                "git_commit": git["commit"], "args": vars(args)}

    def write_row(trn, val, lr, gn):
        gpu.poll(force=val is not None)
        row = {"step": step, "epoch": round(step / spe, 4), "lr": lr, "gnorm": gn}
        if trn:
            row.update({"train_loss": trn["loss"], "train_ppl": math.exp(min(trn["loss"], 50)), "samples_per_s": trn["sps"],
                        "tokens_per_s": trn["tps"], "step_ms": trn["ms"]})
        if val is not None:
            row.update({"val_loss": val, "val_ppl": math.exp(min(val, 50))})
        row.update({"vram_mb": gpu.vram, "temp_c": gpu.temp, "power_w": gpu.power, "gpu_busy_pct": gpu.busy,
                    "sclk_mhz": gpu.sclk, "wall_s": round(wall(), 2), "paused_s": round(paused["s"], 2)})
        cw.writerow(["" if row.get(c) is None else (f"{row[c]:.6g}" if isinstance(row.get(c), float) else row.get(c))
                     for c in CSV_COLS])
        csv_f.flush()

    def temp_guard():
        if not gpu.enabled or args.no_temp_guard:
            return
        gpu.poll()
        if gpu.temp is None or gpu.temp < args.temp_pause:
            return
        t0 = time.perf_counter()
        log.warning("TEMP PAUSE at step %d: %.0fC >= %.0fC", step, gpu.temp, args.temp_pause)
        gpu.recording = False
        try:
            while True:
                time.sleep(5.0)
                gpu.poll(force=True)
                if gpu.temp is None or gpu.temp <= args.temp_resume:
                    break
        finally:
            gpu.recording = True
        paused["s"] += time.perf_counter() - t0
        paused["n"] += 1

    def do_eval(tag=""):
        nonlocal best
        if va is None or len(va) == 0:
            return None
        vl = model.eval_loss(va, args.batch_size)
        log.info("VAL step=%d%s loss=%.4f ppl=%.2f (%d rows)", step, tag, vl, math.exp(min(vl, 50)), len(va))
        if vl < best["val_loss"]:
            best = {"val_loss": vl, "step": step, "val_ppl": math.exp(min(vl, 50))}
            save_lm_ckpt(os.path.join(run_dir, "ckpt", "best"), model, meta_now("best"), with_opt=False)
            log.info("  new best val_loss=%.4f -> ckpt/best", vl)
        return vl

    if not resuming:
        gpu.poll(force=True)
        v0 = do_eval(" (init)")
        write_row(None, v0, 0.0, None)
        save_lm_ckpt(last_path, model, meta_now())

    acc = {"loss": 0.0, "n": 0, "t": 0.0, "steps": 0}
    cur_epoch, order = -1, None
    interrupted = False
    try:
        while step < total and not (args.stop_after and step >= args.stop_after):
            temp_guard()
            ep, pos = divmod(step, spe)
            if ep != cur_epoch:
                order = epoch_order(args.seed, ep, n)
                cur_epoch = ep
            idx = np.sort(order[pos * args.batch_size:(pos + 1) * args.batch_size])
            lr = lr_at(step + 1, total, args.lr, args.warmup, args.min_lr_frac)
            t0 = time.perf_counter()
            st = model.train_step(tr[idx], lr, step + 1, wd=args.wd, clip=args.clip)
            dt = time.perf_counter() - t0
            step += 1
            if not math.isfinite(st["loss"]):
                raise RuntimeError(f"non-finite loss at step {step}: {st}")
            samples_done += len(idx)
            tokens_done += st["ntok"]
            acc["loss"] += st["loss"]
            acc["n"] += 1
            acc["t"] += dt
            acc["steps"] += 1
            acc["tok"] = acc.get("tok", 0) + st["ntok"]
            do_val = (step % eval_every == 0) or step == total
            if step % args.log_every == 0 or do_val:
                gpu.poll()
                trn = {"loss": acc["loss"] / acc["n"], "sps": acc["steps"] * args.batch_size / acc["t"],
                       "tps": acc["tok"] / acc["t"], "ms": acc["t"] / acc["steps"] * 1e3}
                log.info("step %d/%d ep %.3f loss %.4f ppl %.1f gnorm %.3f lr %.2e | %.1f rows/s %.0f tok/s %.0f ms/step "
                         "vram %s temp %s power %s busy %s sclk %s", step, total, step / spe, trn["loss"],
                         math.exp(min(trn["loss"], 50)), st["gnorm"], lr, trn["sps"], trn["tps"], trn["ms"], gpu.vram,
                         gpu.temp, gpu.power, gpu.busy, gpu.sclk)
                vl = do_eval() if do_val else None
                write_row(trn, vl, lr, st["gnorm"])
                acc = {"loss": 0.0, "n": 0, "t": 0.0, "steps": 0}
            if step in ck_pcts:
                save_lm_ckpt(os.path.join(run_dir, "ckpt", f"p{ck_pcts[step]:03d}"), model, meta_now(ck_pcts[step]),
                             with_opt=False)
            if args.ckpt_every and step % args.ckpt_every == 0 and step != total:
                save_lm_ckpt(last_path, model, meta_now())
    except KeyboardInterrupt:
        interrupted = True
        log.info("interrupted at step %d; saving last checkpoint", step)
    save_lm_ckpt(last_path, model, meta_now())
    csv_f.close()
    train_wall = wall()
    summary = {"steps_done": step, "total_steps": total, "interrupted": interrupted, "wall_s": train_wall,
               "paused_s": paused["s"], "pauses": paused["n"], "wall_active_s": train_wall - paused["s"],
               "best": best, "vram_max_mb": gpu.vram_max, "temp_max_c": gpu.temp_max, "samples_done": samples_done,
               "tokens_done": tokens_done, "tokens_per_s_mean": tokens_done / max(1e-9, train_wall - paused["s"]),
               "energy": T_.energy_summary(gpu, samples_done, train_wall - paused["s"])}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log.info("done: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
