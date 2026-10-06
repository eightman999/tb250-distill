"""GRU scorer（手書き forward/backward）と Tokenized shard の読み込み・バッチ詰め。

f(context, question, candidate) -> scalar:
  prefix(question<sep>context) を L 層 GRU で一度読み（右 pad + 長さマスクで最終 state を carry）、
  各候補 token 列を prefix の最終 hidden（全層）から継続して読み、
  最終層の最終 hidden -> Linear(H,H)+tanh -> Linear(H,1)。K 候補を softmax。
  候補位置の情報はモデルに入らない（構造的に permutation 等変）。

GRU 式は PyTorch 互換（ゲート順 r,z,n）:
  r = σ(W_ir x + b_ir + W_hr h + b_hr)
  z = σ(W_iz x + b_iz + W_hz h + b_hz)
  n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
  h' = (1-z)*n + z*h
重みは行列積しやすいよう転置して保持する: wi=(in,3H)=W_ih^T, wh=(H,3H)=W_hh^T。

backend 非依存: backend_np / backend_cl の op（backend_np.py 冒頭参照）だけを使う。
パラメータ・勾配・Adam 状態は各 1 本の平坦 buffer（個別パラメータはそのビュー）で、
AdamW・grad norm・zero_grad は 1 回の kernel で済む。
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict, fields

import numpy as np

# --------------------------------------------------------------------------------------
# 設定・パラメータ
# --------------------------------------------------------------------------------------


@dataclass
class Config:
    vocab: int = 8192
    emb: int = 128
    hidden: int = 192
    layers: int = 2
    lp: int = 128   # prefix 最大長（shard の Lp。GRU は位置埋め込みを持たないので情報用）
    lc: int = 16    # 候補最大長
    sem_dim: int = 0  # >0 なら Candidate Semantic Distillation 用 projection head Linear(H, sem_dim) を持つ（学習専用。推論は使わない）

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        names = {f.name for f in fields(Config)}
        return Config(**{k: v for k, v in d.items() if k in names})


# Common-S は DESIGN.md の通り。他は INSTRUCTIONS §12 の目安（emb/hidden/layers/ctx/vocab）に従い、
# s710/s730 は指示書の「3-5M / 8-12M」に収まるよう実測の param 数で決めた（param_count() 参照）。
PRESETS = {
    "common_s": Config(vocab=8192, emb=128, hidden=192, layers=2, lp=128, lc=16),
    "s430": Config(vocab=8192, emb=128, hidden=128, layers=2, lp=128, lc=16),
    "s710": Config(vocab=8192, emb=256, hidden=256, layers=3, lp=192, lc=16),
    # 実データの spm 語彙は約4k（8192 未満）のため vocab 16384 は未使用行になる。8192 のまま 4 層で 8-12M に合わせる。
    "s730": Config(vocab=8192, emb=256, hidden=512, layers=4, lp=256, lc=16),
}


def get_config(spec, **overrides):
    """spec = preset 名 / JSON ファイル / JSON 文字列 / dict / Config"""
    if isinstance(spec, Config):
        cfg = Config(**spec.to_dict())
    elif isinstance(spec, dict):
        cfg = Config.from_dict(spec)
    elif spec in PRESETS:
        cfg = Config(**PRESETS[spec].to_dict())
    elif os.path.isfile(str(spec)):
        with open(spec) as f:
            d = json.load(f)
        cfg = Config.from_dict(d.get("model", d))
    else:
        cfg = Config.from_dict(json.loads(spec))
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def param_specs(cfg):
    H, E, V, L = cfg.hidden, cfg.emb, cfg.vocab, cfg.layers
    specs = [("emb", (V, E))]
    for l in range(L):
        din = E if l == 0 else H
        specs += [(f"l{l}.wi", (din, 3 * H)), (f"l{l}.wh", (H, 3 * H)),
                  (f"l{l}.bi", (3 * H,)), (f"l{l}.bh", (3 * H,))]
    specs += [("head.w1", (H, H)), ("head.b1", (H,)), ("head.w2", (H, 1)), ("head.b2", (1,))]
    if cfg.sem_dim:
        # 末尾に足す（sem_dim=0 のとき flat layout・init・checkpoint は従来と完全に同一）。
        # 将来の Context Semantic Distillation 用 head（prefix 最終 hidden -> context embedding）もここへ "sem.wc"/"sem.bc" として足す想定。
        specs += [("sem.wp", (H, cfg.sem_dim)), ("sem.bp", (cfg.sem_dim,))]
    return specs


def param_count(cfg, breakdown=False):
    specs = param_specs(cfg)
    tot = int(sum(int(np.prod(s)) for _, s in specs))
    if not breakdown:
        return tot
    emb = int(np.prod(specs[0][1]))
    head = int(sum(int(np.prod(s)) for n, s in specs if n.startswith("head")))
    sem = int(sum(int(np.prod(s)) for n, s in specs if n.startswith("sem.")))
    out = {"total": tot, "embedding": emb, "gru": tot - emb - head - sem, "head": head}
    if sem:
        out["sem_head"] = sem   # 学習専用（推論の parameter 数は total - sem_head）
    return out


def init_params(cfg, seed):
    """seed 決定的な初期値（numpy default_rng/PCG64、float32）。全 GPU で同一 init を使うため npz に保存する。
    emb ~ N(0,1)、GRU/Linear は PyTorch 既定と同じ U(-1/sqrt(fan), 1/sqrt(fan))（GRU は fan=H）。"""
    rng = np.random.default_rng(seed)
    H = cfg.hidden
    out = {}
    for name, shape in param_specs(cfg):
        if name.startswith("sem."):
            continue  # 別 stream（init_sem_params）。base の乱数列を変えない
        if name == "emb":
            a = rng.standard_normal(shape)
        else:
            fan = H
            if name == "head.w1" or name == "head.b1":
                fan = H
            bound = 1.0 / np.sqrt(fan)
            a = rng.uniform(-bound, bound, size=shape)
        out[name] = a.astype(np.float32)
    out.update(init_sem_params(cfg, seed))
    return out


def init_sem_params(cfg, seed):
    """projection head（sem.wp, sem.bp）の seed 決定的な初期値。PyTorch Linear 既定と同じ U(-1/sqrt(H), 1/sqrt(H))。
    base params とは独立の乱数 stream（default_rng([seed, 0x53454D])）なので、sem_dim の有無・値で base の init は変わらない。"""
    if not cfg.sem_dim:
        return {}
    rng = np.random.default_rng([int(seed), 0x53454D])
    bound = 1.0 / np.sqrt(cfg.hidden)
    return {"sem.wp": rng.uniform(-bound, bound, size=(cfg.hidden, cfg.sem_dim)).astype(np.float32),
            "sem.bp": rng.uniform(-bound, bound, size=(cfg.sem_dim,)).astype(np.float32)}


def save_params_npz(path, cfg, params, extra=None):
    meta = {"config": cfg.to_dict(), "format": "tb250distill-student-v1"}
    if extra:
        meta.update(extra)
    np.savez(path, __meta__=np.array(json.dumps(meta)), **params)


def load_params_npz(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    cfg = Config.from_dict(meta["config"])
    params = {n: z[n] for n, _ in param_specs(cfg) if not n.startswith("sem.")}
    if cfg.sem_dim:
        missing = [n for n in ("sem.wp", "sem.bp") if n not in z.files]
        if missing:
            raise ValueError(f"{path}: config.sem_dim={cfg.sem_dim} なのに {missing} が無い")
        params.update({n: z[n] for n in ("sem.wp", "sem.bp")})
    return cfg, params, meta


# --------------------------------------------------------------------------------------
# Tokenized shard / バッチ
# --------------------------------------------------------------------------------------

SHARD_KEYS = ("item_id", "prefix", "prefix_len", "cand", "cand_len", "k", "t_logits", "gold")


def find_shard(data_dir, split, lp=None):
    """`<split>_L{lp}.npz` → `<split>.npz` → `<split>_L*.npz`（最大 Lp）の順に探す。"""
    cands = []
    if lp is not None:
        cands.append(os.path.join(data_dir, f"{split}_L{lp}.npz"))
    cands.append(os.path.join(data_dir, f"{split}.npz"))
    for c in cands:
        if os.path.isfile(c):
            return c
    g = glob.glob(os.path.join(data_dir, f"{split}_L*.npz"))
    if g:
        def lpv(p):
            try:
                return int(os.path.basename(p).rsplit("_L", 1)[1].split(".")[0])
            except Exception:
                return 0
        return sorted(g, key=lpv)[-1]
    return None


class Shard:
    """shard npz の薄いラッパ（全配列をメモリへ）。余分なキー（variant, variant_of など）は extra に残す。"""

    def __init__(self, path_or_dict):
        if isinstance(path_or_dict, dict):
            d = dict(path_or_dict)
            self.path = None
        else:
            z = np.load(path_or_dict, allow_pickle=False)
            d = {k: z[k] for k in z.files}
            self.path = path_or_dict
        miss = [k for k in SHARD_KEYS if k not in d]
        if miss:
            raise ValueError(f"shard {self.path} missing keys {miss}")
        self.item_id = d["item_id"].astype(np.int64)
        self.prefix = d["prefix"].astype(np.int32)
        self.prefix_len = d["prefix_len"].astype(np.int32)
        self.cand = d["cand"].astype(np.int32)
        self.cand_len = d["cand_len"].astype(np.int32)
        self.k = d["k"].astype(np.int32)
        self.t_logits = d["t_logits"].astype(np.float32)
        self.gold = d["gold"].astype(np.int32)
        self.extra = {k: v for k, v in d.items() if k not in SHARD_KEYS}
        self.n = int(self.item_id.shape[0])
        self.lp = int(self.prefix.shape[1])
        self.kmax = int(self.cand.shape[1])
        self.lc = int(self.cand.shape[2])
        # 契約: 長さが pad 長を超えない・pad 候補の cand_len は 0 として扱う
        self.prefix_len = np.minimum(self.prefix_len, self.lp)
        self.cand_len = np.minimum(self.cand_len, self.lc)

    def max_token(self):
        return int(max(self.prefix.max(initial=0), self.cand.max(initial=0)))

    def subset(self, idx):
        d = {k: getattr(self, k) for k in SHARD_KEYS}
        d = {k: v[idx] for k, v in d.items()}
        for k, v in self.extra.items():
            d[k] = v[idx] if getattr(v, "shape", None) and v.shape[:1] == (self.n,) else v
        return Shard(d)


@dataclass
class Batch:
    B: int
    K: int
    T: int           # prefix の実ステップ数（バッチ内最大 prefix_len）
    Tc: int          # 候補の実ステップ数
    ints: np.ndarray  # int32 パック: [ptok(T*B) ctok(Tc*N) plen(B) clen(N) kcnt(B) gold(B) rep(N)]
    tlog: np.ndarray  # float32 (B*K)
    perm: np.ndarray  # (B,K): perm[b,j] = 位置 j に置かれた元の候補 index
    gold: np.ndarray  # (B) 並べ替え後の gold index（無しは -1）
    kcnt: np.ndarray  # (B)
    n_tokens: int     # 実トークン数（prefix + 実候補）
    item_idx: np.ndarray
    w_kd: np.ndarray | None = None  # (B) sample ごとの KD 重み（gold 有りの sample にだけ効く）。None=全 sample 既定重み
    w_ce: np.ndarray | None = None  # (B) 同 CE 重み
    sem_t: np.ndarray | None = None      # (B*K, d) float32: 候補ごとの Teacher PCA embedding（無効候補は 0 行）。Candidate Semantic Distillation 用
    sem_valid: np.ndarray | None = None  # (B*K) bool: sem loss に入る候補（実候補 かつ embedding index >= 0）


def perm_from_keys(keys, k, K):
    """keys (B,>=K) の乱数から各 item の候補 permutation（pad 候補は末尾に固定）を作る。"""
    kk = np.array(keys[:, :K], dtype=np.float64)
    pad = np.arange(K)[None, :] >= k[:, None]
    kk = np.where(pad, 2.0 + np.arange(K)[None, :], kk)  # pad は順序保存で末尾
    return np.argsort(kk, axis=1, kind="stable")


def parse_source_loss(specs):
    """`--source-loss` の解釈。"jnli:kd=0.2,ce=0.8" 形式（複数 source は ';' 区切り、または文字列を複数渡す）。
    戻り値 {source: (w_kd, w_ce)}。kd と ce は両方必須（gold 有り sample の重みを置き換える）。"""
    if not specs:
        return {}
    if isinstance(specs, str):
        specs = [specs]
    out = {}
    for spec in specs:
        for part in str(spec).split(";"):
            part = part.strip()
            if not part:
                continue
            name, sep, rest = part.partition(":")
            name = name.strip()
            if not sep or not name:
                raise ValueError(f"--source-loss: '{part}' は 'source:kd=..,ce=..' 形式で書く")
            kv = {}
            for tok in rest.split(","):
                k_, eq, v_ = tok.partition("=")
                k_ = k_.strip()
                if not eq or k_ not in ("kd", "ce") or k_ in kv:
                    raise ValueError(f"--source-loss: '{part}' の '{tok.strip()}' が不正（kd=,ce= を 1 回ずつ）")
                try:
                    kv[k_] = float(v_)
                except ValueError:
                    raise ValueError(f"--source-loss: '{part}' の {k_} が数値でない: {v_!r}") from None
            if set(kv) != {"kd", "ce"}:
                raise ValueError(f"--source-loss: '{part}' は kd と ce の両方が必要")
            if not all(np.isfinite(v) and v >= 0 for v in kv.values()):
                raise ValueError(f"--source-loss: '{part}' の重みは有限の非負数")
            if name in out:
                raise ValueError(f"--source-loss: source '{name}' が重複")
            out[name] = (kv["kd"], kv["ce"])
    return out


def source_weights(sh, idx, source_loss):
    """item idx の sample ごとの (w_kd, w_ce)（float64, B）。shard に `source` が無い、または source_loss が空なら None。
    指定外の source は既定重み (W_KD_GOLD, W_CE_GOLD)。gold 無し sample には kernel 側で効かない。"""
    if not source_loss:
        return None
    src = sh.extra.get("source") if hasattr(sh, "extra") else None
    if src is None or getattr(src, "shape", None) != (sh.n,):
        return None
    s = np.asarray(src)[np.asarray(idx)]
    w_kd = np.full(s.shape[0], W_KD_GOLD, np.float64)
    w_ce = np.full(s.shape[0], W_CE_GOLD, np.float64)
    for name, (a, b) in source_loss.items():
        m = s == name
        w_kd[m] = a
        w_ce[m] = b
    return w_kd, w_ce


class SemTargets:
    """Candidate Semantic Distillation の target。train shard の行 × 候補位置 -> unique 文字列 index（無効 -1）と、
    unique 文字列ごとの Teacher PCA 済み embedding [U, d]。val/test/robust の文字列は含めない（embed.py が強制）。"""

    def __init__(self, cand_idx, emb, allow_masked=False):
        self.cand_idx = np.asarray(cand_idx, np.int32)
        self.emb = np.ascontiguousarray(emb, np.float32)
        if self.cand_idx.size and int(self.cand_idx.max()) >= self.emb.shape[0]:
            raise ValueError("cand_idx が emb の行数を超えている")
        self.d = int(self.emb.shape[1])
        # True: 候補位置の一部が意味表現蒸留の対象外（-1）でもよい（teacher.embed --exclude-truncated の出力。meta で宣言されたときだけ）
        self.allow_masked = bool(allow_masked)

    def subset(self, idx):
        return SemTargets(self.cand_idx[np.asarray(idx)], self.emb, allow_masked=self.allow_masked)

    def check_against(self, sh):
        """shard と行・候補位置が対応していること（有効 index の位置 == 候補がある位置 j < k）を検証する。
        allow_masked（--exclude-truncated 出力）なら、有効位置は候補がある位置の部分集合であればよい（pad 位置 j >= k は必ず -1）。"""
        if self.cand_idx.shape != (sh.n, sh.kmax):
            raise ValueError(f"cand_idx の shape {self.cand_idx.shape} != shard (n={sh.n}, kmax={sh.kmax})")
        want = np.arange(sh.kmax)[None, :] < sh.k[:, None]
        got = self.cand_idx >= 0
        if self.allow_masked:
            if (got & ~want).any():
                raise ValueError("cand_idx が shard の候補の無い位置（j >= k）に有効 index を持つ")
        elif not np.array_equal(got, want):
            raise ValueError("cand_idx の有効位置が shard の候補位置（j < k）と一致しない")


def item_id_sha1(item_id):
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(item_id, np.int64).tobytes()).hexdigest()


EVAL_ONLY_DIRNAME = "emb_eval"   # 評価専用 Teacher 埋め込み（teacher/diag_sem）の置き場。学習コードは読まない


def assert_not_eval_only_dir(path):
    """data/emb_eval 配下（評価専用 Teacher 埋め込み）を学習用 target として読もうとしたら拒否する。
    パス成分（symlink 解決前後の両方）に `emb_eval` を含む、または EVAL_ONLY マーカー・meta の eval_only があれば ValueError。"""
    p = str(path)
    for q in (os.path.abspath(p), os.path.realpath(p)):
        if EVAL_ONLY_DIRNAME in q.replace("\\", "/").split("/"):
            raise ValueError(f"{path}: 評価専用 Teacher 埋め込み（{EVAL_ONLY_DIRNAME}）は学習に使えない")
    if os.path.exists(os.path.join(p, "EVAL_ONLY")):
        raise ValueError(f"{path}: EVAL_ONLY マーカーがある（評価専用 Teacher 埋め込みは学習に使えない）")
    mp = os.path.join(p, "meta.json")
    if os.path.isfile(mp):
        try:
            with open(mp) as f:
                if json.load(f).get("eval_only"):
                    raise ValueError(f"{path}: meta.json が eval_only（評価専用 Teacher 埋め込みは学習に使えない）")
        except json.JSONDecodeError:
            pass


def load_sem_dir(path, item_id, sem_dim=None):
    """`data/emb/<provider>/<shard>/`（teacher/embed.py の出力）から SemTargets を読む。
    train shard の item_id と meta.json の item_id_sha1 が一致すること（shard を作り直したのに embedding が古い事故を防ぐ）、
    cand_idx の有効位置が「候補がある位置」と一致することを検証する。sem_dim は PCA 次元 d 以下ならその先頭成分だけ使う（既定 = d）。
    戻り値: (SemTargets, info dict)。
    data/emb_eval（評価専用）は assert_not_eval_only_dir で拒否する。"""
    assert_not_eval_only_dir(path)
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("item_id_sha1") != item_id_sha1(item_id):
        raise ValueError(f"{path}: train shard の item_id が embedding 作成時と一致しない（shard を作り直した? embed を再実行する）")
    ci = np.load(os.path.join(path, "cand_idx_train.npy"))
    if ci.shape[0] != len(item_id):
        raise ValueError(f"{path}: cand_idx_train の行数 {ci.shape[0]} != shard {len(item_id)}")
    pca_path = os.path.join(path, "emb_pca.npy")
    if os.path.isfile(pca_path):
        emb = np.load(pca_path)
    else:
        z = np.load(os.path.join(path, "pca.npz"))
        raw = np.load(os.path.join(path, "emb_raw.npy"))
        x = raw.astype(np.float64)
        if int(z["l2_normalize"]):
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        emb = ((x - z["mean"]) @ z["components"].T).astype(np.float32)
    d_all = int(emb.shape[1])
    d = d_all if sem_dim is None else int(sem_dim)
    if d < 1 or d > d_all:
        raise ValueError(f"--sem-dim {d} は PCA 次元 {d_all} 以下の正の整数")
    masked = bool((meta.get("exclude_truncated") or {}).get("applied"))
    info = {"dir": path, "provider": meta.get("provider"), "d_pca": d_all, "d": d,
            "n_strings": int(emb.shape[0]), "shard": meta.get("shard"),
            "explained_variance_ratio": meta.get("explained_variance_ratio")}
    if masked:
        info["exclude_truncated"] = {k: meta["exclude_truncated"].get(k) for k in ("lc", "excluded_slots", "excluded_by_source")}
    return SemTargets(ci, emb[:, :d], allow_masked=masked), info


def make_batch(sh, idx, keys=None, perm=None, source_loss=None, sem=None):
    """shard の item idx からバッチを作る。keys/perm で候補順序を並べ替える（teacher logits・gold も追従）。
    source_loss（parse_source_loss の結果）があり shard に `source` があれば sample ごとの損失重みを付ける。
    sem（SemTargets、shard と行が対応）があれば候補ごとの target embedding（並べ替え後の順序）を付ける。"""
    idx = np.asarray(idx)
    B = int(idx.shape[0])
    k = sh.k[idx]
    K = int(max(1, k.max()))
    if perm is None:
        if keys is None:
            perm = np.tile(np.arange(K), (B, 1))
        else:
            perm = perm_from_keys(keys, k, K)
    cand = sh.cand[idx][:, :K]
    clen = sh.cand_len[idx][:, :K]
    tl = sh.t_logits[idx][:, :K]
    ar = np.arange(B)[:, None]
    cand = cand[ar, perm]
    clen = clen[ar, perm]
    tl = tl[ar, perm]
    gold0 = sh.gold[idx]
    inv = np.argsort(perm, axis=1)
    gold = np.where(gold0 >= 0, inv[np.arange(B), np.clip(gold0, 0, K - 1)], -1).astype(np.int32)
    valid = np.arange(K)[None, :] < k[:, None]
    clen = np.where(valid, clen, 0).astype(np.int32)
    tl = np.where(valid, tl, 0.0).astype(np.float32)
    plen = sh.prefix_len[idx]
    T = int(max(1, plen.max()))
    Tc = int(max(1, clen.max()))
    N = B * K
    ptok = np.ascontiguousarray(sh.prefix[idx][:, :T].T).reshape(-1)
    ctok = np.ascontiguousarray(cand.reshape(N, -1)[:, :Tc].T).reshape(-1)
    rep = (np.arange(N) // K).astype(np.int32)
    ints = np.concatenate([ptok, ctok, plen.astype(np.int32), clen.reshape(-1), k.astype(np.int32),
                           gold, rep]).astype(np.int32)
    ntok = int(plen.sum() + clen.sum())
    sw = source_weights(sh, idx, source_loss)
    sem_t = sem_valid = None
    if sem is not None:
        ci = sem.cand_idx[idx][:, :K][ar, perm]
        sv = valid & (ci >= 0)
        sem_t = sem.emb[np.where(sv, ci, 0).reshape(-1)] * sv.reshape(-1, 1)
        sem_t = np.ascontiguousarray(sem_t, np.float32)
        sem_valid = sv.reshape(-1)
    return Batch(B=B, K=K, T=T, Tc=Tc, ints=ints, tlog=np.ascontiguousarray(tl.reshape(-1)),
                 perm=perm, gold=gold, kcnt=k, n_tokens=ntok, item_idx=idx,
                 w_kd=None if sw is None else sw[0], w_ce=None if sw is None else sw[1],
                 sem_t=sem_t, sem_valid=sem_valid)


# --------------------------------------------------------------------------------------
# モデル
# --------------------------------------------------------------------------------------

KD_T = 2.0
W_KD_GOLD = 0.8
W_CE_GOLD = 0.2


class Student:
    def __init__(self, be, cfg, max_batch, max_k, train=True, max_lp=None, max_lc=None, max_sem=0):
        self.be = be
        self.cfg = cfg
        self.train = train
        self.sem_lambda = 0.0   # Candidate Semantic Distillation の重み λ。0 なら追加 forward/backward を一切行わない
        self.sem_ext = None     # 独立ミニバッチの意味表現 pass（student/semfix.IndepSem）。None なら従来経路
        self.Bm, self.Km = int(max_batch), int(max_k)
        self.Lp = int(max_lp or cfg.lp)
        self.Lc = int(max_lc or cfg.lc)
        self.Nm = self.Bm * self.Km
        # 候補側 buffer の行数。max_sem（独立ミニバッチの候補数）が Nm を超えるときだけ Nm より大きい（既定 max_sem=0 では Nc == Nm で従来と同一）
        self.Nc = max(self.Nm, int(max_sem or 0))
        H, E, L = cfg.hidden, cfg.emb, cfg.layers
        Bm, Nm, Lp, Lc, Nc = self.Bm, self.Nm, self.Lp, self.Lc, self.Nc
        self.specs = param_specs(cfg)
        self.n_params = int(sum(int(np.prod(s)) for _, s in self.specs))
        self.P = be.alloc((self.n_params,))
        self.p = {}
        self.g = {}
        off = 0
        if train:
            self.G = be.alloc((self.n_params,))
            self.M = be.alloc((self.n_params,))
            self.V = be.alloc((self.n_params,))
        for name, shape in self.specs:
            n = int(np.prod(shape))
            self.p[name] = be.view(self.P, off, shape)
            if train:
                self.g[name] = be.view(self.G, off, shape)
            off += n

        # 入力 buffer
        self.ibuf = be.alloc((Lp * Bm + Lc * Nm + Bm + Nm + Bm + Bm + Nm,), "i")
        self.fbuf = be.alloc((Nm,))
        self.wbuf = be.alloc((2 * Bm,))  # sample ごとの損失重み [w_kd(Bm) w_ce(Bm)]（Batch.w_kd があるときだけ使う）
        # 活性 buffer。推論時は層間で gi/gh/hs を共有してメモリを抑える
        nl = L if train else 1
        self.X0p = be.alloc((Lp * Bm * E,))
        self.X0c = be.alloc((Lc * Nc * E,))
        self.gi_p = [be.alloc((Lp * Bm * 3 * H,)) for _ in range(nl)]
        self.gh_p = [be.alloc((Lp * Bm * 3 * H,)) for _ in range(nl)]
        self.gi_c = [be.alloc((Lc * Nc * 3 * H,)) for _ in range(nl)]
        self.gh_c = [be.alloc((Lc * Nc * 3 * H,)) for _ in range(nl)]
        nh = L if train else min(2, L)
        self.hs_p = [be.alloc(((Lp + 1) * Bm * H,)) for _ in range(nh)]
        self.hs_c = [be.alloc(((Lc + 1) * Nc * H,)) for _ in range(nh)]
        self.hfin = [be.alloc((Bm * H,)) for _ in range(L)]
        self.hz = be.alloc((Nm * H,))
        self.sc = be.alloc((Nm,))
        self.losses = be.alloc((Bm * 3,))
        self.ss = be.alloc((1,))
        if train:
            D = max(E, H)
            self.dgi_p = be.alloc((Lp * Bm * 3 * H,))
            self.dgh_p = be.alloc((Lp * Bm * 3 * H,))
            self.dgi_c = be.alloc((Lc * Nc * 3 * H,))
            self.dgh_c = be.alloc((Lc * Nc * 3 * H,))
            self.dxp = [be.alloc((Lp * Bm * D,)) for _ in range(2)]
            self.dxc = [be.alloc((Lc * Nc * D,)) for _ in range(2)]
            self.dhc_c = be.alloc((Nc * H,))
            self.dfin = [be.alloc((Bm * H,)) for _ in range(L)]
            self.dz = be.alloc((Nm * H,))
            if cfg.sem_dim:
                ds = cfg.sem_dim
                self.sem_t = be.alloc((Nm * ds,))
                self.sem_w = be.alloc((Nm,))
                self.zs = be.alloc((Nc * ds,))
                self.dzs = be.alloc((Nc * ds,))
                self.sem_l = be.alloc((Nm,))
                self.sem_ib = be.alloc((Lc * Nc + Nc,), "i")   # 独立ミニバッチ（semfix）の候補 token(Tc*N) と長さ(N)
        self.dsc = be.alloc((Nm,))

    # ---- パラメータ入出力（host <-> device） ----
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

    # ---- バッチ入力 ----
    def _load(self, bt):
        be = self.be
        assert bt.B <= self.Bm and bt.K <= self.Km and bt.T <= self.Lp and bt.Tc <= self.Lc, \
            (bt.B, bt.K, bt.T, bt.Tc, self.Bm, self.Km, self.Lp, self.Lc)
        be.upload(self.ibuf, bt.ints)
        be.upload(self.fbuf, bt.tlog)
        B, K, T, Tc = bt.B, bt.K, bt.T, bt.Tc
        N = B * K
        o = 0
        v = {}
        for name, n in (("ptok", T * B), ("ctok", Tc * N), ("plen", B), ("clen", N),
                        ("kcnt", B), ("gold", B), ("rep", N)):
            v[name] = be.view(self.ibuf, o, (n,))
            o += n
        return v

    # ---- forward ----
    def _forward(self, bt, iv):
        be, cfg = self.be, self.cfg
        H, E, L = cfg.hidden, cfg.emb, cfg.layers
        B, K, T, Tc = bt.B, bt.K, bt.T, bt.Tc
        N = B * K
        v = be.view
        self._acts = {}
        # prefix: 全時刻の埋め込み
        Xin = v(self.X0p, 0, (T * B, E))
        be.gather_rows(Xin, self.p["emb"], iv["ptok"])
        for l in range(L):
            tl = l if self.train else 0
            th = l if self.train else l % 2
            wi, wh = self.p[f"l{l}.wi"], self.p[f"l{l}.wh"]
            bi, bh = self.p[f"l{l}.bi"], self.p[f"l{l}.bh"]
            gi_all = v(self.gi_p[tl], 0, (T * B, 3 * H))
            be.gemm(Xin, wi, gi_all)
            hs = self.hs_p[th]
            be.zero(v(hs, 0, (B, H)))  # h0=0（B が変わると前バッチの残骸が block0 に残るため毎回消す）
            for t in range(T):
                hp = v(hs, t * B * H, (B, H))
                ght = v(self.gh_p[tl], t * B * 3 * H, (B, 3 * H))
                be.gemm(hp, wh, ght)
                be.gru_fwd(v(self.gi_p[tl], t * B * 3 * H, (B, 3 * H)), ght, bi, bh, hp,
                           v(hs, (t + 1) * B * H, (B, H)), iv["plen"], t)
            be.copy(v(self.hfin[l], 0, (B, H)), v(hs, T * B * H, (B, H)))
            Xin = v(hs, B * H, (T * B, H))
        # candidate: prefix 最終 state から継続
        Xin = v(self.X0c, 0, (Tc * N, E))
        be.gather_rows(Xin, self.p["emb"], iv["ctok"])
        for l in range(L):
            tl = l if self.train else 0
            th = l if self.train else l % 2
            wi, wh = self.p[f"l{l}.wi"], self.p[f"l{l}.wh"]
            bi, bh = self.p[f"l{l}.bi"], self.p[f"l{l}.bh"]
            hs = self.hs_c[th]
            be.gather_rows(v(hs, 0, (N, H)), v(self.hfin[l], 0, (B, H)), iv["rep"])
            gi_all = v(self.gi_c[tl], 0, (Tc * N, 3 * H))
            be.gemm(Xin, wi, gi_all)
            for t in range(Tc):
                hp = v(hs, t * N * H, (N, H))
                ght = v(self.gh_c[tl], t * N * 3 * H, (N, 3 * H))
                be.gemm(hp, wh, ght)
                be.gru_fwd(v(self.gi_c[tl], t * N * 3 * H, (N, 3 * H)), ght, bi, bh, hp,
                           v(hs, (t + 1) * N * H, (N, H)), iv["clen"], t)
            Xin = v(hs, N * H, (Tc * N, H))
            last_hs = hs
        hlast = v(last_hs, Tc * N * H, (N, H))
        z = v(self.hz, 0, (N, H))
        be.gemm(hlast, self.p["head.w1"], z)
        be.bias_act(z, self.p["head.b1"], 1)
        sc = v(self.sc, 0, (N, 1))
        be.gemm(z, self.p["head.w2"], sc)
        be.bias_act(sc, self.p["head.b2"], 0)
        return hlast, z, sc

    # ---- backward（forward の活性を使う。train=True 必須） ----
    def _backward(self, bt, iv, hlast, z, dsc):
        be, cfg = self.be, self.cfg
        H, E, L = cfg.hidden, cfg.emb, cfg.layers
        B, K, T, Tc = bt.B, bt.K, bt.T, bt.Tc
        N = B * K
        v = be.view
        g, p = self.g, self.p
        # head
        be.gemm(dsc, p["head.w2"], v(self.dz, 0, (N, H)), transB=True)
        dz = v(self.dz, 0, (N, H))
        be.gemm(z, dsc, g["head.w2"], transA=True, beta=1.0)
        be.colsum(g["head.b2"], dsc)
        be.tanh_bwd(dz, z)
        be.gemm(hlast, dz, g["head.w1"], transA=True, beta=1.0)
        be.colsum(g["head.b1"], dz)
        dhc = v(self.dhc_c, 0, (N, H))
        be.gemm(dz, p["head.w1"], dhc, transB=True)

        # 候補部 BPTT（上位層 -> 下位層）
        self._bptt_cand(iv, B, K, Tc, dhc, to_prefix=True)

        # prefix 部 BPTT
        cur = None
        for l in reversed(range(L)):
            din = E if l == 0 else H
            wi, wh = p[f"l{l}.wi"], p[f"l{l}.wh"]
            bi, bh = p[f"l{l}.bi"], p[f"l{l}.bh"]
            dh = v(self.dfin[l], 0, (B, H))
            dgi_all = v(self.dgi_p, 0, (T * B, 3 * H))
            dgh_all = v(self.dgh_p, 0, (T * B, 3 * H))
            for t in reversed(range(T)):
                dout = None if cur is None else v(cur, t * B * H, (B, H))
                be.gru_bwd(v(self.gi_p[l], t * B * 3 * H, (B, 3 * H)),
                           v(self.gh_p[l], t * B * 3 * H, (B, 3 * H)), bi, bh,
                           v(self.hs_p[l], t * B * H, (B, H)), dh, dout,
                           v(self.dgi_p, t * B * 3 * H, (B, 3 * H)),
                           v(self.dgh_p, t * B * 3 * H, (B, 3 * H)), iv["plen"], t)
                be.gemm(v(self.dgh_p, t * B * 3 * H, (B, 3 * H)), wh, dh, transB=True, beta=1.0)
            if l == 0:
                Xl = v(self.X0p, 0, (T * B, E))
            else:
                Xl = v(self.hs_p[l - 1], B * H, (T * B, H))
            be.gemm(Xl, dgi_all, g[f"l{l}.wi"], transA=True, beta=1.0)
            be.gemm(v(self.hs_p[l], 0, (T * B, H)), dgh_all, g[f"l{l}.wh"], transA=True, beta=1.0)
            be.colsum(g[f"l{l}.bi"], dgi_all)
            be.colsum(g[f"l{l}.bh"], dgh_all)
            nxt = self.dxp[0] if cur is not self.dxp[0] else self.dxp[1]
            dx = v(nxt, 0, (T * B, din))
            be.gemm(dgi_all, wi, dx, transB=True)
            if l == 0:
                be.scatter_add_rows(g["emb"], dx, iv["ptok"])
            cur = nxt

    def _bptt_cand(self, iv, B, K, Tc, dhc, to_prefix):
        """候補部 BPTT（上位層 -> 下位層）。dhc (N,H) = 最終層の最終 hidden への勾配（in/out 兼用の carry buffer）。
        勾配は self.G に累積。to_prefix=True なら各層の初期 state への勾配を group_sum して dfin に書く（prefix 部 BPTT の入力）。
        False（Candidate Semantic Distillation: 初期 hidden=0 固定）なら初期 state への勾配は捨てる。"""
        be, cfg = self.be, self.cfg
        H, E, L = cfg.hidden, cfg.emb, cfg.layers
        N = B * K
        v = be.view
        g, p = self.g, self.p
        cur = None
        for l in reversed(range(L)):
            din = E if l == 0 else H
            wi, wh = p[f"l{l}.wi"], p[f"l{l}.wh"]
            bi, bh = p[f"l{l}.bi"], p[f"l{l}.bh"]
            if l != L - 1:
                be.zero(dhc)
            dgi_all = v(self.dgi_c, 0, (Tc * N, 3 * H))
            dgh_all = v(self.dgh_c, 0, (Tc * N, 3 * H))
            for t in reversed(range(Tc)):
                dout = None if cur is None else v(cur, t * N * H, (N, H))
                be.gru_bwd(v(self.gi_c[l], t * N * 3 * H, (N, 3 * H)),
                           v(self.gh_c[l], t * N * 3 * H, (N, 3 * H)), bi, bh,
                           v(self.hs_c[l], t * N * H, (N, H)), dhc, dout,
                           v(self.dgi_c, t * N * 3 * H, (N, 3 * H)),
                           v(self.dgh_c, t * N * 3 * H, (N, 3 * H)), iv["clen"], t)
                be.gemm(v(self.dgh_c, t * N * 3 * H, (N, 3 * H)), wh, dhc, transB=True, beta=1.0)
            if l == 0:
                Xl = v(self.X0c, 0, (Tc * N, E))
            else:
                Xl = v(self.hs_c[l - 1], N * H, (Tc * N, H))
            be.gemm(Xl, dgi_all, g[f"l{l}.wi"], transA=True, beta=1.0)
            be.gemm(v(self.hs_c[l], 0, (Tc * N, H)), dgh_all, g[f"l{l}.wh"], transA=True, beta=1.0)
            be.colsum(g[f"l{l}.bi"], dgi_all)
            be.colsum(g[f"l{l}.bh"], dgh_all)
            nxt = self.dxc[0] if cur is not self.dxc[0] else self.dxc[1]
            dx = v(nxt, 0, (Tc * N, din))
            be.gemm(dgi_all, wi, dx, transB=True)
            if l == 0:
                be.scatter_add_rows(g["emb"], dx, iv["ctok"])
            cur = nxt
            if to_prefix:
                be.group_sum(v(self.dfin[l], 0, (B, H)), dhc, K)


    # ---- Candidate Semantic Distillation（学習専用の追加 pass） ----
    def _sem_pass(self, bt, iv, want_grads):
        """候補を「文脈なし（全層の初期 hidden = 0）」で同じ GRU（重み共有）に通し、最終層の最終 hidden h_c を
        projection head Linear(H, d) で z_c に写して、Teacher の PCA 済み embedding t_c との cosine loss
        mean_valid(1 - cos(z_c, t_c)) を取る。戻り値: その平均（float）。勾配は self.G に累積（λ 倍済み）。
        主 pass の backward 完了後に呼ぶ（candidate 用 activation buffer を使い回す。X0c は主 forward の埋め込みがそのまま有効）。
        判断用 score の経路（predict / 主 forward）には一切触れない。"""
        be, cfg = self.be, self.cfg
        H, E, L, d = cfg.hidden, cfg.emb, cfg.layers, cfg.sem_dim
        B, K, Tc = bt.B, bt.K, bt.Tc
        N = B * K
        v = be.view
        p, g = self.p, self.g
        valid = bt.sem_valid
        nv = int(valid.sum())
        if nv == 0:
            return 0.0
        w = np.where(valid, self.sem_lambda / nv, 0.0)
        be.upload(self.sem_t, bt.sem_t.reshape(-1))
        be.upload(self.sem_w, w)
        Xin = v(self.X0c, 0, (Tc * N, E))
        for l in range(L):
            wi, wh = p[f"l{l}.wi"], p[f"l{l}.wh"]
            bi, bh = p[f"l{l}.bi"], p[f"l{l}.bh"]
            hs = self.hs_c[l]
            be.zero(v(hs, 0, (N, H)))   # 初期 hidden = 0（文脈なし）
            gi_all = v(self.gi_c[l], 0, (Tc * N, 3 * H))
            be.gemm(Xin, wi, gi_all)
            for t in range(Tc):
                hp = v(hs, t * N * H, (N, H))
                ght = v(self.gh_c[l], t * N * 3 * H, (N, 3 * H))
                be.gemm(hp, wh, ght)
                be.gru_fwd(v(self.gi_c[l], t * N * 3 * H, (N, 3 * H)), ght, bi, bh, hp,
                           v(hs, (t + 1) * N * H, (N, H)), iv["clen"], t)
            Xin = v(hs, N * H, (Tc * N, H))
        hlast = v(self.hs_c[L - 1], Tc * N * H, (N, H))
        zs = v(self.zs, 0, (N, d))
        be.gemm(hlast, p["sem.wp"], zs)
        be.bias_act(zs, p["sem.bp"], 0)
        dzs = v(self.dzs, 0, (N, d))
        be.cos_loss(zs, v(self.sem_t, 0, (N, d)), v(self.sem_w, 0, (N,)), dzs, v(self.sem_l, 0, (N,)), N, d)
        l_rows = be.download(v(self.sem_l, 0, (N,))).astype(np.float64)
        if want_grads:
            be.gemm(hlast, dzs, g["sem.wp"], transA=True, beta=1.0)
            be.colsum(g["sem.bp"], dzs)
            dhc = v(self.dhc_c, 0, (N, H))
            be.gemm(dzs, p["sem.wp"], dhc, transB=True)
            self._bptt_cand(iv, B, K, Tc, dhc, to_prefix=False)
        return float(l_rows.sum() / nv)

    # ---- 公開 API ----
    def predict(self, bt):
        """forward のみ。host の scores (B,K) を返す（pad 候補は無意味な値）。"""
        iv = self._load(bt)
        self._forward(bt, iv)
        out = self.be.download(self.be.view(self.sc, 0, (bt.B * bt.K,)))
        return out.reshape(bt.B, bt.K).astype(np.float64)

    def loss_grads(self, bt, want_grads=True):
        """forward(+backward)。勾配は self.G に累積（先に zero される）。
        戻り値: dict(loss, kd, ce, n_gold, sem)。loss は batch 平均（KD は T^2 込み）+ λ*sem（sem_lambda>0 のとき）。
        sem = 候補 cosine loss の平均（sem_lambda=0 なら 0.0 で、追加 forward は行わない）。"""
        be = self.be
        B, K = bt.B, bt.K
        N = B * K
        iv = self._load(bt)
        if want_grads:
            be.zero(self.G)
        hlast, z, sc = self._forward(bt, iv)
        scf = be.view(self.sc, 0, (N,))
        dsc = be.view(self.dsc, 0, (N, 1))
        dscf = be.view(self.dsc, 0, (N,))
        if bt.w_kd is None:
            w_kd, w_ce = W_KD_GOLD, W_CE_GOLD  # 従来どおりスカラー（source 別重み無しではビット一致）
        else:
            w_kd, w_ce = be.view(self.wbuf, 0, (B,)), be.view(self.wbuf, self.Bm, (B,))
            be.upload(w_kd, bt.w_kd)
            be.upload(w_ce, bt.w_ce)
        be.kd_loss(scf, be.view(self.fbuf, 0, (N,)), iv["kcnt"], iv["gold"], dscf,
                   be.view(self.losses, 0, (B * 3,)), B, K, KD_T, w_kd, w_ce, 1.0 / B)
        if want_grads:
            self._backward(bt, iv, hlast, z, dsc)
        sem = None
        if self.sem_lambda > 0 and self.sem_ext is not None:
            if not (self.train and self.cfg.sem_dim):
                raise ValueError("sem_ext には train=True かつ cfg.sem_dim>0 が必要")
            sem = self.sem_ext.run(self, want_grads)   # 独立ミニバッチ（判断バッチの候補は使わない）
        elif self.sem_lambda > 0:
            if not (self.train and self.cfg.sem_dim):
                raise ValueError("sem_lambda>0 には train=True かつ cfg.sem_dim>0 が必要")
            if bt.sem_t is None or bt.sem_t.shape[1] != self.cfg.sem_dim:
                raise ValueError("sem_lambda>0 なのに batch に sem target が無い（make_batch(sem=...)）/ 次元不一致")
            sem = self._sem_pass(bt, iv, want_grads)
        L = be.download(be.view(self.losses, 0, (B * 3,))).reshape(B, 3).astype(np.float64)
        has = bt.gold >= 0
        out = {"loss": float(L[:, 0].mean()), "kd": float(L[:, 1].mean()),
               "ce": float(L[has, 2].mean()) if has.any() else 0.0, "n_gold": int(has.sum())}
        if sem is not None:
            out["loss"] += self.sem_lambda * sem   # L_total = L_KD/CE + λ * L_cand_sem
        out["sem"] = 0.0 if sem is None else sem
        return out

    def train_step(self, bt, lr, step, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, clip=1.0):
        """1 step: forward + backward + grad norm + clip + AdamW。step は 1 始まり。"""
        be = self.be
        st = self.loss_grads(bt, want_grads=True)
        be.sumsq(self.G, self.ss)
        be.adamw(self.P, self.G, self.M, self.V, self.ss, lr, beta1, beta2, eps, wd, step, clip)
        ss = float(be.download(self.ss).reshape(-1)[0])
        st["gnorm"] = float(np.sqrt(ss))
        return st


def make_backend(name, device=None, dtype="float32"):
    """backend 名（np|cl）から backend を作る。cl はデバイス名の部分一致（例 "GT 430"）。"""
    if name == "np":
        from .backend_np import NPBackend
        return NPBackend(np.dtype(dtype))
    if name == "cl":
        from .backend_cl import CLBackend
        return CLBackend(device or "GT 430")
    raise ValueError(f"unknown backend {name!r}")
