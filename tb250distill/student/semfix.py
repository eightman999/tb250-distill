"""候補意味表現蒸留の修正（SEMFIX）: 独立ミニバッチ + 損失の種類（cos / mse / rkd / infonce とその組合せ）。

従来（`--sem-cand-weight` のみ）は判断バッチ内の候補（重複が多く約 100 件）にだけ cos loss をかけていた。
ここでは判断バッチとは別に、train の unique 候補文字列集合（`--sem-emb` で有効な位置のもの）から M 個を毎 step 取り出し
（`--sem-batch M`、source 層別・seed 決定的）、文脈なし（全層の初期 hidden=0）の候補 pass -> projection head(H->d) -> 損失、を計算する。

損失（`--sem-loss`、'+' で組合せ、重みは `--sem-loss-weights`。既定 全て 1.0）:
  cos      mean_i (1 - cos(z_i, t_i))                        t = Teacher PCA 埋め込み
  mse      mean_i |zhat_i - that_i|^2（L2 正規化後。= 2*(1-cos) で cos と同値の方向・2 倍のスケール）
  rkd      類似度行列蒸留: 行ごとに KL( softmax_j(St_ij/tau_t) || softmax_j(Ss_ij/tau_s) )（j は自分と同一文字列を除く）。
           Ss = Student z の cosine 行列、St = Teacher 埋め込み（train unique 集合の平均で中心化 + L2 正規化。
           `--sem-rkd-space raw` = 元の 1024 次元（既定）/ `pca` = PCA128 を同様に中心化）。共通成分でなく相対関係を学ばせる
  infonce  クロスモーダル対照学習: logits_ij = cos(z_i, t_j)/tau、正例は対角、バッチ内の他を負例（同一文字列の重複は除外）。
           既定は対称（Student->Teacher と Teacher->Student の平均）。t は `--sem-nce-target pca`（PCA128）/ `pca_c`（同、プール平均で中心化）/
           `rproj`（中心化 1024 次元 -> d への固定乱数射影。学習しない）

全ての損失・勾配は float64 の numpy（host）で計算する（M x M 行列は小さい）。Student 側（np / cl backend）は z を download し、
dL/dz を upload して head -> GRU の backward を通す。よって np / cl の違いは z の forward だけで、損失の数値勾配チェックは host 関数単体でできる。
λ（`--sem-cand-weight`）は全体にかかる（loss_total = L_judge + λ * L_sem）。`--sem-batch 0`（既定）は従来経路（判断バッチ内の cos、cl では kernel）で、
このファイルは何も呼ばれない（ビット一致）。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

LOSS_NAMES = ("cos", "mse", "rkd", "infonce")
EPS = 1e-12          # backend_np.COS_EPS と同じ（norm = sqrt(|z|^2 + EPS)）


# --------------------------------------------------------------------------------------
# 損失仕様
# --------------------------------------------------------------------------------------

@dataclass
class LossSpec:
    terms: dict = field(default_factory=lambda: {"cos": 1.0})   # 名前 -> 重み（挿入順）
    tau_nce: float = 0.07
    tau_t: float = 0.1
    tau_s: float = 0.1
    rkd_space: str = "raw"       # raw | pca
    nce_target: str = "pca"      # pca | pca_c | rproj
    nce_sym: bool = True

    def describe(self):
        return {"terms": dict(self.terms), "tau_nce": self.tau_nce, "tau_t": self.tau_t, "tau_s": self.tau_s,
                "rkd_space": self.rkd_space, "nce_target": self.nce_target, "nce_sym": self.nce_sym}


def parse_loss(loss, weights=None, **kw):
    """'infonce+rkd' と 'infonce=1,rkd=0.5'（無ければ全て 1.0）から LossSpec を作る。"""
    names = [x.strip() for x in str(loss).split("+") if x.strip()]
    if not names:
        raise ValueError("--sem-loss が空")
    for n in names:
        if n not in LOSS_NAMES:
            raise ValueError(f"--sem-loss: 未知の損失 '{n}'（{'/'.join(LOSS_NAMES)} を '+' で組合せる）")
    if len(set(names)) != len(names):
        raise ValueError("--sem-loss: 同じ損失を重複指定している")
    w = {n: 1.0 for n in names}
    if weights:
        for part in str(weights).split(","):
            k, eq, v = part.partition("=")
            k = k.strip()
            if not eq or k not in w:
                raise ValueError(f"--sem-loss-weights: '{part}' は 'name=重み' で、name は --sem-loss の損失のいずれか")
            try:
                w[k] = float(v)
            except ValueError:
                raise ValueError(f"--sem-loss-weights: '{part}' の重みが数値でない") from None
            if not (math.isfinite(w[k]) and w[k] >= 0):
                raise ValueError(f"--sem-loss-weights: '{part}' は有限の非負数")
    spec = LossSpec(terms=w, **kw)
    if spec.rkd_space not in ("raw", "pca"):
        raise ValueError("--sem-rkd-space は raw|pca")
    if spec.nce_target not in ("pca", "pca_c", "rproj"):
        raise ValueError("--sem-nce-target は pca|pca_c|rproj")
    for k in ("tau_nce", "tau_t", "tau_s"):
        if not (math.isfinite(getattr(spec, k)) and getattr(spec, k) > 0):
            raise ValueError(f"--sem-{k.replace('_', '-')} は正の有限値")
    return spec


# --------------------------------------------------------------------------------------
# 損失関数（float64 numpy、(loss, dL/dz) を返す）
# --------------------------------------------------------------------------------------

def _unit(x):
    """行ごとの L2 正規化。norm = sqrt(|x|^2 + EPS)（cos_loss と同じ）。戻り値 (xhat, norm)。"""
    x = np.asarray(x, np.float64)
    n = np.sqrt((x * x).sum(1) + EPS)
    return x / n[:, None], n


def _unit_back(dzh, zh, nz):
    """zhat = z / sqrt(|z|^2+EPS) の逆伝播: dz = (dzhat - zhat (zhat . dzhat)) / norm。"""
    return (dzh - (dzh * zh).sum(1, keepdims=True) * zh) / nz[:, None]


def loss_cos(z, t):
    """mean_i (1 - cos(z_i, t_i))。t の正規化は定数扱い（勾配なし）。"""
    zh, nz = _unit(z)
    th, _ = _unit(t)
    c = (zh * th).sum(1)
    N = z.shape[0]
    dzh = -th / N
    return float((1.0 - c).mean()), _unit_back(dzh, zh, nz)


def loss_mse(z, t):
    """mean_i |zhat_i - that_i|^2（L2 正規化後の二乗誤差の次元和。= 2(1-cos)）。"""
    zh, nz = _unit(z)
    th, _ = _unit(t)
    d = zh - th
    N = z.shape[0]
    return float((d * d).sum(1).mean()), _unit_back(2.0 * d / N, zh, nz)


def _log_softmax_masked(logits, mask):
    """行方向。mask=False の位置は確率 0。全行が mask 全 False なら 0 行。戻り値 (logp, p)（マスク位置は logp=0, p=0）。"""
    x = np.where(mask, logits, -np.inf)
    mx = np.max(np.where(mask, x, -np.inf), axis=1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    e = np.where(mask, np.exp(np.where(mask, x, 0.0) - mx), 0.0)
    s = e.sum(1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    p = e / s
    logp = np.where(mask, np.where(mask, x, 0.0) - mx - np.log(s), 0.0)
    return logp, p


def loss_rkd(z, e, excl, tau_t, tau_s):
    """類似度行列蒸留。z (N,d) Student、e (N,D) Teacher（呼び出し側で中心化・L2 正規化済みの単位行）、
    excl (N,N) bool: True の (i,j) は除外（対角と同一文字列）。行ごとの KL(P_t || P_s) の有効行平均。"""
    zh, nz = _unit(z)
    e = np.asarray(e, np.float64)
    N = z.shape[0]
    mask = ~np.asarray(excl, bool)
    row_ok = mask.any(1)
    nrow = int(row_ok.sum())
    if nrow == 0:
        return 0.0, np.zeros_like(zh)
    Ss = zh @ zh.T
    St = e @ e.T
    logp_t, pt = _log_softmax_masked(St / tau_t, mask)
    logp_s, ps = _log_softmax_masked(Ss / tau_s, mask)
    kl = np.where(mask, pt * (logp_t - logp_s), 0.0).sum(1)
    loss = float(kl[row_ok].sum() / nrow)
    G = (ps - pt) / tau_s * (row_ok / nrow)[:, None]          # dL/dSs（非対称。Ss は対称なので G + G^T を使う）
    dzh = (G + G.T) @ zh
    return loss, _unit_back(dzh, zh, nz)


def loss_infonce(z, t, uidx, tau, sym=True):
    """クロスモーダル InfoNCE。z (N,d)、t (N,d) の行 j が z の行 j の正例。logits_ij = cos(z_i, t_j)/tau。
    同一文字列（uidx が等しい）の非対角は負例から除外する。sym なら (Student->Teacher + Teacher->Student)/2。"""
    zh, nz = _unit(z)
    th, _ = _unit(t)
    N = z.shape[0]
    u = np.asarray(uidx)
    mask = ~((u[:, None] == u[None, :]) & ~np.eye(N, dtype=bool))
    L = zh @ th.T / tau
    lp_r, p_r = _log_softmax_masked(L, mask)               # 行（Student -> Teacher: 各 z_i に対し正例 t_i を選ぶ）
    I = np.eye(N)
    loss_r = -float(np.diag(lp_r).mean())
    gr = (p_r - I) / N
    if sym:
        lp_c, p_c = _log_softmax_masked(L.T, mask.T)         # 列（Teacher -> Student: 各 t_j に対し正例 z_j を選ぶ）
        loss_c = -float(np.diag(lp_c).mean())
        gc = (p_c.T - I) / N
        loss = 0.5 * (loss_r + loss_c)
        g = 0.5 * (gr + gc)
    else:
        loss, g = loss_r, gr
    dzh = (g / tau) @ th
    return float(loss), _unit_back(dzh, zh, nz)


def compute_loss(spec, z, tg, uidx):
    """z (N,d) float64、tg = dict(pca=(N,d), rkd=(N,D), nce=(N,d))（行は z と対応）、uidx (N) 文字列 id。
    戻り値 (total, dz, parts)。parts = 各項の（重み前の）値。"""
    z = np.asarray(z, np.float64)
    N = z.shape[0]
    total = 0.0
    dz = np.zeros_like(z)
    parts = {}
    u = np.asarray(uidx)
    for name, w in spec.terms.items():
        if name == "cos":
            l, g = loss_cos(z, tg["pca"])
        elif name == "mse":
            l, g = loss_mse(z, tg["pca"])
        elif name == "rkd":
            l, g = loss_rkd(z, tg["rkd"], u[:, None] == u[None, :], spec.tau_t, spec.tau_s)
        elif name == "infonce":
            l, g = loss_infonce(z, tg["nce"], u, spec.tau_nce, spec.nce_sym)
        else:  # pragma: no cover
            raise ValueError(name)
        parts[name] = l
        total += w * l
        dz += w * g
    return float(total), dz, parts


# --------------------------------------------------------------------------------------
# 候補文字列プールと層別サンプラ
# --------------------------------------------------------------------------------------

class SemPool:
    """train の unique 候補文字列（sem.cand_idx の有効位置にあるもの）。1 文字列 = 初出の (行, 位置) の token 列・長さ・source。
    uid = SemTargets.emb の行番号（unique 文字列 index）。"""

    def __init__(self, tr, sem):
        ci = sem.cand_idx
        rows, poss = np.nonzero(ci >= 0)
        u = ci[rows, poss].astype(np.int64)
        uniq, first = np.unique(u, return_index=True)
        r, p = rows[first], poss[first]
        ln = tr.cand_len[r, p].astype(np.int32)
        keep = ln > 0
        self.uid = uniq[keep]
        r, p, ln = r[keep], p[keep], ln[keep]
        self.tok = np.ascontiguousarray(tr.cand[r, p, :], np.int32)
        self.len = ln
        src = tr.extra.get("source") if hasattr(tr, "extra") else None
        self.source = np.asarray(src)[r].astype(str) if src is not None and getattr(src, "shape", None) == (tr.n,) \
            else np.full(len(self.uid), "all")
        self.n = int(len(self.uid))

    def counts(self):
        s, c = np.unique(self.source, return_counts=True)
        return {str(a): int(b) for a, b in zip(s, c)}


def _largest_remainder(M, weights):
    """重み（dict name->非負）から合計 M の整数 quota（最大剰余法、同順位は名前順）。"""
    tot = float(sum(weights.values()))
    if tot <= 0:
        raise ValueError("mix の重みの合計が 0")
    names = sorted(weights)
    raw = {n: M * weights[n] / tot for n in names}
    q = {n: int(math.floor(raw[n])) for n in names}
    rest = M - sum(q.values())
    for n in sorted(names, key=lambda n: (-(raw[n] - q[n]), n))[:rest]:
        q[n] += 1
    return q


class SemSampler:
    """step（0 始まり）だけから M 個の pool index を決める（状態を持たない: --resume で厳密に再現）。
    mix:
      'equal'    source ごとに等量（M を source 数で割る。端数は最大剰余法）。各 source は seed 固定の置換を順に消費（1 周するまで重複なし）
      'uniform'  pool 全体から一様（step ごとに非復元抽出）
      'a:0.5,b:0.25'  source ごとの比率（未指定の source は 0）
      'items:B'  （sem-only のスクリーニング専用）従来相当: train の item を B 件（train.py と同じ epoch_order）取り、その判断バッチに出る
                 候補を全て使う（重複あり・件数は可変で最大 B*kmax）。cand_idx が必要"""

    def __init__(self, pool, M, seed, mix="equal", cand_idx=None, n_items=None):
        self.pool, self.M, self.seed, self.mix = pool, int(M), int(seed), mix
        self.max_n = self.M
        self.items = None
        if str(mix).startswith("items:"):
            if cand_idx is None:
                raise ValueError("--sem-batch-mix items:B は sem-only 専用（cand_idx が必要）")
            self.items = int(str(mix).split(":", 1)[1])
            self.cand_idx = np.asarray(cand_idx)
            self.n_items = self.cand_idx.shape[0]
            self.max_n = self.items * self.cand_idx.shape[1]
            self.u2p = np.full(int(self.cand_idx.max()) + 1, -1, np.int64)
            self.u2p[pool.uid] = np.arange(pool.n)
            self.spe = -(-self.n_items // self.items)
            self.groups, self.quota = {}, None
            self._perm_cache = {}
            return
        if self.M < 2:
            raise ValueError("--sem-batch は 2 以上")
        self.groups = {}
        for s in sorted(set(pool.source.tolist())):
            self.groups[s] = np.nonzero(pool.source == s)[0]
        if mix == "uniform":
            self.quota = None
        else:
            if mix == "equal":
                w = {s: 1.0 for s in self.groups}
            else:
                w = {}
                for part in str(mix).split(","):
                    k, sep, v = part.partition(":")
                    k = k.strip()
                    if not sep or k not in self.groups:
                        raise ValueError(f"--sem-batch-mix: '{part}' は 'source:比率' で、source は {sorted(self.groups)} のいずれか")
                    w[k] = float(v)
            self.quota = _largest_remainder(self.M, {k: v for k, v in w.items() if v > 0})
        self._perm_cache = {}

    def _perm(self, name, epoch, n):
        key = (name, epoch)
        if key not in self._perm_cache:
            if len(self._perm_cache) > 64:
                self._perm_cache.clear()
            si = sorted(self.groups).index(name)
            self._perm_cache[key] = np.random.default_rng([self.seed, epoch, si, 0x53424D]).permutation(n)
        return self._perm_cache[key]

    def indices(self, step):
        if self.items is not None:
            ep, pos = divmod(int(step), self.spe)
            if ("items", ep) not in self._perm_cache:
                self._perm_cache.clear()
                self._perm_cache[("items", ep)] = np.random.default_rng([self.seed, ep, 1]).permutation(self.n_items)
            rows = self._perm_cache[("items", ep)][pos * self.items:(pos + 1) * self.items]
            u = self.cand_idx[rows].reshape(-1)
            u = u[u >= 0]
            p = self.u2p[u]
            return p[p >= 0]
        if self.quota is None:
            rng = np.random.default_rng([self.seed, int(step), 0x53424D])
            return np.sort(rng.choice(self.pool.n, size=min(self.M, self.pool.n), replace=False))
        out = []
        for name in sorted(self.quota):
            q = self.quota[name]
            g = self.groups[name]
            n = len(g)
            for pos in range(int(step) * q, int(step) * q + q):
                ep, off = divmod(pos, n)
                out.append(g[self._perm(name, ep, n)[off]])
        return np.array(out, np.int64)

    def describe(self):
        return {"M": self.M, "max_n": self.max_n, "mix": self.mix, "quota": self.quota, "seed": self.seed,
                "pool_by_source": self.pool.counts(), "pool_size": self.pool.n}


# --------------------------------------------------------------------------------------
# Teacher 側 target（プール行に対応）
# --------------------------------------------------------------------------------------

def _unit_rows(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class PoolTargets:
    """プールの各文字列の Teacher target。float32 の単位行（cos/mse は pca 行を内部で正規化するので生のまま pca も保持）。
      pca  = SemTargets.emb（PCA 済み、中心化はプール外を含む全 unique 集合の平均）
      rkd  = rkd_space が raw なら 1024 次元（L2 正規化 -> プール平均で中心化 -> L2 正規化）、pca ならその PCA 版（プール平均で再中心化）
      nce  = nce_target（pca / pca_c / rproj）"""

    def __init__(self, pool, sem, spec, emb_dir, seed=0):
        self.pca = sem.emb[pool.uid]
        need_raw = ("rkd" in spec.terms and spec.rkd_space == "raw") or ("infonce" in spec.terms and spec.nce_target == "rproj")
        raw_c = None
        if need_raw:
            from .model import assert_not_eval_only_dir
            assert_not_eval_only_dir(emb_dir)
            raw = np.load(os.path.join(emb_dir, "emb_raw.npy"), mmap_mode="r")
            if raw.shape[0] != sem.emb.shape[0]:
                raise ValueError(f"{emb_dir}/emb_raw.npy の行数 {raw.shape[0]} != emb_pca の {sem.emb.shape[0]}")
            x = _unit_rows(np.asarray(raw[pool.uid]))
            raw_c = _unit_rows(x - x.mean(0, keepdims=True))
        pca_c = _unit_rows(self.pca.astype(np.float64) - self.pca.astype(np.float64).mean(0, keepdims=True))
        self.rkd = None
        if "rkd" in spec.terms:
            self.rkd = (raw_c if spec.rkd_space == "raw" else pca_c).astype(np.float32)
        self.nce = None
        if "infonce" in spec.terms:
            if spec.nce_target == "pca":
                self.nce = _unit_rows(self.pca).astype(np.float32)
            elif spec.nce_target == "pca_c":
                self.nce = pca_c.astype(np.float32)
            else:
                d = sem.d
                R = np.random.default_rng([seed, 0x52504A]).normal(size=(raw_c.shape[1], d)) / math.sqrt(d)
                self.nce = _unit_rows(raw_c @ R).astype(np.float32)

    def rows(self, idx):
        return {"pca": self.pca[idx], "rkd": None if self.rkd is None else self.rkd[idx],
                "nce": None if self.nce is None else self.nce[idx]}


# --------------------------------------------------------------------------------------
# Student への結線
# --------------------------------------------------------------------------------------

class IndepSem:
    """独立ミニバッチの意味表現 pass（Student.sem_ext に差す）。`run(student, want_grads)` が 1 step 分の損失（λ 前）を返し、
    勾配（λ 倍）を student.G に累積する。step は呼び出し側が `self.step = <0 始まりの step>` で渡す。"""

    def __init__(self, pool, sampler, targets, spec):
        self.pool, self.sampler, self.targets, self.spec = pool, sampler, targets, spec
        self.step = 0
        self._parts_sum = {}
        self._parts_n = 0
        self.last = None

    def run(self, st, want_grads):
        be, cfg = st.be, st.cfg
        H, E, L, d = cfg.hidden, cfg.emb, cfg.layers, cfg.sem_dim
        idx = self.sampler.indices(self.step)
        N = int(len(idx))
        ln = self.pool.len[idx]
        Tc = int(max(1, ln.max()))
        if N > st.Nc or Tc > st.Lc:
            raise ValueError(f"sem-batch {N} > buffer {st.Nc} または Tc {Tc} > Lc {st.Lc}（Student(max_sem=...) を指定する）")
        v, p, g = be.view, st.p, st.g
        ctok = np.ascontiguousarray(self.pool.tok[idx][:, :Tc].T).reshape(-1)
        be.upload(st.sem_ib, np.concatenate([ctok, ln]).astype(np.int32))
        iv = {"ctok": v(st.sem_ib, 0, (Tc * N,)), "clen": v(st.sem_ib, Tc * N, (N,))}
        Xin = v(st.X0c, 0, (Tc * N, E))
        be.gather_rows(Xin, p["emb"], iv["ctok"])
        for l in range(L):
            wi, wh = p[f"l{l}.wi"], p[f"l{l}.wh"]
            bi, bh = p[f"l{l}.bi"], p[f"l{l}.bh"]
            hs = st.hs_c[l]
            be.zero(v(hs, 0, (N, H)))     # 初期 hidden = 0（文脈なし）
            gi_all = v(st.gi_c[l], 0, (Tc * N, 3 * H))
            be.gemm(Xin, wi, gi_all)
            for t in range(Tc):
                hp = v(hs, t * N * H, (N, H))
                ght = v(st.gh_c[l], t * N * 3 * H, (N, 3 * H))
                be.gemm(hp, wh, ght)
                be.gru_fwd(v(st.gi_c[l], t * N * 3 * H, (N, 3 * H)), ght, bi, bh, hp, v(hs, (t + 1) * N * H, (N, H)), iv["clen"], t)
            Xin = v(hs, N * H, (Tc * N, H))
        hlast = v(st.hs_c[L - 1], Tc * N * H, (N, H))
        zs = v(st.zs, 0, (N, d))
        be.gemm(hlast, p["sem.wp"], zs)
        be.bias_act(zs, p["sem.bp"], 0)
        z = be.download(zs).astype(np.float64)
        tg = self.targets.rows(idx)
        loss, dz, parts = compute_loss(self.spec, z, tg, self.pool.uid[idx])
        for k, x in parts.items():
            self._parts_sum[k] = self._parts_sum.get(k, 0.0) + x
        self._parts_n += 1
        self.last = {"loss": loss, "parts": parts, "N": N}
        if want_grads:
            dzs = v(st.dzs, 0, (N, d))
            be.upload(dzs, (dz * st.sem_lambda).astype(be.dtype))
            be.gemm(hlast, dzs, g["sem.wp"], transA=True, beta=1.0)
            be.colsum(g["sem.bp"], dzs)
            dhc = v(st.dhc_c, 0, (N, H))
            be.gemm(dzs, p["sem.wp"], dhc, transB=True)
            st._bptt_cand(iv, N, 1, Tc, dhc, to_prefix=False)
        return loss

    def pop_parts(self):
        """前回 pop 以降の各項の平均（重み前）。無ければ None。"""
        if not self._parts_n:
            return None
        out = {k: x / self._parts_n for k, x in self._parts_sum.items()}
        self._parts_sum, self._parts_n = {}, 0
        return out

    def describe(self):
        return {"loss": self.spec.describe(), "sampler": self.sampler.describe(), "host_float64_loss": True}


def sem_only_step(st, lr, step, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, clip=1.0):
    """判断（KD/CE）を使わず、独立ミニバッチの意味表現 loss だけで 1 step（zero grad -> sem pass -> AdamW）。step は 1 始まり。"""
    be = st.be
    if st.sem_ext is None or st.sem_lambda <= 0:
        raise ValueError("sem_only_step には sem_ext と sem_lambda>0 が必要")
    be.zero(st.G)
    st.sem_ext.step = step - 1
    sem = st.sem_ext.run(st, True)
    be.sumsq(st.G, st.ss)
    be.adamw(st.P, st.G, st.M, st.V, st.ss, lr, beta1, beta2, eps, wd, step, clip)
    ss = float(be.download(st.ss).reshape(-1)[0])
    return {"loss": st.sem_lambda * sem, "sem": sem, "gnorm": float(np.sqrt(ss)), "kd": 0.0, "ce": 0.0, "n_gold": 0}


# --------------------------------------------------------------------------------------
# CLI 引数と構築
# --------------------------------------------------------------------------------------

def add_args(ap):
    ap.add_argument("--sem-batch", type=int, default=0, metavar="M",
                    help="意味表現用の独立ミニバッチ（train の unique 候補文字列から step ごとに M 個。0=従来の判断バッチ内候補・cos のみ）。"
                         "要 --sem-cand-weight>0")
    ap.add_argument("--sem-batch-mix", default="equal",
                    help="独立ミニバッチの source 配分: equal（source 等量）/ uniform（全体から一様）/ 'massive:0.5,synth:0.25,...'")
    ap.add_argument("--sem-loss", default="cos",
                    help="意味表現の損失: cos|mse|rkd|infonce を '+' で組合せ（例 infonce+rkd）。cos 以外は --sem-batch>0 が必要")
    ap.add_argument("--sem-loss-weights", default=None, help="組合せの項の重み 'infonce=1,rkd=0.5'（既定 全て 1）")
    ap.add_argument("--sem-tau-nce", type=float, default=0.07, help="infonce の温度")
    ap.add_argument("--sem-tau-t", type=float, default=0.1, help="rkd の Teacher 側 softmax 温度")
    ap.add_argument("--sem-tau-s", type=float, default=0.1, help="rkd の Student 側 softmax 温度")
    ap.add_argument("--sem-rkd-space", default="raw", choices=["raw", "pca"], help="rkd の Teacher 空間（raw=元の 1024 次元を中心化+L2 正規化 / pca=PCA128 を中心化）")
    ap.add_argument("--sem-nce-target", default="pca", choices=["pca", "pca_c", "rproj"], help="infonce の Teacher 側 target")
    ap.add_argument("--sem-nce-oneway", action="store_true", help="infonce を Student->Teacher の片方向だけにする（既定は対称）")


def spec_from_args(args):
    return parse_loss(args.sem_loss, args.sem_loss_weights, tau_nce=args.sem_tau_nce, tau_t=args.sem_tau_t, tau_s=args.sem_tau_s,
                      rkd_space=args.sem_rkd_space, nce_target=args.sem_nce_target, nce_sym=not args.sem_nce_oneway)


def build_hook(args, tr, sem, sem_w, allow_items=False):
    """args から IndepSem（または None=従来経路）を作る。不正な組合せは ValueError。tr/sem は limit-train 適用後。"""
    spec = spec_from_args(args)
    if args.sem_batch <= 0:
        if list(spec.terms) != ["cos"]:
            raise ValueError("cos 以外の --sem-loss は --sem-batch > 0（独立ミニバッチ）が必要")
        if args.sem_loss_weights:
            raise ValueError("--sem-loss-weights は --sem-batch > 0 のときだけ使える")
        return None
    if sem is None or not sem_w > 0:
        raise ValueError("--sem-batch > 0 には --sem-cand-weight > 0 と --sem-emb が必要")
    pool = SemPool(tr, sem)
    if pool.n < 2:
        raise ValueError("意味表現の対象文字列が 2 未満")
    if str(args.sem_batch_mix).startswith("items:") and not allow_items:
        raise ValueError("--sem-batch-mix items:B は sem_only 専用")
    sampler = SemSampler(pool, args.sem_batch, args.seed, args.sem_batch_mix, cand_idx=sem.cand_idx)
    targets = PoolTargets(pool, sem, spec, args.sem_emb, seed=args.seed)
    return IndepSem(pool, sampler, targets, spec)
