"""numpy backend（CPU）。backend_cl.py と同一の op インターフェース。

インターフェース規約（両 backend 共通、model.py はこの op だけを使う）:

- 配列は不透明ハンドル。`alloc` で確保し、`view(arr, off, shape)` で連続領域の部分ビューを作る
  （off は要素単位）。np backend ではハンドル = ndarray、cl backend では CLArr。
- 全ての行列は row-major・連続。`gemm` は shape から m,n,k を推定する。
- 勾配系 op（gemm beta=1, colsum, scatter_add_rows）は累積（+=）。
- dtype は 'f'（float: np backend は構築時に float32/float64 を選べる、cl は float32）か 'i'（int32）。
"""
from __future__ import annotations

import numpy as np


def _prod(shape):
    p = 1
    for s in shape:
        p *= int(s)
    return p


class NPBackend:
    name = "np"

    def __init__(self, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        self.pci_bus_id = None
        self.device_name = "numpy-cpu"

    # ---- メモリ ----
    def alloc(self, shape, kind="f"):
        if isinstance(shape, int):
            shape = (shape,)
        return np.zeros(shape, dtype=self.dtype if kind == "f" else np.int32)

    def view(self, arr, off, shape):
        if isinstance(shape, int):
            shape = (shape,)
        n = _prod(shape)
        return arr.reshape(-1)[off:off + n].reshape(shape)

    def zero(self, a):
        a[...] = 0

    def upload(self, a, host):
        h = np.ascontiguousarray(host).reshape(-1)
        a.reshape(-1)[:h.size] = h

    def download(self, a):
        return np.array(a, copy=True)

    def copy(self, dst, src):
        dst[...] = src

    def sync(self):
        pass

    # ---- BLAS ----
    def gemm(self, A, B, C, transA=False, transB=False, alpha=1.0, beta=0.0):
        a = A.T if transA else A
        b = B.T if transB else B
        if beta == 0.0:
            if alpha == 1.0:
                np.matmul(a, b, out=C)
            else:
                np.matmul(a, b, out=C)
                C *= alpha
        else:
            tmp = a @ b
            if alpha != 1.0:
                tmp *= alpha
            if beta == 1.0:
                C += tmp
            else:
                C *= beta
                C += tmp

    def colsum(self, dst, src):
        """dst(cols) += src(rows, cols).sum(0)"""
        dst.reshape(-1)[:] += src.sum(axis=0, dtype=self.dtype).reshape(-1)

    # ---- embedding ----
    def gather_rows(self, dst, table, idx):
        np.take(table, idx, axis=0, out=dst)

    def scatter_add_rows(self, tab_grad, src, idx):
        """tab_grad[idx[i]] += src[i]（重複 index は合算）"""
        n = idx.shape[0]
        if n == 0:
            return
        order = np.argsort(idx, kind="stable")
        sidx = idx[order]
        starts = np.flatnonzero(np.r_[True, sidx[1:] != sidx[:-1]])
        sums = np.add.reduceat(src[order], starts, axis=0)
        tab_grad[sidx[starts]] += sums

    def group_sum(self, dst, src, K):
        """dst(B,d) = src(B*K,d) を K 行ずつ合計"""
        B, d = dst.shape
        dst[...] = src.reshape(B, K, d).sum(axis=1)

    # ---- GRU 融合 op ----
    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _gates(self, gi, gh, bi, bh, H):
        a = gi + bi
        g = gh + bh
        r = self._sig(a[:, :H] + g[:, :H])
        z = self._sig(a[:, H:2 * H] + g[:, H:2 * H])
        ghn = g[:, 2 * H:]
        n = np.tanh(a[:, 2 * H:] + r * ghn)
        return r, z, n, ghn

    def gru_fwd(self, gi, gh, bi, bh, hprev, hout, lens, t):
        """h_t = m ? (1-z)*n + z*h_prev : h_prev。gi/gh は (B,3H)、bi/bh は (3H)。"""
        H = hprev.shape[1]
        r, z, n, _ = self._gates(gi, gh, bi, bh, H)
        hn = (1.0 - z) * n + z * hprev
        m = (t < lens)[:, None]
        hout[...] = np.where(m, hn, hprev)

    def gru_bwd(self, gi, gh, bi, bh, hprev, dhc, dout, dgi, dgh, lens, t):
        """1 ステップの backward。dhc(B,H) は入力=後続ステップからの carry、出力=h_prev への勾配
        （直接項+carry 項。dgh@Wh^T は呼び出し側が gemm(beta=1) で加える）。
        dout は上位層からの勾配 (B,H) または None。dgi/dgh に (B,3H) を書く。"""
        H = hprev.shape[1]
        r, z, n, ghn = self._gates(gi, gh, bi, bh, H)
        dh = dhc if dout is None else dhc + dout
        m = (t < lens)[:, None]
        dhm = np.where(m, dh, 0.0)
        dn = dhm * (1.0 - z)
        dz = dhm * (hprev - n)
        dnp = dn * (1.0 - n * n)
        dzp = dz * z * (1.0 - z)
        drp = dnp * ghn * r * (1.0 - r)
        dgi[:, :H] = drp
        dgi[:, H:2 * H] = dzp
        dgi[:, 2 * H:] = dnp
        dgh[:, :H] = drp
        dgh[:, H:2 * H] = dzp
        dgh[:, 2 * H:] = dnp * r
        dhc[...] = dhm * z + np.where(m, 0.0, dh)

    # ---- head ----
    def bias_act(self, x, bias, act):
        """x(rows,cols) += bias(cols); act=1 なら tanh。in-place"""
        x += bias.reshape(1, -1)
        if act == 1:
            np.tanh(x, out=x)

    def tanh_bwd(self, dz, z):
        dz *= (1.0 - z * z)

    # ---- loss ----
    def kd_loss(self, scores, tlog, kcnt, gold, dscores, losses, B, K, T, w_kd_g, w_ce_g, inv_b):
        """scores/tlog/dscores は (B*K) 平坦、kcnt/gold は (B)、losses は (B*3)=[total,kd,ce]。
        gold>=0: w_kd_g*KD + w_ce_g*CE(T=1)、gold<0: KD のみ。dscores は inv_b 倍済み。
        w_kd_g / w_ce_g はスカラー、または sample ごとの重み配列（B 要素。gold 有りの sample にだけ効く）。"""
        dt = self.dtype
        s = scores.reshape(B, K).astype(np.float64)
        t = tlog.reshape(B, K).astype(np.float64)
        k = kcnt[:B]
        g = gold[:B]
        mask = np.arange(K)[None, :] < k[:, None]
        neg = -1e30

        def logsm(x, tmp):
            x = np.where(mask, x / tmp, neg)
            mx = x.max(axis=1, keepdims=True)
            e = np.where(mask, np.exp(x - mx), 0.0)
            lse = mx + np.log(e.sum(axis=1, keepdims=True))
            return np.where(mask, x - lse, neg), e / e.sum(axis=1, keepdims=True)

        lps, ps = logsm(s, T)
        lpt, pt = logsm(t, T)
        kl = np.where(mask, pt * (lpt - lps), 0.0).sum(axis=1)
        kd = T * T * kl
        ls1, p1 = logsm(s, 1.0)
        has = g >= 0
        gi = np.where(has, g, 0)
        ce = np.where(has, -ls1[np.arange(B), gi], 0.0)
        wk_g = w_kd_g if np.ndim(w_kd_g) == 0 else np.asarray(w_kd_g).reshape(-1)[:B].astype(np.float64)
        wc_g = w_ce_g if np.ndim(w_ce_g) == 0 else np.asarray(w_ce_g).reshape(-1)[:B].astype(np.float64)
        wk = np.where(has, wk_g, 1.0)
        wc = np.where(has, wc_g, 0.0)
        total = wk * kd + wc * ce
        onehot = (np.arange(K)[None, :] == gi[:, None]) & has[:, None]
        ds = wk[:, None] * T * (ps - pt) + wc[:, None] * (p1 - onehot)
        ds = np.where(mask, ds, 0.0) * inv_b
        dscores.reshape(-1)[:B * K] = ds.reshape(-1).astype(dt)
        L = np.stack([total, kd, ce], axis=1)
        losses.reshape(-1)[:B * 3] = L.reshape(-1).astype(dt)

    def softmax_ce(self, x, tgt, losses, N, V, inv_n):
        """言語モデル（Wikipedia 事前学習）用。x(N,V) のロジットを in-place で dlogits=(softmax-onehot)*inv_n に置き換え、
        losses[i] = -log softmax(x_i)[tgt_i]（行ごと）を書く。tgt は int32 (N)。"""
        xx = x.reshape(N, V).astype(np.float64)
        m = xx.max(axis=1, keepdims=True)
        e = np.exp(xx - m)
        s = e.sum(axis=1, keepdims=True)
        t = tgt.reshape(-1)[:N]
        ar = np.arange(N)
        losses.reshape(-1)[:N] = ((m + np.log(s))[:, 0] - xx[ar, t]).astype(self.dtype)
        p = e / s
        p[ar, t] -= 1.0
        x.reshape(N, V)[...] = (p * inv_n).astype(self.dtype)

    COS_EPS = 1e-12

    def cos_loss(self, z, t, w, dz, lrow, N, d):
        """Candidate Semantic Distillation の cosine loss（行ごと）。z/t/dz は (N,d)、w/lrow は (N)。
        cos = z·t / (sqrt(|z|^2+eps) sqrt(|t|^2+eps))。w[i]!=0 の行だけ: lrow[i] = 1 - cos、dz[i] = w[i] * d(1-cos)/dz（厳密な導関数）。
        w[i]==0 の行は lrow=0・dz=0（無効候補）。w は呼び出し側が λ/N_valid（有効行）にして渡す。"""
        zz = z.reshape(N, d).astype(np.float64)
        tt = t.reshape(N, d).astype(np.float64)
        ww = w.reshape(-1)[:N].astype(np.float64)
        a = (zz * tt).sum(axis=1)
        nz = np.sqrt((zz * zz).sum(axis=1) + self.COS_EPS)
        nt = np.sqrt((tt * tt).sum(axis=1) + self.COS_EPS)
        cos = a / (nz * nt)
        on = ww != 0
        l = np.where(on, 1.0 - cos, 0.0)
        g = (tt / nt[:, None] - cos[:, None] * zz / nz[:, None]) / nz[:, None]   # d cos / d z
        dzz = np.where(on[:, None], -ww[:, None] * g, 0.0)
        dz.reshape(-1)[:N * d] = dzz.reshape(-1).astype(self.dtype)
        lrow.reshape(-1)[:N] = l.astype(self.dtype)

    # ---- optimizer ----
    def sumsq(self, g, out):
        gg = g.reshape(-1).astype(np.float64)
        out.reshape(-1)[0] = float(np.dot(gg, gg))

    def adamw(self, p, g, m, v, ss, lr, beta1, beta2, eps, wd, step, clip):
        """PyTorch AdamW と同じ更新式 + grad clip（ss は sumsq の結果、スカラー）"""
        gn = float(np.sqrt(ss.reshape(-1)[0]))
        coef = min(1.0, clip / (gn + 1e-6)) if clip and clip > 0 else 1.0
        gc = g * self.dtype.type(coef)
        m *= beta1
        m += (1.0 - beta1) * gc
        v *= beta2
        v += (1.0 - beta2) * gc * gc
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        p *= (1.0 - lr * wd)
        denom = np.sqrt(v) / np.sqrt(bc2) + eps
        p -= (lr / bc1) * m / denom
