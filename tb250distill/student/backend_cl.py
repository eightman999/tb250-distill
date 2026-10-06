"""pyopencl + pyclblast（CLBlast SGEMM）backend。backend_np.py と同一の op インターフェース。

- OpenCL 1.1 互換（GT 430 / Fermi）: clEnqueueFillBuffer 等の 1.2 機能・float atomic 拡張は使わない。
  float の atomic add は atomic_cmpxchg ループ（1.1 コア機能）で実装。
- デバイスは名前の部分一致で選択（"GT 430" 等、OpenCL デバイス順は不定のため）。
- 配列は 2D (n,1) の pyopencl.array.Array を基底とし、CLArr が (base, 要素 offset, shape) を持つ。
  pyclblast は offset を要素単位で受けるため、パラメータ・勾配・活性値を 1 本の大きい buffer から切り出して使える。
"""
from __future__ import annotations

import time
import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cla
except Exception as e:  # pragma: no cover
    cl = None
    _IMPORT_ERR = e
try:
    import pyclblast
except Exception as e:  # pragma: no cover
    pyclblast = None
    _CLBLAST_ERR = e


def _prod(shape):
    p = 1
    for s in shape:
        p *= int(s)
    return p


KM = 16  # kd_loss カーネルの最大候補数

KERNEL_SRC = r"""
#define KM 16

inline void atomic_add_f(volatile __global float* addr, float v) {
    union { unsigned int u; float f; } o, n;
    do {
        o.f = *addr;
        n.f = o.f + v;
    } while (atomic_cmpxchg((volatile __global unsigned int*)addr, o.u, n.u) != o.u);
}

inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

__kernel void fill_f(__global float* a, int ao, int n, float v) {
    int i = get_global_id(0);
    if (i < n) a[ao + i] = v;
}
__kernel void fill_i(__global int* a, int ao, int n, int v) {
    int i = get_global_id(0);
    if (i < n) a[ao + i] = v;
}

__kernel void gather_rows(__global float* dst, int dsto, __global const float* tab, int tabo,
                          __global const int* idx, int idxo, int n, int d) {
    int i = get_global_id(0);
    if (i >= n * d) return;
    int r = i / d, c = i - r * d;
    dst[dsto + i] = tab[tabo + idx[idxo + r] * d + c];
}

__kernel void scatter_add_rows(__global float* tab, int tabo, __global const float* src, int srco,
                               __global const int* idx, int idxo, int n, int d) {
    int i = get_global_id(0);
    if (i >= n * d) return;
    float v = src[srco + i];
    if (v == 0.0f) return;
    int r = i / d, c = i - r * d;
    atomic_add_f(&tab[tabo + idx[idxo + r] * d + c], v);
}

__kernel void group_sum(__global float* dst, int dsto, __global const float* src, int srco,
                        int B, int K, int d) {
    int i = get_global_id(0);
    if (i >= B * d) return;
    int b = i / d, c = i - b * d;
    float s = 0.0f;
    for (int k = 0; k < K; k++) s += src[srco + (b * K + k) * d + c];
    dst[dsto + i] = s;
}

__kernel void gru_fwd(__global const float* gi, int gio, __global const float* gh, int gho,
                      __global const float* bi, int bio, __global const float* bh, int bho,
                      __global const float* hp, int hpo, __global float* ho, int hoo,
                      __global const int* lens, int lenso, int t, int B, int H) {
    int i = get_global_id(0);
    if (i >= B * H) return;
    int b = i / H, j = i - b * H;
    int g3 = b * 3 * H + j;
    float r = sigm(gi[gio + g3] + bi[bio + j] + gh[gho + g3] + bh[bho + j]);
    float z = sigm(gi[gio + g3 + H] + bi[bio + H + j] + gh[gho + g3 + H] + bh[bho + H + j]);
    float ghn = gh[gho + g3 + 2 * H] + bh[bho + 2 * H + j];
    float n = tanh(gi[gio + g3 + 2 * H] + bi[bio + 2 * H + j] + r * ghn);
    float h0 = hp[hpo + i];
    float hn = (1.0f - z) * n + z * h0;
    ho[hoo + i] = (t < lens[lenso + b]) ? hn : h0;
}

__kernel void gru_bwd(__global const float* gi, int gio, __global const float* gh, int gho,
                      __global const float* bi, int bio, __global const float* bh, int bho,
                      __global const float* hp, int hpo, __global float* dhc, int dhco,
                      __global const float* dout, int douto, int has_dout,
                      __global float* dgi, int dgio, __global float* dgh, int dgho,
                      __global const int* lens, int lenso, int t, int B, int H) {
    int i = get_global_id(0);
    if (i >= B * H) return;
    int b = i / H, j = i - b * H;
    int g3 = b * 3 * H + j;
    float dh = dhc[dhco + i];
    if (has_dout) dh += dout[douto + i];
    if (t >= lens[lenso + b]) {
        dgi[dgio + g3] = 0.0f; dgi[dgio + g3 + H] = 0.0f; dgi[dgio + g3 + 2 * H] = 0.0f;
        dgh[dgho + g3] = 0.0f; dgh[dgho + g3 + H] = 0.0f; dgh[dgho + g3 + 2 * H] = 0.0f;
        dhc[dhco + i] = dh;
        return;
    }
    float r = sigm(gi[gio + g3] + bi[bio + j] + gh[gho + g3] + bh[bho + j]);
    float z = sigm(gi[gio + g3 + H] + bi[bio + H + j] + gh[gho + g3 + H] + bh[bho + H + j]);
    float ghn = gh[gho + g3 + 2 * H] + bh[bho + 2 * H + j];
    float n = tanh(gi[gio + g3 + 2 * H] + bi[bio + 2 * H + j] + r * ghn);
    float h0 = hp[hpo + i];
    float dn = dh * (1.0f - z);
    float dz = dh * (h0 - n);
    float dnp = dn * (1.0f - n * n);
    float dzp = dz * z * (1.0f - z);
    float drp = dnp * ghn * r * (1.0f - r);
    dgi[dgio + g3] = drp;
    dgi[dgio + g3 + H] = dzp;
    dgi[dgio + g3 + 2 * H] = dnp;
    dgh[dgho + g3] = drp;
    dgh[dgho + g3 + H] = dzp;
    dgh[dgho + g3 + 2 * H] = dnp * r;
    dhc[dhco + i] = dh * z;
}


__kernel void bias_act(__global float* x, int xo, __global const float* bias, int bo,
                       int n, int cols, int act) {
    int i = get_global_id(0);
    if (i >= n) return;
    float v = x[xo + i] + bias[bo + (i % cols)];
    if (act == 1) v = tanh(v);
    x[xo + i] = v;
}

__kernel void tanh_bwd(__global float* dz, int dzo, __global const float* z, int zo, int n) {
    int i = get_global_id(0);
    if (i >= n) return;
    float zz = z[zo + i];
    dz[dzo + i] *= (1.0f - zz * zz);
}

__kernel void kd_loss(__global const float* scores, int so, __global const float* tlog, int to,
                      __global const int* kcnt, int ko, __global const int* gold, int go,
                      __global float* dsc, int dso, __global float* losses, int lo,
                      __global const float* wkd, int wko, __global const float* wce, int wco,
                      int B, int K, float T, float invB) {
    int b = get_global_id(0);
    if (b >= B) return;
    int k = kcnt[ko + b];
    if (k > KM) k = KM;
    int g = gold[go + b];
    float s[KM], t[KM], ps[KM], pt[KM], p1[KM];
    float ms = -1e30f, mt = -1e30f, m1 = -1e30f;
    for (int j = 0; j < k; j++) {
        s[j] = scores[so + b * K + j];
        t[j] = tlog[to + b * K + j];
        ms = fmax(ms, s[j] / T);
        mt = fmax(mt, t[j] / T);
        m1 = fmax(m1, s[j]);
    }
    float es = 0.0f, et = 0.0f, e1 = 0.0f;
    for (int j = 0; j < k; j++) {
        ps[j] = exp(s[j] / T - ms); es += ps[j];
        pt[j] = exp(t[j] / T - mt); et += pt[j];
        p1[j] = exp(s[j] - m1); e1 += p1[j];
    }
    float lses = ms + log(es), lset = mt + log(et), lse1 = m1 + log(e1);
    float kl = 0.0f;
    for (int j = 0; j < k; j++) {
        float lps = s[j] / T - lses;
        float lpt = t[j] / T - lset;
        ps[j] /= es; pt[j] /= et; p1[j] /= e1;
        kl += pt[j] * (lpt - lps);
    }
    float kd = T * T * kl;
    float ce = 0.0f;
    float wk = 1.0f, wc = 0.0f;
    if (g >= 0 && g < k) { ce = -(s[g] - lse1); wk = wkd[wko + b]; wc = wce[wco + b]; }
    losses[lo + b * 3 + 0] = wk * kd + wc * ce;
    losses[lo + b * 3 + 1] = kd;
    losses[lo + b * 3 + 2] = ce;
    for (int j = 0; j < K; j++) {
        float d = 0.0f;
        if (j < k) {
            d = wk * T * (ps[j] - pt[j]);
            if (wc != 0.0f) d += wc * (p1[j] - (j == g ? 1.0f : 0.0f));
        }
        dsc[dso + b * K + j] = d * invB;
    }
}

// Candidate Semantic Distillation: 行ごとの cosine loss と dL/dz（backend_np.cos_loss と同一式）。w==0 の行は 0。
__kernel void cos_loss(__global const float* z, int zo, __global const float* t, int to,
                       __global const float* w, int wo, __global float* dz, int dzo,
                       __global float* lrow, int lo, int N, int d) {
    int i = get_global_id(0);
    if (i >= N) return;
    float ww = w[wo + i];
    if (ww == 0.0f) {
        for (int c = 0; c < d; c++) dz[dzo + i * d + c] = 0.0f;
        lrow[lo + i] = 0.0f;
        return;
    }
    float a = 0.0f, zz = 0.0f, tt = 0.0f;
    for (int c = 0; c < d; c++) {
        float x = z[zo + i * d + c], y = t[to + i * d + c];
        a += x * y; zz += x * x; tt += y * y;
    }
    float nz = sqrt(zz + 1e-12f), nt = sqrt(tt + 1e-12f);
    float cs = a / (nz * nt);
    lrow[lo + i] = 1.0f - cs;
    float inz = 1.0f / nz, int_ = 1.0f / nt;
    for (int c = 0; c < d; c++) {
        float x = z[zo + i * d + c], y = t[to + i * d + c];
        dz[dzo + i * d + c] = -ww * (y * int_ - cs * x * inz) * inz;
    }
}

// 言語モデル用: 行ごとに in-place で logits -> dlogits=(softmax-onehot)*invN、losses[row] = -log p[tgt]。
// 1 work-group(256) = 1 行。backend_np.softmax_ce と同一式。
__kernel void softmax_ce(__global float* x, int xo, __global const int* tgt, int to,
                         __global float* losses, int lo, int N, int V, float invN) {
    __local float sh[256];
    __local float shx;
    int row = get_group_id(0);
    int lid = get_local_id(0);
    if (row >= N) return;
    __global float* r = x + xo + row * V;
    int t = tgt[to + row];
    float m = -1e30f;
    for (int j = lid; j < V; j += 256) m = fmax(m, r[j]);
    sh[lid] = m;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = 128; st > 0; st >>= 1) {
        if (lid < st) sh[lid] = fmax(sh[lid], sh[lid + st]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    m = sh[0];
    if (lid == 0) shx = r[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.0f;
    for (int j = lid; j < V; j += 256) s += exp(r[j] - m);
    sh[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = 128; st > 0; st >>= 1) {
        if (lid < st) sh[lid] += sh[lid + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    s = sh[0];
    float inv_s = 1.0f / s;
    if (lid == 0) losses[lo + row] = m + log(s) - shx;
    barrier(CLK_GLOBAL_MEM_FENCE);
    for (int j = lid; j < V; j += 256) {
        float p = exp(r[j] - m) * inv_s;
        if (j == t) p -= 1.0f;
        r[j] = p * invN;
    }
}

__kernel void sumsq_partial(__global const float* g, int go, int n, __global float* part, int po) {
    __local float sh[256];
    int lid = get_local_id(0), grp = get_group_id(0), ng = get_num_groups(0);
    float s = 0.0f;
    for (int i = grp * 256 + lid; i < n; i += ng * 256) {
        float v = g[go + i];
        s += v * v;
    }
    sh[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = 128; st > 0; st >>= 1) {
        if (lid < st) sh[lid] += sh[lid + st];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) part[po + grp] = sh[0];
}

__kernel void sumsq_final(__global const float* part, int po, int G, __global float* out, int oo) {
    if (get_global_id(0) != 0) return;
    float s = 0.0f;
    for (int i = 0; i < G; i++) s += part[po + i];
    out[oo] = s;
}

__kernel void adamw(__global float* p, int po, __global const float* g, int go,
                    __global float* m, int mo, __global float* v, int vo,
                    __global const float* ss, int sso, int n,
                    float lr, float b1, float b2, float eps, float wd,
                    float bc1, float bc2s, float clip) {
    int i = get_global_id(0);
    if (i >= n) return;
    float gn = sqrt(ss[sso]);
    float coef = (clip > 0.0f) ? fmin(1.0f, clip / (gn + 1e-6f)) : 1.0f;
    float gg = g[go + i] * coef;
    float mm = b1 * m[mo + i] + (1.0f - b1) * gg;
    float vv = b2 * v[vo + i] + (1.0f - b2) * gg * gg;
    m[mo + i] = mm;
    v[vo + i] = vv;
    float pp = p[po + i] * (1.0f - lr * wd);
    float denom = sqrt(vv) / bc2s + eps;
    p[po + i] = pp - (lr / bc1) * mm / denom;
}
"""

# カーネルごとの引数仕様: a=配列(buffer+offset の 2 引数に展開), i=int32, f=float32
KERNEL_SPECS = {
    "fill_f": "aif",
    "fill_i": "aii",
    "gather_rows": "aaaii",          # dst tab idx n d
    "scatter_add_rows": "aaaii",     # tab src idx n d
    "group_sum": "aaiii",            # dst src B K d
    "gru_fwd": "aaaaaaaiii",         # gi gh bi bh hp ho lens t B H
    "gru_bwd": "aaaaaaaiaaaiii",     # gi gh bi bh hp dhc dout has_dout dgi dgh lens t B H
    "bias_act": "aaiii",             # x bias n cols act
    "tanh_bwd": "aai",               # dz z n
    "kd_loss": "aaaaaaaaiiff",       # scores tlog kcnt gold dsc losses wkd wce(sample ごとの重み配列) B K T invB
    "cos_loss": "aaaaaii",           # z t w dz lrow N d
    "softmax_ce": "aaaiif",          # x tgt losses N V invN（言語モデル用）
    "sumsq_partial": "aia",          # g n part
    "sumsq_final": "aia",            # part G out
    "adamw": "aaaaaiffffffff",       # p g m v ss n lr b1 b2 eps wd bc1 bc2s clip
}


class CLArr:
    __slots__ = ("base", "off", "shape", "kind")

    def __init__(self, base, off, shape, kind):
        self.base = base
        self.off = off
        self.shape = shape
        self.kind = kind

    @property
    def size(self):
        return _prod(self.shape)


def list_devices():
    out = []
    if cl is None:
        return out
    for p in cl.get_platforms():
        for d in p.get_devices():
            out.append((p, d))
    return out


def select_device(substr):
    """デバイス名の部分一致（大文字小文字無視）。'GT 430#1' で同名複数の 2 番目。"""
    idx = 0
    if "#" in substr:
        substr, _, s = substr.partition("#")
        idx = int(s)
    cands = [d for _, d in list_devices() if substr.lower() in d.name.lower()]
    if len(cands) <= idx:
        names = [d.name for _, d in list_devices()]
        raise RuntimeError(f"OpenCL device matching {substr!r} not found; available: {names}")
    return cands[idx]


class CLBackend:
    name = "cl"
    dtype = np.dtype(np.float32)

    def __init__(self, device="GT 430"):
        if cl is None:
            raise RuntimeError(f"pyopencl unavailable: {_IMPORT_ERR}")
        if pyclblast is None:
            raise RuntimeError(f"pyclblast unavailable: {_CLBLAST_ERR}")
        self.dev = select_device(device)
        self.device_name = self.dev.name
        self.ctx = cl.Context([self.dev])
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, KERNEL_SRC).build()
        self.k = {}
        for name, spec in KERNEL_SPECS.items():
            kern = getattr(self.prg, name)
            dts = []
            for c in spec:
                if c == "a":
                    dts += [None, np.int32]
                elif c == "i":
                    dts.append(np.int32)
                else:
                    dts.append(np.float32)
            kern.set_scalar_arg_dtypes(dts)
            self.k[name] = (kern, spec)
        self._ones = None
        self._part = None
        self.pci_bus_id = self._pci_id()
        self.version = self.dev.version

    def _pci_id(self):
        try:
            bus = self.dev.get_info(cl.device_info.PCI_BUS_ID_NV)
            slot = self.dev.get_info(cl.device_info.PCI_SLOT_ID_NV)
            return "%02x:%02x.%d" % (bus, slot >> 3, slot & 7)
        except Exception:
            return None

    # ---- 起動 ----
    def _call(self, name, gsize, lsize, *args):
        kern, spec = self.k[name]
        flat = []
        ai = 0
        for c in spec:
            a = args[ai]
            ai += 1
            if c == "a":
                flat.append(a.base.data)
                flat.append(a.off)
            else:
                flat.append(a)
        if isinstance(gsize, tuple):
            gs, ls = tuple(int(x) for x in gsize), tuple(int(x) for x in lsize)
        elif lsize is None:
            # local size を未指定にすると Nvidia(390) は大きい global で極端に遅い（adamw 1.5M 要素が 60ms）ので
            # 明示的に LS 固定 + global を切り上げる（全カーネルが範囲チェックを持つ）
            ls = (self.LS,)
            gs = (((int(gsize) + self.LS - 1) // self.LS) * self.LS,)
        else:
            ls, gs = (int(lsize),), (int(gsize),)
        ev = kern(self.queue, gs, ls, *flat)
        if self.events is not None:
            self.events.append((name, ev))

    LS = 128
    events = None  # プロファイル用: list を入れると (kernel名, event) を記録（queue は PROFILING_ENABLE が必要）

    @staticmethod
    def _gs(n):
        return max(1, int(n))

    # ---- メモリ ----
    def alloc(self, shape, kind="f"):
        if isinstance(shape, int):
            shape = (shape,)
        n = _prod(shape)
        base = cla.zeros(self.queue, (max(n, 1), 1), np.float32 if kind == "f" else np.int32)
        return CLArr(base, 0, tuple(shape), kind)

    def view(self, arr, off, shape):
        if isinstance(shape, int):
            shape = (shape,)
        return CLArr(arr.base, arr.off + off, tuple(shape), arr.kind)

    def zero(self, a):
        if a.kind == "f":
            self._call("fill_f", self._gs(a.size), None, a, a.size, 0.0)
        else:
            self._call("fill_i", self._gs(a.size), None, a, a.size, 0)

    def upload(self, a, host):
        h = np.ascontiguousarray(host, dtype=np.float32 if a.kind == "f" else np.int32).reshape(-1)
        itemsize = 4
        cl.enqueue_copy(self.queue, a.base.data, h, dst_offset=a.off * itemsize, is_blocking=True)

    def _wait(self, ev):
        """event をポーリングして待つ（clFinish / blocking read は CPU を 1 コア占有する。Celeron 2 コアでは致命的）。
        最初の 0.3ms だけ spin し、以降は 0.2ms sleep で待つ。"""
        self.queue.flush()
        done = cl.command_execution_status.COMPLETE
        get = cl.event_info.COMMAND_EXECUTION_STATUS
        t0 = time.perf_counter()
        while True:
            st = ev.get_info(get)
            if st == done:
                return
            if st < 0:
                raise RuntimeError(f"OpenCL command failed with status {st}")
            if time.perf_counter() - t0 > 3e-4:
                time.sleep(2e-4)

    def download(self, a):
        out = np.empty(a.shape, dtype=np.float32 if a.kind == "f" else np.int32)
        ev = cl.enqueue_copy(self.queue, out, a.base.data, src_offset=a.off * 4, is_blocking=False)
        self._wait(ev)
        return out

    def copy(self, dst, src):
        cl.enqueue_copy(self.queue, dst.base.data, src.base.data, byte_count=src.size * 4,
                        src_offset=src.off * 4, dst_offset=dst.off * 4)

    def sync(self):
        self._wait(cl.enqueue_marker(self.queue))

    # ---- BLAS ----
    def gemm(self, A, B, C, transA=False, transB=False, alpha=1.0, beta=0.0):
        ar, ac = A.shape
        br, bc = B.shape
        m, k = (ac, ar) if transA else (ar, ac)
        k2, n = (bc, br) if transB else (br, bc)
        assert k == k2, (A.shape, B.shape, transA, transB)
        assert C.shape == (m, n), (C.shape, m, n)
        pyclblast.gemm(self.queue, m, n, k, A.base, B.base, C.base, ac, bc, n,
                       alpha=alpha, beta=beta, a_transp=transA, b_transp=transB,
                       a_offset=A.off, b_offset=B.off, c_offset=C.off)

    def _get_ones(self, n):
        if self._ones is None or self._ones.size < n:
            self._ones = self.alloc((max(n, 4096),))
            self._call("fill_f", self._gs(self._ones.size), None, self._ones, self._ones.size, 1.0)
        return self._ones

    def colsum(self, dst, src):
        rows, cols = src.shape
        ones = self.view(self._get_ones(rows), 0, (1, rows))
        self.gemm(ones, src, self.view(dst, 0, (1, cols)), beta=1.0)

    # ---- embedding ----
    def gather_rows(self, dst, table, idx):
        n, d = dst.shape
        self._call("gather_rows", self._gs(n * d), None, dst, table, idx, n, d)

    def scatter_add_rows(self, tab_grad, src, idx):
        n, d = src.shape
        self._call("scatter_add_rows", self._gs(n * d), None, tab_grad, src, idx, n, d)

    def group_sum(self, dst, src, K):
        B, d = dst.shape
        self._call("group_sum", self._gs(B * d), None, dst, src, B, K, d)

    # ---- GRU ----
    def gru_fwd(self, gi, gh, bi, bh, hprev, hout, lens, t):
        B, H = hprev.shape
        self._call("gru_fwd", self._gs(B * H), None, gi, gh, bi, bh, hprev, hout, lens, t, B, H)

    def gru_bwd(self, gi, gh, bi, bh, hprev, dhc, dout, dgi, dgh, lens, t):
        B, H = hprev.shape
        has = 1 if dout is not None else 0
        self._call("gru_bwd", self._gs(B * H), None, gi, gh, bi, bh, hprev, dhc,
                   dout if dout is not None else dhc, has, dgi, dgh, lens, t, B, H)

    # ---- head ----
    def bias_act(self, x, bias, act):
        n = x.size
        cols = x.shape[1]
        self._call("bias_act", self._gs(n), None, x, bias, n, cols, act)

    def tanh_bwd(self, dz, z):
        self._call("tanh_bwd", self._gs(dz.size), None, dz, z, dz.size)

    # ---- loss ----
    def kd_loss(self, scores, tlog, kcnt, gold, dscores, losses, B, K, T, w_kd_g, w_ce_g, inv_b):
        if K > KM:
            raise ValueError(f"K={K} exceeds kd_loss kernel limit {KM}")
        wkd, wce = self._loss_weights(w_kd_g, w_ce_g, B)
        self._call("kd_loss", self._gs(B), None, scores, tlog, kcnt, gold, dscores, losses,
                   wkd, wce, B, K, T, inv_b)

    def _loss_weights(self, w_kd_g, w_ce_g, B):
        """w_kd_g / w_ce_g がスカラーなら B 要素の定数 buffer（キャッシュ、値/容量が変わった時だけ再 upload）、
        CLArr（sample ごとの重み。呼び出し側が device へ upload 済み）ならそのまま使う。"""
        if isinstance(w_kd_g, CLArr) and isinstance(w_ce_g, CLArr):
            return w_kd_g, w_ce_g
        if isinstance(w_kd_g, CLArr) or isinstance(w_ce_g, CLArr):
            raise TypeError("w_kd_g と w_ce_g は両方スカラーか両方 CLArr にする")
        key = (float(w_kd_g), float(w_ce_g))
        c = getattr(self, "_wconst", None)
        if c is None or c["B"] < B:
            cap = max(B, 1)
            c = {"B": cap, "key": None, "kd": self.alloc((cap,)), "ce": self.alloc((cap,))}
            self._wconst = c
        if c["key"] != key:
            self.upload(c["kd"], np.full(c["B"], key[0], np.float32))
            self.upload(c["ce"], np.full(c["B"], key[1], np.float32))
            c["key"] = key
        return c["kd"], c["ce"]

    def cos_loss(self, z, t, w, dz, lrow, N, d):
        self._call("cos_loss", self._gs(N), None, z, t, w, dz, lrow, N, d)

    def softmax_ce(self, x, tgt, losses, N, V, inv_n):
        self._call("softmax_ce", (N * 256,), (256,), x, tgt, losses, N, V, inv_n)

    # ---- optimizer ----
    NG = 64

    def sumsq(self, g, out):
        if self._part is None:
            self._part = self.alloc((self.NG,))
        self._call("sumsq_partial", self.NG * 256, 256, g, g.size, self._part)
        self._call("sumsq_final", 1, None, self._part, self.NG, out)

    def adamw(self, p, g, m, v, ss, lr, beta1, beta2, eps, wd, step, clip):
        bc1 = 1.0 - beta1 ** step
        bc2s = (1.0 - beta2 ** step) ** 0.5
        self._call("adamw", self._gs(p.size), None, p, g, m, v, ss, p.size,
                   lr, beta1, beta2, eps, wd, bc1, bc2s, float(clip or 0.0))
