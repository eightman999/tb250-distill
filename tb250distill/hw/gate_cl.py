"""OpenCL Gate test（GT730 / GT710 / GT430 想定）。

    python -m tb250distill.hw.gate_cl --device "GT 430" --out runs/hw [--minutes 10]

1. SGEMM 正当性: pyclblast と自前 tiled kernel の両方を numpy と比較。どちらが使えたかを記録。
2. backward に必要な基本演算カーネル（sigmoid/tanh と微分、mul/add、行 softmax(+backward)、
   reduction sum / col sum、atomic_cmpxchg による float scatter-add、AdamW、gather）を numpy と照合。
3. --minutes N: SGEMM 連続負荷。1 分ごとに GFLOPS・結果チェックサム・nvidia-smi 温度/クロック/VRAM を JSONL 記録。
   結果不一致やエラーで FAIL。

OpenCL 1.1（GT430）で動くよう 1.2 専用機能は使わない（clEnqueueFillBuffer 等を使う pyopencl の
Array.fill/zeros は使用しない）。FP32 のみ。1 プロセス 1 GPU（同一プロセスに Fermi/Kepler の
コンテキストを併存させると GT430 が INVALID_DEVICE になる事象を実機で確認済み）。
結果: <out>/gate_<dev>.json, <out>/gate_<dev>_load.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cla
except Exception as _e:  # noqa: BLE001
    cl = None
    cla = None
    _IMPORT_ERR = repr(_e)

try:
    import pyclblast
except Exception as _e:  # noqa: BLE001
    pyclblast = None
    _CLBLAST_IMPORT_ERR = repr(_e)

GEMM_TOL = 2e-5   # 最大絶対誤差 / max|ref|（FP32 蓄積誤差は ~1e-6 程度）
OP_TOL = 2e-5

# 以降の kernel は OpenCL C 1.0/1.1 で通る記述に限る。
KERNEL_SRC = r"""
#define TS 16

// C[m,n] = alpha * sum_k A[m,k]*B[k,n] + beta*C.  A[m,k]=A[m*a_rs+k*a_cs], B[k,n]=B[k*b_rs+n*b_cs]
__kernel void sgemm_tiled(const int M, const int N, const int K,
    __global const float* A, const int a_rs, const int a_cs,
    __global const float* B, const int b_rs, const int b_cs,
    __global float* C, const int ldc, const float alpha, const float beta)
{
    __local float As[TS][TS];
    __local float Bs[TS][TS];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int col = get_group_id(0) * TS + tx;
    const int row = get_group_id(1) * TS + ty;
    float acc = 0.0f;
    for (int t = 0; t < K; t += TS) {
        const int ak = t + tx;
        const int bk = t + ty;
        As[ty][tx] = (row < M && ak < K) ? A[row * a_rs + ak * a_cs] : 0.0f;
        Bs[ty][tx] = (bk < K && col < N) ? B[bk * b_rs + col * b_cs] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int kk = 0; kk < TS; kk++) acc += As[ty][kk] * Bs[kk][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) {
        float r = alpha * acc;
        if (beta != 0.0f) r += beta * C[row * ldc + col];
        C[row * ldc + col] = r;
    }
}

__kernel void k_sigmoid(__global const float* x, __global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = 1.0f / (1.0f + exp(-x[i]));
}
__kernel void k_tanh(__global const float* x, __global float* y, const int n) {
    int i = get_global_id(0); if (i < n) y[i] = tanh(x[i]);
}
// y = 出力値。dx = dy * y * (1-y)
__kernel void k_dsigmoid(__global const float* y, __global const float* dy, __global float* dx, const int n) {
    int i = get_global_id(0); if (i < n) dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}
// y = 出力値。dx = dy * (1-y^2)
__kernel void k_dtanh(__global const float* y, __global const float* dy, __global float* dx, const int n) {
    int i = get_global_id(0); if (i < n) dx[i] = dy[i] * (1.0f - y[i] * y[i]);
}
__kernel void k_mul(__global const float* a, __global const float* b, __global float* c, const int n) {
    int i = get_global_id(0); if (i < n) c[i] = a[i] * b[i];
}
__kernel void k_add(__global const float* a, __global const float* b, __global float* c, const int n) {
    int i = get_global_id(0); if (i < n) c[i] = a[i] + b[i];
}

// 1 work-group = 1 行。local size は 2 の冪。
__kernel void softmax_rows(__global const float* x, __global float* y, const int cols, __local float* sm) {
    const int row = get_group_id(0), lid = get_local_id(0), ls = get_local_size(0);
    __global const float* xr = x + row * cols;
    __global float* yr = y + row * cols;
    float m = -3.0e38f;
    for (int j = lid; j < cols; j += ls) m = fmax(m, xr[j]);
    sm[lid] = m; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls / 2; s > 0; s >>= 1) { if (lid < s) sm[lid] = fmax(sm[lid], sm[lid + s]); barrier(CLK_LOCAL_MEM_FENCE); }
    m = sm[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float acc = 0.0f;
    for (int j = lid; j < cols; j += ls) acc += exp(xr[j] - m);
    sm[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls / 2; s > 0; s >>= 1) { if (lid < s) sm[lid] += sm[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float tot = sm[0];
    for (int j = lid; j < cols; j += ls) yr[j] = exp(xr[j] - m) / tot;
}
// dx = y * (dy - sum(dy*y))
__kernel void softmax_rows_bwd(__global const float* y, __global const float* dy, __global float* dx, const int cols, __local float* sm) {
    const int row = get_group_id(0), lid = get_local_id(0), ls = get_local_size(0);
    __global const float* yr = y + row * cols;
    __global const float* dyr = dy + row * cols;
    __global float* dxr = dx + row * cols;
    float acc = 0.0f;
    for (int j = lid; j < cols; j += ls) acc += yr[j] * dyr[j];
    sm[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ls / 2; s > 0; s >>= 1) { if (lid < s) sm[lid] += sm[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    const float dot = sm[0];
    for (int j = lid; j < cols; j += ls) dxr[j] = yr[j] * (dyr[j] - dot);
}

// 全要素 sum（2 段: partial -> partial を 1 group で再度）。local size は 2 の冪。
__kernel void reduce_sum_partial(__global const float* x, __global float* partial, const int n, __local float* sm) {
    const int lid = get_local_id(0), gid = get_global_id(0), gs = get_global_size(0);
    float acc = 0.0f;
    for (int i = gid; i < n; i += gs) acc += x[i];
    sm[lid] = acc; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) { if (lid < s) sm[lid] += sm[lid + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (lid == 0) partial[get_group_id(0)] = sm[0];
}
// 行方向 sum（bias 勾配）: out[c] = sum_r x[r,c]
__kernel void col_sum(__global const float* x, __global float* out, const int rows, const int cols) {
    int c = get_global_id(0);
    if (c < cols) { float acc = 0.0f; for (int r = 0; r < rows; r++) acc += x[r * cols + c]; out[c] = acc; }
}

inline void atomic_add_f(volatile __global float* p, const float v) {
    volatile __global unsigned int* q = (volatile __global unsigned int*)p;
    unsigned int prev, old = *q;
    do { prev = old; old = atomic_cmpxchg(q, prev, as_uint(as_float(prev) + v)); } while (old != prev);
}
// embedding 勾配: out[idx[r], :] += g[r, :]
__kernel void scatter_add_rows(__global const int* idx, __global const float* g, __global float* out, const int n, const int dim) {
    int i = get_global_id(0);
    if (i < n * dim) { int r = i / dim, c = i - r * dim; atomic_add_f(out + idx[r] * dim + c, g[i]); }
}
// embedding forward: out[r,:] = emb[idx[r],:]
__kernel void gather_rows(__global const float* emb, __global const int* idx, __global float* out, const int n, const int dim) {
    int i = get_global_id(0);
    if (i < n * dim) { int r = i / dim, c = i - r * dim; out[i] = emb[idx[r] * dim + c]; }
}

// AdamW（bc1=1-b1^t, bc2=1-b2^t はホストから渡す）。decoupled weight decay。
__kernel void adamw(__global float* p, __global const float* g, __global float* m, __global float* v, const int n,
                    const float lr, const float b1, const float b2, const float eps, const float wd,
                    const float bc1, const float bc2) {
    int i = get_global_id(0);
    if (i < n) {
        float gi = g[i];
        float mi = b1 * m[i] + (1.0f - b1) * gi;
        float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i] = mi; v[i] = vi;
        float mh = mi / bc1, vh = vi / bc2;
        p[i] = p[i] * (1.0f - lr * wd) - lr * mh / (sqrt(vh) + eps);
    }
}
"""


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


# ------------------------------------------------------------ device 選択
def find_device(name_part: str):
    if cl is None:
        raise RuntimeError("pyopencl import failed: " + _IMPORT_ERR)
    want = name_part.lower()
    found = []
    for pi, p in enumerate(cl.get_platforms()):
        for d in p.get_devices():
            if want in d.name.lower():
                found.append((p, d))
    if not found:
        allnames = [d.name for p in cl.get_platforms() for d in p.get_devices()]
        raise RuntimeError("device matching %r not found; available: %s" % (name_part, allnames))
    return found[0]


def device_info(p, d) -> dict:
    info = {"platform": p.name, "platform_version": p.version, "name": d.name,
            "version": d.version, "opencl_c_version": d.opencl_c_version, "driver_version": d.driver_version}
    for key, attr in (("global_mem_bytes", "global_mem_size"), ("max_alloc_bytes", "max_mem_alloc_size"),
                      ("local_mem_bytes", "local_mem_size"), ("compute_units", "max_compute_units"),
                      ("clock_mhz", "max_clock_frequency"), ("max_work_group_size", "max_work_group_size")):
        try:
            info[key] = getattr(d, attr)
        except Exception as ex:  # noqa: BLE001
            info[key] = "error: %s" % ex
    try:
        info["pci_bus_id_nv"] = d.get_info(cl.device_info.PCI_BUS_ID_NV)
    except Exception:  # noqa: BLE001
        info["pci_bus_id_nv"] = None
    m = re.search(r"OpenCL (\d+)\.(\d+)", d.version)
    info["cl_version_tuple"] = [int(m.group(1)), int(m.group(2))] if m else None
    return info


# ------------------------------------------------------------ nvidia-smi
def nvsmi_snapshot(dev_info: dict) -> dict:
    """対象 GPU の nvidia-smi 値（温度・クロック・VRAM・pstate）。失敗時は error 文字列。"""
    q = "pci.bus_id,name,temperature.gpu,clocks.sm,clocks.mem,memory.used,memory.total,pstate,power.draw"
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        if p.returncode != 0:
            return {"error": "rc=%d %s" % (p.returncode, p.stderr.strip()[:200])}
        keys = q.split(",")
        rows = [dict(zip(keys, [x.strip() for x in line.split(",")])) for line in p.stdout.strip().splitlines()]
        bus = dev_info.get("pci_bus_id_nv")
        pick = None
        for r in rows:
            m = re.match(r"[0-9a-fA-F]+:([0-9a-fA-F]{2}):", r["pci.bus_id"])
            if bus is not None and m and int(m.group(1), 16) == bus:
                pick = r
        if pick is None:
            nm = dev_info["name"].replace("GeForce", "").strip().lower()
            for r in rows:
                if nm and nm in r["name"].lower():
                    pick = r
        if pick is None:
            return {"error": "no matching nvidia-smi row"}
        out = {}
        for k, v in pick.items():
            out[k.replace("clocks.", "clock_").replace(".", "_")] = v
        t = pick.get("temperature.gpu", "")
        out["temp_c"] = int(t) if t.isdigit() else None
        return out
    except Exception as ex:  # noqa: BLE001
        return {"error": "%s: %s" % (type(ex).__name__, ex)}


# ------------------------------------------------------------ GEMM backends
class Gemm:
    """row-major: C[m,n] = op(A) @ op(B)。A は ta ? [k,m] : [m,k]、B は tb ? [n,k] : [k,n] で格納。"""

    def __init__(self, q, prg_kernels):
        self.q = q
        self.k = prg_kernels

    def clblast(self, m, n, k, a, b, c, ta=False, tb=False):
        a_ld = m if ta else k
        b_ld = k if tb else n
        return pyclblast.gemm(self.q, m, n, k, a, b, c, a_ld, b_ld, n, a_transp=ta, b_transp=tb)

    def custom(self, m, n, k, a, b, c, ta=False, tb=False):
        a_rs, a_cs = (1, m) if ta else (k, 1)
        b_rs, b_cs = (1, k) if tb else (n, 1)
        gsz = (_round_up(n, 16), _round_up(m, 16))
        return self.k["sgemm_tiled"](self.q, gsz, (16, 16), np.int32(m), np.int32(n), np.int32(k),
                                     a.data, np.int32(a_rs), np.int32(a_cs),
                                     b.data, np.int32(b_rs), np.int32(b_cs),
                                     c.data, np.int32(n), np.float32(1.0), np.float32(0.0))


def wait_event(q, ev, poll=0.004):
    """clFinish は busy-wait で CPU を 1 コア占有するので、イベントをポーリングして sleep する。"""
    q.flush()
    if ev is None:
        q.finish()
        return
    while ev.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) > 0:
        time.sleep(poll)
    # ERROR（負値）なら wait() が例外を出す
    ev.wait()


GEMM_SHAPES = [  # (m, n, k, 備考)
    (64, 576, 192, "GRU gates fwd (batch64, 3H=576, in=192)"),
    (4096, 384, 128, "token rows x 2H x emb"),
    (64, 192, 576, "dX 相当"),
    (192, 576, 64, "dW 相当 (k=batch)"),
    (128, 192, 4096, "dW 相当 (k=tokens)"),
    (256, 8192, 128, "vocab 方向"),
    (63, 577, 191, "境界（16 の倍数でない）"),
    (1, 576, 192, "m=1"),
]


def gemm_correctness(q, fn, shapes=GEMM_SHAPES) -> dict:
    rng = np.random.default_rng(1234)
    cases = []
    ok_all = True
    err = None
    for (m, n, k, note) in shapes:
        for ta in (False, True):
            for tb in (False, True):
                case = {"m": m, "n": n, "k": k, "ta": ta, "tb": tb, "note": note}
                try:
                    A = rng.standard_normal((k, m) if ta else (m, k)).astype(np.float32)
                    B = rng.standard_normal((n, k) if tb else (k, n)).astype(np.float32)
                    ref = (A.T if ta else A).astype(np.float64) @ (B.T if tb else B).astype(np.float64)
                    a = cla.to_device(q, A)
                    b = cla.to_device(q, B)
                    # 初期値は NaN にして「書かれていない」ことを検出する
                    c = cla.to_device(q, np.full((m, n), np.nan, dtype=np.float32))
                    ev = fn(m, n, k, a, b, c, ta, tb)
                    wait_event(q, ev)
                    got = c.get().astype(np.float64)
                    if not np.all(np.isfinite(got)):
                        case.update(ok=False, rel_err=None, error="non-finite output")
                    else:
                        rel = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))
                        case.update(ok=bool(rel < GEMM_TOL), rel_err=rel)
                except Exception as ex:  # noqa: BLE001
                    case.update(ok=False, error="%s: %s" % (type(ex).__name__, str(ex)[:300]))
                    err = err or case["error"]
                ok_all &= case["ok"]
                cases.append(case)
    worst = max([c["rel_err"] for c in cases if c.get("rel_err") is not None] or [None])
    return {"ok": ok_all, "n_cases": len(cases), "n_fail": sum(not c["ok"] for c in cases),
            "max_rel_err": worst, "first_error": err, "cases": cases}


def gemm_bench(q, fn, size=1024, reps=8) -> dict:
    """GFLOPS 計測（size^3 SGEMM を reps 回、最初の 1 回はウォームアップ）。"""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((size, size)).astype(np.float32)
    B = rng.standard_normal((size, size)).astype(np.float32)
    a, b = cla.to_device(q, A), cla.to_device(q, B)
    c = cla.to_device(q, np.zeros((size, size), np.float32))
    wait_event(q, fn(size, size, size, a, b, c, False, False))
    t0 = time.time()
    ev = None
    for _ in range(reps):
        ev = fn(size, size, size, a, b, c, False, False)
    wait_event(q, ev)
    dt = time.time() - t0
    return {"shape": [size, size, size], "reps": reps, "seconds": dt,
            "gflops": 2.0 * size ** 3 * reps / dt / 1e9}


# ------------------------------------------------------------ 基本演算
def _rel(got, ref) -> float:
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    if not np.all(np.isfinite(got)):
        return float("inf")
    return float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))


def run_ops(ctx, q, kern) -> dict:
    rng = np.random.default_rng(7)
    res = {}

    def record(name, fn):
        try:
            rel = fn()
            res[name] = {"ok": bool(rel < OP_TOL), "max_rel_err": rel}
        except Exception as ex:  # noqa: BLE001
            res[name] = {"ok": False, "error": "%s: %s" % (type(ex).__name__, str(ex)[:300]),
                         "trace": traceback.format_exc()[-600:]}

    def dev(x):
        return cla.to_device(q, x)

    def out(n, dtype=np.float32):
        return cla.to_device(q, np.full(n, np.nan, dtype=dtype))

    def run(name, gsz, lsz, *args):
        kern[name](q, (gsz,), (lsz,) if lsz else None, *args)
        q.finish()

    N = 100003
    x = (rng.standard_normal(N) * 3).astype(np.float32)
    dy = rng.standard_normal(N).astype(np.float32)
    ls = 128
    gs = _round_up(N, ls)

    def t_sigmoid():
        y = out(N)
        run("k_sigmoid", gs, ls, dev(x).data, y.data, np.int32(N))
        return _rel(y.get(), 1.0 / (1.0 + np.exp(-x.astype(np.float64))))

    def t_tanh():
        y = out(N)
        run("k_tanh", gs, ls, dev(x).data, y.data, np.int32(N))
        return _rel(y.get(), np.tanh(x.astype(np.float64)))

    ysig = (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)
    ytanh = np.tanh(x.astype(np.float64)).astype(np.float32)

    def t_dsigmoid():
        dx = out(N)
        run("k_dsigmoid", gs, ls, dev(ysig).data, dev(dy).data, dx.data, np.int32(N))
        return _rel(dx.get(), dy.astype(np.float64) * ysig * (1 - ysig.astype(np.float64)))

    def t_dtanh():
        dx = out(N)
        run("k_dtanh", gs, ls, dev(ytanh).data, dev(dy).data, dx.data, np.int32(N))
        return _rel(dx.get(), dy.astype(np.float64) * (1 - ytanh.astype(np.float64) ** 2))

    def t_mul():
        c = out(N)
        run("k_mul", gs, ls, dev(x).data, dev(dy).data, c.data, np.int32(N))
        return _rel(c.get(), x.astype(np.float64) * dy)

    def t_add():
        c = out(N)
        run("k_add", gs, ls, dev(x).data, dev(dy).data, c.data, np.int32(N))
        return _rel(c.get(), x.astype(np.float64) + dy)

    def softmax_ref(X):
        X = X.astype(np.float64)
        e = np.exp(X - X.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def t_softmax():
        worst = 0.0
        for rows, cols in ((64, 8192), (4096, 5), (37, 1000)):
            X = (rng.standard_normal((rows, cols)) * 4).astype(np.float32)
            y = cla.to_device(q, np.full((rows, cols), np.nan, np.float32))
            kern["softmax_rows"](q, (rows * 64,), (64,), dev(X.reshape(-1)).data, y.data,
                                 np.int32(cols), cl.LocalMemory(64 * 4))
            q.finish()
            worst = max(worst, _rel(y.get(), softmax_ref(X)))
        return worst

    def t_softmax_bwd():
        rows, cols = 128, 300
        X = (rng.standard_normal((rows, cols)) * 2).astype(np.float32)
        Y = softmax_ref(X).astype(np.float32)
        DY = rng.standard_normal((rows, cols)).astype(np.float32)
        dx = cla.to_device(q, np.full((rows, cols), np.nan, np.float32))
        kern["softmax_rows_bwd"](q, (rows * 64,), (64,), dev(Y.reshape(-1)).data, dev(DY.reshape(-1)).data,
                                 dx.data, np.int32(cols), cl.LocalMemory(64 * 4))
        q.finish()
        Y64 = Y.astype(np.float64)
        ref = Y64 * (DY - (DY * Y64).sum(1, keepdims=True))
        return _rel(dx.get(), ref)

    def t_reduce_sum():
        n = 1000003
        v = rng.random(n).astype(np.float32)
        groups, lsz = 64, 128
        partial = out(groups)
        kern["reduce_sum_partial"](q, (groups * lsz,), (lsz,), dev(v).data, partial.data, np.int32(n),
                                   cl.LocalMemory(lsz * 4))
        total = out(1)
        kern["reduce_sum_partial"](q, (lsz,), (lsz,), partial.data, total.data, np.int32(groups),
                                   cl.LocalMemory(lsz * 4))
        q.finish()
        return _rel(total.get()[0], v.astype(np.float64).sum())

    def t_col_sum():
        rows, cols = 4096, 576
        X = rng.standard_normal((rows, cols)).astype(np.float32)
        o = out(cols)
        run("col_sum", _round_up(cols, 64), 64, dev(X.reshape(-1)).data, o.data, np.int32(rows), np.int32(cols))
        return _rel(o.get(), X.astype(np.float64).sum(0))

    def t_scatter_add():
        worst = 0.0
        # (vocab, dim, n, idx の取りうる範囲) — 後者は重複を非常に多くしてアトミック競合を起こす
        for vocab, dim, n, span in ((8192, 128, 4096, 8192), (512, 128, 4096, 16)):
            idx = rng.integers(0, span, size=n).astype(np.int32)
            g = rng.standard_normal((n, dim)).astype(np.float32)
            ref = np.zeros((vocab, dim), np.float64)
            np.add.at(ref, idx, g.astype(np.float64))
            o = cla.to_device(q, np.zeros((vocab, dim), np.float32))
            run("scatter_add_rows", _round_up(n * dim, 128), 128, dev(idx).data, dev(g.reshape(-1)).data,
                o.data, np.int32(n), np.int32(dim))
            worst = max(worst, _rel(o.get(), ref))
        return worst

    def t_gather():
        vocab, dim, n = 8192, 128, 4096
        emb = rng.standard_normal((vocab, dim)).astype(np.float32)
        idx = rng.integers(0, vocab, size=n).astype(np.int32)
        o = out(n * dim)
        run("gather_rows", _round_up(n * dim, 128), 128, dev(emb.reshape(-1)).data, dev(idx).data, o.data,
            np.int32(n), np.int32(dim))
        return _rel(o.get().reshape(n, dim), emb[idx])

    def t_adamw():
        n = 100003
        lr, b1, b2, eps, wd = 2e-3, 0.9, 0.999, 1e-8, 0.01
        p = rng.standard_normal(n).astype(np.float32)
        m = np.zeros(n, np.float32)
        v = np.zeros(n, np.float32)
        pd, md, vd = dev(p), dev(m), dev(v)
        pr, mr, vr = p.copy(), m.copy(), v.copy()
        f = np.float32
        for t in range(1, 6):
            g = rng.standard_normal(n).astype(np.float32)
            bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
            run("adamw", _round_up(n, 128), 128, pd.data, dev(g).data, md.data, vd.data, np.int32(n),
                f(lr), f(b1), f(b2), f(eps), f(wd), f(bc1), f(bc2))
            mr = f(b1) * mr + (f(1) - f(b1)) * g
            vr = f(b2) * vr + (f(1) - f(b2)) * g * g
            mh, vh = mr / f(bc1), vr / f(bc2)
            pr = pr * (f(1) - f(lr) * f(wd)) - f(lr) * mh / (np.sqrt(vh) + f(eps))
        return max(_rel(pd.get(), pr), _rel(md.get(), mr), _rel(vd.get(), vr))

    for name, fn in (("sigmoid", t_sigmoid), ("tanh", t_tanh), ("dsigmoid", t_dsigmoid), ("dtanh", t_dtanh),
                     ("mul", t_mul), ("add", t_add), ("softmax_rows", t_softmax),
                     ("softmax_rows_bwd", t_softmax_bwd), ("reduce_sum", t_reduce_sum), ("col_sum", t_col_sum),
                     ("scatter_add_atomic_cmpxchg", t_scatter_add), ("gather_rows", t_gather),
                     ("adamw", t_adamw)):
        record(name, fn)
    return res


# ------------------------------------------------------------ 連続負荷
def load_test(q, gemm_fn, backend: str, dev_info: dict, minutes: float, jsonl_path: Path, size: int) -> dict:
    """SGEMM を回し続ける。1 enqueue 分のバッチを大きめにして CPU を使わない（ポーリング wait）。"""
    rng = np.random.default_rng(99)
    A = rng.standard_normal((size, size)).astype(np.float32)
    B = rng.standard_normal((size, size)).astype(np.float32)
    ref = (A @ B).astype(np.float64)
    a, b = cla.to_device(q, A), cla.to_device(q, B)
    c = cla.to_device(q, np.full((size, size), np.nan, np.float32))
    flops_per = 2.0 * size ** 3

    result = {"backend": backend, "size": size, "minutes_requested": minutes, "ok": True, "errors": [],
              "records": 0, "temp_max_c": None, "gflops_per_min": []}
    f = open(jsonl_path, "w")

    def emit(rec):
        f.write(json.dumps(rec) + "\n")
        f.flush()
        result["records"] += 1

    try:
        wait_event(q, gemm_fn(size, size, size, a, b, c, False, False))
        c0 = c.get()
        rel0 = float(np.max(np.abs(c0.astype(np.float64) - ref)) / np.max(np.abs(ref)))
        if not np.isfinite(c0).all() or rel0 > GEMM_TOL:
            raise RuntimeError("initial load GEMM wrong: rel_err=%g" % rel0)
        # キャリブレーション: 1 バッチ ~1 秒
        t0 = time.time()
        wait_event(q, gemm_fn(size, size, size, a, b, c, False, False))
        t1 = max(time.time() - t0, 1e-4)
        batch = int(min(max(1, 1.0 / t1), 400))
        result.update(single_call_s=t1, batch=batch, initial_rel_err=rel0)

        t_start = time.time()
        t_end = t_start + minutes * 60.0
        t_win = t_start
        win_calls = 0
        win_temp_max = None
        last_sample = 0.0
        minute_idx = 0
        bit_mismatch_events = 0
        while True:
            now = time.time()
            if now - last_sample >= 15.0:
                s = nvsmi_snapshot(dev_info)
                if s.get("temp_c") is not None:
                    win_temp_max = s["temp_c"] if win_temp_max is None else max(win_temp_max, s["temp_c"])
                last_sample = now
            done = now >= t_end
            if done or now - t_win >= 60.0:
                # チェックポイント: C を NaN で潰してから 1 回計算し直し、結果を初回結果と照合
                c.set(np.full((size, size), np.nan, np.float32))
                wait_event(q, gemm_fn(size, size, size, a, b, c, False, False))
                got = c.get()
                win_calls += 1
                elapsed_win = time.time() - t_win
                finite = bool(np.isfinite(got).all())
                bit_equal = bool(finite and np.array_equal(got, c0))
                rel = float(np.max(np.abs(got.astype(np.float64) - ref)) / np.max(np.abs(ref))) if finite else None
                snap = nvsmi_snapshot(dev_info)
                if snap.get("temp_c") is not None:
                    win_temp_max = snap["temp_c"] if win_temp_max is None else max(win_temp_max, snap["temp_c"])
                gflops = win_calls * flops_per / elapsed_win / 1e9
                minute_idx += 1
                rec = {"t": _now(), "minute": minute_idx, "elapsed_s": round(time.time() - t_start, 1),
                       "window_s": round(elapsed_win, 1), "gemm_calls": win_calls, "gflops": round(gflops, 2),
                       "checksum_sum": float(np.sum(got, dtype=np.float64)) if finite else None,
                       "checksum_abs": float(np.sum(np.abs(got), dtype=np.float64)) if finite else None,
                       "bitwise_equal_to_first": bit_equal, "rel_err_vs_numpy": rel,
                       "temp_max_in_window_c": win_temp_max, "nvidia_smi": snap}
                emit(rec)
                result["gflops_per_min"].append(round(gflops, 2))
                if win_temp_max is not None:
                    result["temp_max_c"] = win_temp_max if result["temp_max_c"] is None else max(result["temp_max_c"], win_temp_max)
                if not bit_equal:
                    bit_mismatch_events += 1
                if (not finite) or rel is None or rel > GEMM_TOL:
                    result["ok"] = False
                    result["errors"].append("minute %d: result mismatch/non-finite (rel_err=%s)" % (minute_idx, rel))
                    break
                t_win = time.time()
                win_calls = 0
                win_temp_max = None
                if done:
                    break
                continue
            ev = None
            for _ in range(batch):
                ev = gemm_fn(size, size, size, a, b, c, False, False)
            wait_event(q, ev, poll=0.01)
            win_calls += batch
        result["bitwise_mismatch_checkpoints"] = bit_mismatch_events
        result["elapsed_s"] = round(time.time() - t_start, 1)
        if result["gflops_per_min"]:
            g = result["gflops_per_min"]
            result["gflops_min"], result["gflops_max"] = min(g), max(g)
            result["gflops_mean"] = round(sum(g) / len(g), 2)
    except Exception as ex:  # noqa: BLE001
        result["ok"] = False
        result["errors"].append("%s: %s" % (type(ex).__name__, str(ex)[:400]))
        result["trace"] = traceback.format_exc()[-1200:]
    finally:
        f.close()
    return result


# ------------------------------------------------------------ main
def run_gate(device: str, out_dir, minutes: float = 0.0, load_size: int = 1024) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", device).strip("_")
    path = out_dir / ("gate_%s.json" % tag)
    jsonl = out_dir / ("gate_%s_load.jsonl" % tag)
    R: dict = {"device_query": device, "started": _now(), "status": "FAIL", "errors": [],
               "python": sys.version.split()[0]}

    def _on_term(signum, frame):  # 外部からの停止でも結果 JSON を確定させる
        raise RuntimeError("terminated by signal %d" % signum)
    try:
        signal.signal(signal.SIGTERM, _on_term)
    except Exception:  # noqa: BLE001
        pass

    def save():
        path.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str) + "\n")

    try:
        p, d = find_device(device)
        R["device"] = device_info(p, d)
        R["opencl_version"] = d.version
        ctx = cl.Context([d])
        q = cl.CommandQueue(ctx)
        R["nvidia_smi_start"] = nvsmi_snapshot(R["device"])
        save()

        prg = cl.Program(ctx, KERNEL_SRC).build()
        kern = {name: cl.Kernel(prg, name) for name in (
            "sgemm_tiled", "k_sigmoid", "k_tanh", "k_dsigmoid", "k_dtanh", "k_mul", "k_add", "softmax_rows",
            "softmax_rows_bwd", "reduce_sum_partial", "col_sum", "scatter_add_rows", "gather_rows", "adamw")}
        g = Gemm(q, kern)
        R["kernels_built"] = True

        # (1) GEMM
        backends = {}
        if pyclblast is not None:
            backends["clblast"] = g.clblast
        else:
            R["gemm_clblast"] = {"ok": False, "first_error": "pyclblast import failed: " + _CLBLAST_IMPORT_ERR}
        backends["custom"] = g.custom
        gflops = {}
        for name, fn in backends.items():
            try:
                r = gemm_correctness(q, fn)
            except Exception as ex:  # noqa: BLE001
                r = {"ok": False, "first_error": "%s: %s" % (type(ex).__name__, ex)}
            if r.get("ok"):
                try:
                    gflops[name] = gemm_bench(q, fn)
                except Exception as ex:  # noqa: BLE001
                    r["bench_error"] = "%s: %s" % (type(ex).__name__, ex)
            R["gemm_" + name] = r
            save()
        R["gflops"] = {k: round(v["gflops"], 2) for k, v in gflops.items()}
        R["gflops_detail"] = gflops
        used = "clblast" if R.get("gemm_clblast", {}).get("ok") else ("custom" if R["gemm_custom"].get("ok") else "none")
        R["gemm_backend"] = used
        R["clblast_ok"] = bool(R.get("gemm_clblast", {}).get("ok"))

        # (2) 基本演算
        R["ops"] = run_ops(ctx, q, kern)
        R["ops_ok"] = all(v["ok"] for v in R["ops"].values())
        save()

        # (3) 連続負荷
        if minutes and minutes > 0 and used != "none":
            R["nvidia_smi_before_load"] = nvsmi_snapshot(R["device"])
            R["load"] = load_test(q, backends[used], used, R["device"], minutes, jsonl, load_size)
            R["nvidia_smi_after_load"] = nvsmi_snapshot(R["device"])
        elif minutes and minutes > 0:
            R["load"] = {"ok": False, "errors": ["no working GEMM backend; load test skipped"]}

        ok = used != "none" and R["ops_ok"] and (not R.get("load") or R["load"].get("ok"))
        R["status"] = "PASS" if ok else "FAIL"
        if not R["ops_ok"]:
            R["errors"] += ["op failed: %s" % k for k, v in R["ops"].items() if not v["ok"]]
        if used == "none":
            R["errors"].append("no GEMM backend passed")
        if R.get("load") and not R["load"].get("ok"):
            R["errors"] += R["load"].get("errors", [])
    except Exception as ex:  # noqa: BLE001
        R["status"] = "FAIL"
        R["errors"].append("%s: %s" % (type(ex).__name__, str(ex)[:500]))
        R["trace"] = traceback.format_exc()[-1500:]
    R["finished"] = _now()
    save()
    return R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OpenCL gate test")
    ap.add_argument("--device", required=True, help='OpenCL デバイス名の部分一致（例 "GT 430"）')
    ap.add_argument("--out", default="runs/hw")
    ap.add_argument("--minutes", type=float, default=0.0, help="連続負荷の分数（0 でスキップ）")
    ap.add_argument("--load-size", type=int, default=1024, help="連続負荷の SGEMM 一辺")
    a = ap.parse_args(argv)
    R = run_gate(a.device, a.out, a.minutes, a.load_size)
    print("device     :", R.get("device", {}).get("name"), "|", R.get("opencl_version"))
    print("gemm       :", R.get("gemm_backend"), "clblast_ok=%s" % R.get("clblast_ok"), "gflops=%s" % R.get("gflops"))
    print("ops        :", {k: v["ok"] for k, v in R.get("ops", {}).items()})
    if "load" in R:
        L = R["load"]
        print("load       : ok=%s minutes=%s temp_max=%s gflops_mean=%s errors=%s" % (
            L.get("ok"), L.get("minutes_requested"), L.get("temp_max_c"), L.get("gflops_mean"), L.get("errors")))
    print("errors     :", R.get("errors"))
    print("STATUS     :", R["status"])
    return 0 if R["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
