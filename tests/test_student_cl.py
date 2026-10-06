"""OpenCL backend と numpy backend の一致テスト（pyopencl/pyclblast とデバイスが無ければ skip）。

環境変数: TB250_CL_DEVICE（デバイス名の部分一致、既定 "GT 730"）、TB250_CL_FULL=1 で Common-S 実寸の比較も実行。
許容誤差（FP32 の OpenCL vs float64 numpy）:
  op 単体: 相対 1e-5 程度 / forward scores: 絶対 1e-4 / 勾配: 全体 L2 相対 2e-3、パラメータ種類別 L2 相対 5e-3
  1 step 後の weight: |g|>1e-5 の要素で絶対 2e-5（AdamW 初回更新は ±lr*sign(g) なので、|g| が丸め誤差級の要素は符号が反転しうる）
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip("pyopencl")
pytest.importorskip("pyclblast")
from tb250distill.student import model as M  # noqa: E402
from tb250distill.student.backend_np import NPBackend  # noqa: E402
from tb250distill.student.backend_cl import CLBackend  # noqa: E402

DEVICE = os.environ.get("TB250_CL_DEVICE", "GT 730")


@pytest.fixture(scope="module")
def cl_be():
    try:
        return CLBackend(DEVICE)
    except RuntimeError as e:
        pytest.skip(str(e))


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(1e-30, np.linalg.norm(b)))


def to_dev(be, x, kind="f"):
    a = be.alloc(x.shape, kind)
    be.upload(a, x)
    return a


def test_gemm_variants(cl_be):
    rng = np.random.default_rng(0)
    for (m, n, k, ta, tb, alpha, beta) in [(7, 5, 3, False, False, 1.0, 0.0), (7, 5, 3, True, False, 1.0, 1.0),
                                          (7, 5, 3, False, True, 0.5, 1.0), (33, 64, 17, True, True, 1.0, 0.0),
                                          (32, 576, 192, False, False, 1.0, 0.0), (1, 40, 300, False, False, 1.0, 1.0),
                                          (200, 1, 64, False, False, 1.0, 0.0), (64, 192, 576, False, True, 1.0, 1.0)]:
        A = rng.normal(size=(k, m) if ta else (m, k)).astype(np.float32)
        B = rng.normal(size=(n, k) if tb else (k, n)).astype(np.float32)
        C0 = rng.normal(size=(m, n)).astype(np.float32)
        ref = alpha * ((A.T if ta else A).astype(np.float64) @ (B.T if tb else B).astype(np.float64)) + beta * C0
        # offset 付き view も検証: 大きい buffer の途中から
        big = cl_be.alloc((m * n + 11,))
        cl_be.upload(big, np.concatenate([np.zeros(11, np.float32), C0.ravel()]))
        Cv = cl_be.view(big, 11, (m, n))
        cl_be.gemm(to_dev(cl_be, A), to_dev(cl_be, B), Cv, transA=ta, transB=tb, alpha=alpha, beta=beta)
        got = cl_be.download(big).reshape(-1)[11:11 + m * n].reshape(m, n)
        assert rel(got, ref) < 1e-5, (m, n, k, ta, tb, rel(got, ref))


def test_embedding_ops(cl_be):
    rng = np.random.default_rng(1)
    V, d, n = 50, 8, 400
    tab = rng.normal(size=(V, d)).astype(np.float32)
    idx = rng.integers(0, V, n).astype(np.int32)
    idx[:100] = 3  # 重複を増やして atomic を叩く
    src = rng.normal(size=(n, d)).astype(np.float32)
    src[200:230] = 0
    out = cl_be.alloc((n, d))
    cl_be.gather_rows(out, to_dev(cl_be, tab), to_dev(cl_be, idx, "i"))
    assert np.array_equal(cl_be.download(out), tab[idx])
    g0 = rng.normal(size=(V, d)).astype(np.float32)
    g = to_dev(cl_be, g0)
    cl_be.scatter_add_rows(g, to_dev(cl_be, src), to_dev(cl_be, idx, "i"))
    ref = g0.astype(np.float64)
    np.add.at(ref, idx, src)
    assert rel(cl_be.download(g), ref) < 1e-6
    # group_sum / colsum
    B, K, h = 6, 5, 9
    s2 = rng.normal(size=(B * K, h)).astype(np.float32)
    dst = cl_be.alloc((B, h))
    cl_be.group_sum(dst, to_dev(cl_be, s2), K)
    assert rel(cl_be.download(dst), s2.reshape(B, K, h).sum(1)) < 1e-6
    acc0 = rng.normal(size=(h,)).astype(np.float32)
    acc = to_dev(cl_be, acc0)
    cl_be.colsum(acc, to_dev(cl_be, s2))
    assert rel(cl_be.download(acc), acc0 + s2.sum(0)) < 1e-5


def test_gru_ops_and_loss_vs_np(cl_be):
    rng = np.random.default_rng(2)
    npb = NPBackend(np.float64)
    B, H = 11, 13
    f = lambda *s: rng.normal(size=s).astype(np.float32)
    gi, gh, bi, bh, hp = f(B, 3 * H), f(B, 3 * H), f(3 * H), f(3 * H), f(B, H)
    lens = rng.integers(0, 6, B).astype(np.int32)
    t = 3
    ho_np = np.zeros((B, H))
    npb.gru_fwd(gi.astype(np.float64), gh.astype(np.float64), bi.astype(np.float64), bh.astype(np.float64),
                hp.astype(np.float64), ho_np, lens, t)
    ho = cl_be.alloc((B, H))
    cl_be.gru_fwd(to_dev(cl_be, gi), to_dev(cl_be, gh), to_dev(cl_be, bi), to_dev(cl_be, bh), to_dev(cl_be, hp), ho,
                  to_dev(cl_be, lens, "i"), t)
    assert np.abs(cl_be.download(ho) - ho_np).max() < 1e-5
    # backward
    dhc0, dout = f(B, H), f(B, H)
    for has in (True, False):
        dgi_n, dgh_n, dhc_n = np.zeros((B, 3 * H)), np.zeros((B, 3 * H)), dhc0.astype(np.float64).copy()
        npb.gru_bwd(gi.astype(np.float64), gh.astype(np.float64), bi.astype(np.float64), bh.astype(np.float64),
                    hp.astype(np.float64), dhc_n, dout.astype(np.float64) if has else None, dgi_n, dgh_n, lens, t)
        dgi, dgh, dhc = cl_be.alloc((B, 3 * H)), cl_be.alloc((B, 3 * H)), to_dev(cl_be, dhc0)
        cl_be.gru_bwd(to_dev(cl_be, gi), to_dev(cl_be, gh), to_dev(cl_be, bi), to_dev(cl_be, bh), to_dev(cl_be, hp),
                      dhc, to_dev(cl_be, dout) if has else None, dgi, dgh, to_dev(cl_be, lens, "i"), t)
        assert np.abs(cl_be.download(dgi) - dgi_n).max() < 1e-5
        assert np.abs(cl_be.download(dgh) - dgh_n).max() < 1e-5
        assert np.abs(cl_be.download(dhc) - dhc_n).max() < 1e-5
    # kd_loss（gold 有/無、k 可変）
    B, K = 9, 5
    sc, tl = f(B, K) * 2, f(B, K) * 2
    k = rng.integers(1, K + 1, B).astype(np.int32)
    gold = np.where(rng.random(B) < 0.6, rng.integers(0, 10, B) % k, -1).astype(np.int32)
    ds_n, ls_n = np.zeros(B * K), np.zeros(B * 3)
    npb.kd_loss(sc.ravel().astype(np.float64), tl.ravel().astype(np.float64), k, gold, ds_n, ls_n, B, K, 2.0, 0.8, 0.2, 1.0 / B)
    ds, ls = cl_be.alloc((B * K,)), cl_be.alloc((B * 3,))
    cl_be.kd_loss(to_dev(cl_be, sc.ravel()), to_dev(cl_be, tl.ravel()), to_dev(cl_be, k, "i"), to_dev(cl_be, gold, "i"),
                  ds, ls, B, K, 2.0, 0.8, 0.2, 1.0 / B)
    assert np.abs(cl_be.download(ds) - ds_n).max() < 1e-5
    assert np.abs(cl_be.download(ls) - ls_n).max() < 1e-5
    # sample ごとの重み配列（--source-loss）。np と一致、スカラー指定の再呼び出し（キャッシュ経路）も壊れない
    wk, wc = rng.random(B).astype(np.float32), rng.random(B).astype(np.float32)
    ds_n2, ls_n2 = np.zeros(B * K), np.zeros(B * 3)
    npb.kd_loss(sc.ravel().astype(np.float64), tl.ravel().astype(np.float64), k, gold, ds_n2, ls_n2, B, K, 2.0,
                wk.astype(np.float64), wc.astype(np.float64), 1.0 / B)
    ds2, ls2 = cl_be.alloc((B * K,)), cl_be.alloc((B * 3,))
    cl_be.kd_loss(to_dev(cl_be, sc.ravel()), to_dev(cl_be, tl.ravel()), to_dev(cl_be, k, "i"), to_dev(cl_be, gold, "i"),
                  ds2, ls2, B, K, 2.0, to_dev(cl_be, wk), to_dev(cl_be, wc), 1.0 / B)
    assert np.abs(cl_be.download(ds2) - ds_n2).max() < 1e-5
    assert np.abs(cl_be.download(ls2) - ls_n2).max() < 1e-5
    cl_be.kd_loss(to_dev(cl_be, sc.ravel()), to_dev(cl_be, tl.ravel()), to_dev(cl_be, k, "i"), to_dev(cl_be, gold, "i"),
                  ds, ls, B, K, 2.0, 0.8, 0.2, 1.0 / B)
    assert np.abs(cl_be.download(ds) - ds_n).max() < 1e-5
    assert np.abs(cl_be.download(ls) - ls_n).max() < 1e-5


def test_adamw_and_sumsq_vs_np(cl_be):
    rng = np.random.default_rng(3)
    n = 100003
    npb = NPBackend(np.float32)
    p0, g0 = rng.normal(size=n).astype(np.float32), (rng.normal(size=n) * 3).astype(np.float32)
    m0, v0 = rng.normal(size=n).astype(np.float32) * 0.1, np.abs(rng.normal(size=n)).astype(np.float32) * 0.1
    ss_n = np.zeros(1, np.float32)
    npb.sumsq(g0, ss_n)
    ss = cl_be.alloc((1,))
    gd = to_dev(cl_be, g0)
    cl_be.sumsq(gd, ss)
    assert abs(cl_be.download(ss)[0] - ss_n[0]) / ss_n[0] < 1e-5
    pn, mn, vn = p0.copy(), m0.copy(), v0.copy()
    npb.adamw(pn, g0, mn, vn, ss_n, 2e-3, 0.9, 0.999, 1e-8, 0.01, 7, 1.0)
    pd, md, vd = to_dev(cl_be, p0), to_dev(cl_be, m0), to_dev(cl_be, v0)
    cl_be.adamw(pd, gd, md, vd, ss, 2e-3, 0.9, 0.999, 1e-8, 0.01, 7, 1.0)
    assert np.abs(cl_be.download(pd).ravel() - pn).max() < 1e-6
    assert np.abs(cl_be.download(md).ravel() - mn).max() < 1e-6
    assert np.abs(cl_be.download(vd).ravel() - vn).max() < 1e-6


def _tiny_shard(n, vocab, lp, lc, kmax, seed):
    rng = np.random.default_rng(seed)
    d = dict(item_id=np.arange(n, dtype=np.int64), prefix=np.zeros((n, lp), np.int32), prefix_len=np.zeros(n, np.int32),
             cand=np.zeros((n, kmax, lc), np.int32), cand_len=np.zeros((n, kmax), np.int32), k=np.zeros(n, np.int32),
             t_logits=np.zeros((n, kmax), np.float32), gold=np.full(n, -1, np.int32))
    for i in range(n):
        pl = int(rng.integers(max(1, lp // 3), lp + 1)) if i % 5 else lp
        d["prefix_len"][i] = pl
        d["prefix"][i, :pl] = rng.integers(5, vocab, pl)
        k = int(rng.integers(2, kmax + 1))
        d["k"][i] = k
        for j in range(k):
            cl = int(rng.integers(1, lc + 1))
            d["cand_len"][i, j] = cl
            d["cand"][i, j, :cl] = rng.integers(5, vocab, cl)
        d["t_logits"][i, :k] = rng.normal(size=k) * 2
        d["gold"][i] = int(rng.integers(0, k)) if i % 3 else -1
    return M.Shard(d)


def _tiny_sem(sh, d, seed, U=17, invalid_frac=0.3):
    rng = np.random.default_rng(seed + 100)
    ci = rng.integers(0, U, size=(sh.n, sh.kmax)).astype(np.int32)
    ci[rng.random(ci.shape) < invalid_frac] = -1             # embedding 無しの候補
    ci[np.arange(sh.kmax)[None, :] >= sh.k[:, None]] = -1    # 候補無しの位置
    return M.SemTargets(ci, rng.normal(size=(U, d)).astype(np.float32))


def compare_models(cl_be, cfg, B, kmax, lp, lc, seed=0, scale=1.0, source_loss=None, sem_lambda=0.0):
    sh = _tiny_shard(B, cfg.vocab, lp, lc, kmax, seed)
    if source_loss:  # sample ごとの損失重み（--source-loss）経路。source を交互に割り当てる
        d = {k: getattr(sh, k) for k in M.SHARD_KEYS}
        d["source"] = np.where(np.arange(B) % 2 == 0, "synth", "jnli").astype("U16")
        sh = M.Shard(d)
    sem = _tiny_sem(sh, cfg.sem_dim, seed) if sem_lambda > 0 else None
    bt = M.make_batch(sh, np.arange(B), keys=np.random.default_rng(seed).random((B, kmax)), source_loss=source_loss, sem=sem)
    assert (bt.w_kd is not None) == bool(source_loss)
    params = M.init_params(cfg, seed)
    params = {k: v * np.float32(scale if k != "emb" else 1.0) for k, v in params.items()}
    ref_be = NPBackend(np.float64)
    ref = M.Student(ref_be, cfg, B, kmax, train=True, max_lp=lp, max_lc=lc)
    ref.set_params({k: v.astype(np.float64) for k, v in params.items()})
    cl = M.Student(cl_be, cfg, B, kmax, train=True, max_lp=lp, max_lc=lc)
    cl.set_params(params)
    ref.sem_lambda = cl.sem_lambda = sem_lambda
    out = {}
    # forward
    s_ref, s_cl = ref.predict(bt), cl.predict(bt)
    mask = np.arange(bt.K)[None, :] < bt.kcnt[:, None]
    out["fwd_max_abs"] = float(np.abs(np.where(mask, s_ref - s_cl, 0)).max())
    # 勾配
    r_ref, r_cl = ref.loss_grads(bt), cl.loss_grads(bt)
    out["loss_abs"] = abs(r_ref["loss"] - r_cl["loss"])
    out["sem_abs"] = abs(r_ref["sem"] - r_cl["sem"])
    g_ref, g_cl = ref.get_grads(), cl.get_grads()
    cat = lambda g: np.concatenate([g[n].ravel() for n, _ in M.param_specs(cfg)])
    out["grad_rel_l2_total"] = rel(cat(g_cl), cat(g_ref))
    # 種類別: 全体 L2 の 1e-4 倍を分母の床にする（head.b2 は softmax のシフト不変性で解析的に勾配 0）
    floor = 1e-4 * np.linalg.norm(cat(g_ref))
    out["grad_rel_l2_by_param"] = {n: float(np.linalg.norm(g_cl[n] - g_ref[n]) / max(floor, np.linalg.norm(g_ref[n])))
                                   for n, _ in M.param_specs(cfg)}
    # 1 step 後の weight（同じ batch、AdamW step=1）
    ref.train_step(bt, 2e-3, 1)
    cl.train_step(bt, 2e-3, 1)
    p_ref, p_cl = ref.get_params(), cl.get_params()
    wcat = lambda p: np.concatenate([p[n].ravel() for n, _ in M.param_specs(cfg)])
    gflat = cat(g_ref)
    d = np.abs(wcat(p_cl) - wcat(p_ref))
    big = np.abs(gflat) > 1e-5
    out["w1_max_abs_big_grad"] = float(d[big].max()) if big.any() else 0.0
    out["w1_mean_abs"] = float(d.mean())
    out["w1_frac_gt_1e-5"] = float((d > 1e-5).mean())
    return out


def test_model_matches_numpy_small(cl_be):
    cfg = M.Config(vocab=200, emb=24, hidden=32, layers=2, lp=20, lc=6)
    o = compare_models(cl_be, cfg, B=8, kmax=4, lp=20, lc=6, scale=2.0)
    print("small", o)
    assert o["fwd_max_abs"] < 1e-4 and o["loss_abs"] < 1e-5
    assert o["grad_rel_l2_total"] < 2e-3
    assert max(o["grad_rel_l2_by_param"].values()) < 5e-3
    assert o["w1_max_abs_big_grad"] < 2e-5


def test_cos_loss_op_vs_np(cl_be):
    """cos_loss kernel（行ごとの 1-cos と dL/dz、w==0 の行は 0）が numpy 版と一致。"""
    rng = np.random.default_rng(3)
    N, d = 37, 24
    z = rng.normal(size=(N, d)).astype(np.float32)
    t = rng.normal(size=(N, d)).astype(np.float32)
    t[5] = 0.0
    w = (rng.random(N) * 0.1).astype(np.float32)
    w[[0, 7, 20]] = 0.0
    npb = NPBackend(np.float64)
    dz_ref, l_ref = np.zeros((N, d)), np.zeros(N)
    npb.cos_loss(z.astype(np.float64), t.astype(np.float64), w.astype(np.float64), dz_ref, l_ref, N, d)
    dz, lr = cl_be.alloc((N, d)), cl_be.alloc((N,))
    cl_be.cos_loss(to_dev(cl_be, z), to_dev(cl_be, t), to_dev(cl_be, w), dz, lr, N, d)
    assert np.abs(cl_be.download(lr).ravel() - l_ref).max() < 1e-5
    assert np.abs(cl_be.download(dz) - dz_ref).max() < 1e-5 * max(1.0, np.abs(dz_ref).max())
    assert (cl_be.download(dz)[[0, 7, 20]] == 0).all() and cl_be.download(lr)[[0, 7, 20]].tolist() == [0, 0, 0]


def test_model_matches_numpy_sem(cl_be):
    """Candidate Semantic Distillation（sem loss 込み）の loss・全パラメータ勾配（projection head 含む）・1 step 後 weight が np float64 と一致。"""
    cfg = M.Config(vocab=200, emb=24, hidden=32, layers=2, lp=20, lc=6, sem_dim=12)
    o = compare_models(cl_be, cfg, B=8, kmax=4, lp=20, lc=6, scale=2.0, sem_lambda=0.5)
    print("small+sem", o)
    assert o["fwd_max_abs"] < 1e-4 and o["loss_abs"] < 1e-5 and o["sem_abs"] < 1e-5
    assert o["grad_rel_l2_total"] < 2e-3
    assert max(o["grad_rel_l2_by_param"].values()) < 5e-3
    assert {"sem.wp", "sem.bp"} <= set(o["grad_rel_l2_by_param"])
    assert o["w1_max_abs_big_grad"] < 2e-5


def test_cl_sem_lambda_zero_launches_no_sem_kernels(cl_be):
    """λ=0 では cos_loss を含む追加 kernel を一切起動しない（追加 forward なし）。"""
    cfg = M.Config(vocab=200, emb=24, hidden=32, layers=2, lp=20, lc=6, sem_dim=12)
    sh = _tiny_shard(8, cfg.vocab, 20, 6, 4, 1)
    cl = M.Student(cl_be, cfg, 8, 4, train=True, max_lp=20, max_lc=6)
    cl.set_params(M.init_params(cfg, 1))
    calls = []
    cl_be.events = calls     # (kernel 名, event) を記録（PROFILING 無しでも名前は取れる）
    try:
        cl.sem_lambda = 0.0
        cl.train_step(M.make_batch(sh, np.arange(8), sem=_tiny_sem(sh, 12, 1)), 2e-3, 1)
        n0 = len(calls)
        assert not any(n == "cos_loss" for n, _ in calls)
        cl.sem_lambda = 0.5
        cl.train_step(M.make_batch(sh, np.arange(8), sem=_tiny_sem(sh, 12, 1)), 2e-3, 2)
        assert sum(n == "cos_loss" for n, _ in calls) == 1 and len(calls) > 1.3 * n0
    finally:
        cl_be.events = None


def test_model_matches_numpy_source_loss(cl_be):
    """--source-loss の sample ごと重みで np float64 と一致（loss・勾配・1 step 後の weight）。"""
    cfg = M.Config(vocab=200, emb=24, hidden=32, layers=2, lp=20, lc=6)
    o = compare_models(cl_be, cfg, B=8, kmax=4, lp=20, lc=6, scale=2.0, source_loss={"jnli": (0.2, 0.8)})
    print("small+source_loss", o)
    assert o["fwd_max_abs"] < 1e-4 and o["loss_abs"] < 1e-5
    assert o["grad_rel_l2_total"] < 2e-3
    assert max(o["grad_rel_l2_by_param"].values()) < 5e-3
    assert o["w1_max_abs_big_grad"] < 2e-5


@pytest.mark.skipif(not os.environ.get("TB250_CL_FULL"), reason="TB250_CL_FULL=1 で Common-S 実寸比較")
def test_model_matches_numpy_common_s(cl_be):
    cfg = M.get_config("common_s")
    o = compare_models(cl_be, cfg, B=16, kmax=5, lp=128, lc=16)
    print("common_s", o)
    assert o["fwd_max_abs"] < 1e-4 and o["loss_abs"] < 1e-5
    assert o["grad_rel_l2_total"] < 2e-3
    assert max(o["grad_rel_l2_by_param"].values()) < 5e-3
    assert o["w1_max_abs_big_grad"] < 2e-5


@pytest.mark.skipif(not os.environ.get("TB250_CL_FULL"), reason="TB250_CL_FULL=1 で Common-S 実寸比較")
def test_model_matches_numpy_common_s_sem(cl_be):
    cfg = M.get_config("common_s")
    cfg.sem_dim = 128
    o = compare_models(cl_be, cfg, B=16, kmax=5, lp=128, lc=16, sem_lambda=0.5)
    print("common_s+sem", o)
    assert o["fwd_max_abs"] < 1e-4 and o["loss_abs"] < 1e-5 and o["sem_abs"] < 1e-5
    assert o["grad_rel_l2_total"] < 2e-3
    assert max(o["grad_rel_l2_by_param"].values()) < 5e-3
    assert o["w1_max_abs_big_grad"] < 2e-5


def test_inference_mode_buffers_shared(cl_be):
    """train=False（層間で buffer 共有）の forward が train=True と一致、B が変わる連続バッチでも安定。"""
    cfg = M.Config(vocab=200, emb=24, hidden=32, layers=3, lp=20, lc=6)
    sh = _tiny_shard(12, cfg.vocab, 20, 6, 4, 5)
    params = M.init_params(cfg, 5)
    a = M.Student(cl_be, cfg, 12, 4, train=True, max_lp=20, max_lc=6)
    b = M.Student(cl_be, cfg, 12, 4, train=False, max_lp=20, max_lc=6)
    a.set_params(params)
    b.set_params(params)
    full = M.make_batch(sh, np.arange(12))
    ref = a.predict(full)
    b.predict(M.make_batch(sh, [1, 2, 3]))
    got = b.predict(full)
    mask = np.arange(4)[None, :] < full.kcnt[:, None]
    assert np.abs(np.where(mask, ref - got, 0)).max() < 1e-5
