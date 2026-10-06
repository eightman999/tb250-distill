"""SEMFIX: 意味表現の独立ミニバッチと損失（cos / mse / rkd / infonce）、sem-only 学習、SIGTERM 保存。

- host 損失（float64）の数値勾配チェック（重複文字列・マスク込み）
- 独立ミニバッチ pass の Student 全パラメータ（head・GRU・embedding）の数値勾配チェック（np float64）
- 判断 loss と独立 pass の勾配の加法性、sem_ext 無し（従来経路）はビット一致
- サンプラ（決定性・層別・状態なし）、train.py / sem_only.py の結線・resume・検証・SIGTERM
- np / cl 一致（pyopencl とデバイスがあるときだけ。TB250_CL_DEVICE）
"""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill.student import model as M, train as T, semfix as SF, sem_only as SO, fake_data as F  # noqa: E402
from tb250distill.student.backend_np import NPBackend  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------
# host 損失の数値勾配
# ---------------------------------------------------------------------------------------

def num_grad(f, z, eps=1e-6):
    g = np.zeros_like(z)
    it = np.nditer(z, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        a, b = z.copy(), z.copy()
        a[i] += eps
        b[i] -= eps
        g[i] = (f(a) - f(b)) / (2 * eps)
    return g


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(1e-12, np.linalg.norm(b)))


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    N, d, D = 9, 6, 11
    z = rng.normal(size=(N, d))
    t = rng.normal(size=(N, d))
    e = rng.normal(size=(N, D))
    e = e - e.mean(0)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    uidx = np.arange(N)
    uidx[4] = uidx[1]      # 重複文字列（1 と 4 は同一）
    uidx[8] = uidx[1]
    return z, t, e, uidx


def test_cos_mse_match_backend_cos_loss_and_gradient(toy):
    z, t, e, uidx = toy
    N, d = z.shape
    be = NPBackend(np.float64)
    dz_ref, l_ref = np.zeros((N, d)), np.zeros(N)
    be.cos_loss(z, t, np.full(N, 1.0 / N), dz_ref, l_ref, N, d)      # w = 1/N（λ=1 の平均）
    l, dz = SF.loss_cos(z, t)
    assert abs(l - l_ref.mean()) < 1e-12      # lrow は行ごとの 1-cos（w は勾配だけに掛かる）
    assert np.abs(dz - dz_ref).max() < 1e-12
    assert rel(dz, num_grad(lambda x: SF.loss_cos(x, t)[0], z)) < 1e-6
    lm, dzm = SF.loss_mse(z, t)
    assert abs(lm - 2.0 * l) < 1e-12 and np.abs(dzm - 2.0 * dz).max() < 1e-12     # mse = 2 * cos（方向は同じ）
    assert rel(dzm, num_grad(lambda x: SF.loss_mse(x, t)[0], z)) < 1e-6


def test_rkd_gradient_and_properties(toy):
    z, t, e, uidx = toy
    excl = uidx[:, None] == uidx[None, :]
    l, dz = SF.loss_rkd(z, e, excl, 0.1, 0.2)
    assert l > 0
    assert rel(dz, num_grad(lambda x: SF.loss_rkd(x, e, excl, 0.1, 0.2)[0], z)) < 1e-6
    # Student の類似度行列が Teacher と同一（z = e）で tau も同じなら KL = 0 / 勾配 0
    l0, dz0 = SF.loss_rkd(e, e, excl, 0.1, 0.1)
    assert abs(l0) < 1e-12 and np.abs(dz0).max() < 1e-10
    # スケール・共通ベクトルの平行移動不変ではないが、スケール不変（cosine）
    l2, _ = SF.loss_rkd(3.0 * z, e, excl, 0.1, 0.2)
    assert abs(l2 - l) < 1e-9
    # 重複文字列（同一 uidx）は行列から除外される: 重複行を差し替えても、それを見る項が変わらない
    z2 = z.copy()
    z2[4] = z[1]
    e2 = e.copy()
    e2[4] = e[1]
    la, _ = SF.loss_rkd(z2, e2, excl, 0.1, 0.2)
    assert np.isfinite(la)
    # 全て重複（1 種類だけ）: 有効行なし -> 0
    one = np.zeros(z.shape[0], int)
    l3, g3 = SF.loss_rkd(z, e, one[:, None] == one[None, :], 0.1, 0.1)
    assert l3 == 0.0 and not g3.any()


@pytest.mark.parametrize("sym", [True, False])
def test_infonce_gradient_and_properties(toy, sym):
    z, t, e, uidx = toy
    l, dz = SF.loss_infonce(z, t, uidx, 0.2, sym)
    assert rel(dz, num_grad(lambda x: SF.loss_infonce(x, t, uidx, 0.2, sym)[0], z)) < 1e-6
    # 完全に整列していれば損失は小さい（z = t * 大きなスケール）、ランダムは ln(N) 付近
    N = z.shape[0]
    la, _ = SF.loss_infonce(t, t, np.arange(N), 0.05, sym)
    lr, _ = SF.loss_infonce(z, t, np.arange(N), 1e6, sym)    # tau 大 -> 一様 -> ln(N)
    assert la < lr and abs(lr - np.log(N)) < 1e-3
    # 同一文字列は負例から除外される（重複を含む行の損失が重複を除いた場合の値になる）
    z1 = z[[0, 1, 2]]
    t1 = t[[0, 1, 2]]
    u3 = np.array([0, 1, 2])
    lw, _ = SF.loss_infonce(np.vstack([z1, z1[1:2]]), np.vstack([t1, t1[1:2]]), np.array([0, 1, 2, 1]), 0.3, False)
    # 行 3 は行 1 と同一文字列。行 3 の正例は t_3(=t_1)。負例から j=1 は除外される（i=3 の行は {0,2,3} のみ）
    zz = np.vstack([z1, z1[1:2]])
    tt = np.vstack([t1, t1[1:2]])
    L = (zz / np.linalg.norm(zz, axis=1, keepdims=True)) @ (tt / np.linalg.norm(tt, axis=1, keepdims=True)).T / 0.3
    manual = 0.0
    for i in range(4):
        allowed = [j for j in range(4) if j == i or [0, 1, 2, 1][j] != [0, 1, 2, 1][i]]
        manual += -(L[i, i] - np.log(np.exp(L[i, allowed]).sum()))
    assert abs(lw - manual / 4) < 1e-12


def test_combined_loss_is_weighted_sum_and_validation(toy):
    z, t, e, uidx = toy
    spec = SF.parse_loss("infonce+rkd", "infonce=2,rkd=0.5")
    tg = {"pca": t, "rkd": e, "nce": t}
    tot, dz, parts = SF.compute_loss(spec, z, tg, uidx)
    li, gi = SF.loss_infonce(z, t, uidx, spec.tau_nce, True)
    lr, gr = SF.loss_rkd(z, e, uidx[:, None] == uidx[None, :], spec.tau_t, spec.tau_s)
    assert abs(tot - (2 * li + 0.5 * lr)) < 1e-12 and np.abs(dz - (2 * gi + 0.5 * gr)).max() < 1e-12
    assert set(parts) == {"infonce", "rkd"}
    for bad in ("foo", "cos+cos", "", "cos+bar"):
        with pytest.raises(ValueError):
            SF.parse_loss(bad)
    with pytest.raises(ValueError):
        SF.parse_loss("cos", "mse=1")
    with pytest.raises(ValueError):
        SF.parse_loss("cos", "cos=-1")
    with pytest.raises(ValueError):
        SF.parse_loss("rkd", tau_t=0.0)


# ---------------------------------------------------------------------------------------
# サンプラ
# ---------------------------------------------------------------------------------------

class FakePool:
    def __init__(self, counts, seed=0):
        src, uid = [], []
        for name, n in counts.items():
            src += [name] * n
            uid += list(range(len(uid), len(uid) + n))
        self.source = np.array(src)
        self.uid = np.array(uid)
        self.n = len(uid)
        self.len = np.ones(self.n, np.int32)
        self.tok = np.ones((self.n, 4), np.int32)

    def counts(self):
        return {k: int((self.source == k).sum()) for k in sorted(set(self.source))}


def test_sampler_equal_mix_is_stratified_deterministic_and_stateless():
    pool = FakePool({"massive": 100, "synth": 300, "jcqa": 50, "w2c": 7})
    a = SF.SemSampler(pool, 128, 0, "equal")
    assert a.quota == {"jcqa": 32, "massive": 32, "synth": 32, "w2c": 32}
    b = SF.SemSampler(pool, 128, 0, "equal")
    idx = {s: a.indices(s) for s in (0, 1, 5, 999)}
    for s, v in idx.items():
        assert np.array_equal(v, b.indices(s)) and len(v) == 128      # 状態なし・別インスタンスで同じ
    assert not np.array_equal(idx[0], idx[1]) and not np.array_equal(a.indices(0), SF.SemSampler(pool, 128, 1, "equal").indices(0))
    # 5 回呼ぶ順序に依存しない
    c = SF.SemSampler(pool, 128, 0, "equal")
    assert np.array_equal(c.indices(999), idx[999]) and np.array_equal(c.indices(0), idx[0])
    # massive は 32/step、100 文字列が 1 周するまで（3 step = 96）重複しない
    mass = [i for s in range(3) for i in a.indices(s) if pool.source[i] == "massive"]
    assert len(mass) == 96 and len(set(mass)) == 96
    # 4 step で 1 周を超える（128 > 100）が、全 massive 文字列が 1 周以内に出る
    seen = {i for s in range(4) for i in a.indices(s) if pool.source[i] == "massive"}
    assert len(seen) == 100
    # 端数配分と明示 mix
    assert SF.SemSampler(pool, 100, 0, "equal").quota == {"jcqa": 25, "massive": 25, "synth": 25, "w2c": 25}
    assert sum(SF.SemSampler(pool, 10, 0, "equal").quota.values()) == 10
    m = SF.SemSampler(pool, 100, 0, "massive:0.5,synth:0.25,jcqa:0.25")
    assert m.quota == {"massive": 50, "synth": 25, "jcqa": 25}
    assert all(pool.source[i] in ("massive", "synth", "jcqa") for i in m.indices(3))
    u = SF.SemSampler(pool, 64, 0, "uniform")
    v = u.indices(2)
    assert len(set(v.tolist())) == 64 and np.array_equal(v, SF.SemSampler(pool, 64, 0, "uniform").indices(2))
    with pytest.raises(ValueError):
        SF.SemSampler(pool, 64, 0, "nosuch:1")


# ---------------------------------------------------------------------------------------
# Student（np float64）: 独立 pass の数値勾配・加法性・従来経路
# ---------------------------------------------------------------------------------------

def _tiny(seed=0, n=24, vocab=40, lc=5, kmax=3, lp=7):
    rng = np.random.default_rng(seed)
    d = dict(item_id=np.arange(n, dtype=np.int64), prefix=np.zeros((n, lp), np.int32), prefix_len=np.zeros(n, np.int32),
             cand=np.zeros((n, kmax, lc), np.int32), cand_len=np.zeros((n, kmax), np.int32), k=np.zeros(n, np.int32),
             t_logits=np.zeros((n, kmax), np.float32), gold=np.full(n, -1, np.int32))
    for i in range(n):
        pl = int(rng.integers(2, lp + 1))
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
    d["source"] = np.where(np.arange(n) % 2 == 0, "synth", "massive").astype("U8")
    return M.Shard(d)


def _tiny_sem_dir(sh, tmp_path, d=6, D=12, seed=3, invalid_frac=0.15):
    rng = np.random.default_rng(seed)
    U = 40
    ci = rng.integers(0, U, size=(sh.n, sh.kmax)).astype(np.int32)
    ci[rng.random(ci.shape) < invalid_frac] = -1
    ci[np.arange(sh.kmax)[None, :] >= sh.k[:, None]] = -1
    raw = rng.normal(size=(U, D)).astype(np.float32)
    pca = (raw[:, :d] * 0.3).astype(np.float32)
    out = tmp_path / "semdir"
    os.makedirs(out, exist_ok=True)
    np.save(out / "emb_raw.npy", raw)
    np.save(out / "emb_pca.npy", pca)
    np.save(out / "cand_idx_train.npy", ci)
    json.dump({"provider": "toy", "shard": "tiny", "item_id_sha1": M.item_id_sha1(sh.item_id), "exclude_truncated": {"applied": True}},
              open(out / "meta.json", "w"))
    return str(out)


def _mk(tmp_path, loss="infonce+rkd", weights=None, M_=10, mix="equal", extra=None, d=6):
    import argparse
    sh = _tiny()
    sem_dir = _tiny_sem_dir(sh, tmp_path, d=d)
    sem, info = M.load_sem_dir(sem_dir, sh.item_id)
    sem.check_against(sh)
    ap = argparse.ArgumentParser()
    SF.add_args(ap)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sem-emb", default=sem_dir)
    a = ap.parse_args(["--sem-batch", str(M_), "--sem-loss", loss, "--sem-batch-mix", mix, *(["--sem-loss-weights", weights] if weights else []),
                       *(extra or [])])
    hook = SF.build_hook(a, sh, sem, 0.7)
    cfg = M.Config(vocab=40, emb=5, hidden=6, layers=2, lp=7, lc=5, sem_dim=d)
    be = NPBackend(np.float64)
    st = M.Student(be, cfg, 6, 3, train=True, max_lp=7, max_lc=5, max_sem=M_)
    params = {k: (v * 2.0).astype(np.float64) for k, v in M.init_params(cfg, 1).items()}
    st.set_params(params)
    st.sem_lambda = 0.7
    st.sem_ext = hook
    return sh, sem, hook, cfg, be, st, params


def _num_param_grad(st, params, name, flat_idx, f, eps=1e-6):
    base = {k: v.copy() for k, v in params.items()}
    out = []
    for fi in flat_idx:
        vals = []
        for sgn in (+1, -1):
            p = {k: v.copy() for k, v in base.items()}
            p[name].reshape(-1)[fi] += sgn * eps
            st.set_params(p)
            vals.append(f())
        out.append((vals[0] - vals[1]) / (2 * eps))
    st.set_params(base)
    return np.array(out)


@pytest.mark.parametrize("loss", ["cos", "mse", "rkd", "infonce", "infonce+rkd"])
def test_independent_pass_gradient_matches_numeric(tmp_path, loss):
    sh, sem, hook, cfg, be, st, params = _mk(tmp_path, loss=loss, M_=9)
    hook.step = 4
    be.zero(st.G)
    st.sem_ext.run(st, True)
    g = st.get_grads()
    # 損失（λ 倍）の数値微分: run(want_grads=False) の loss * λ
    f = lambda: 0.7 * hook.run(st, False)
    rng = np.random.default_rng(1)
    worst = 0.0
    used_tokens = np.unique(hook.pool.tok[hook.sampler.indices(4)])
    for name, shape in M.param_specs(cfg):
        if name.startswith("head."):
            assert not g[name].any(), name        # 判断 head は触らない
            continue
        n = int(np.prod(shape))
        idx = rng.choice(n, size=min(n, 8), replace=False)
        if name == "emb":
            idx = np.concatenate([u * cfg.emb + rng.integers(0, cfg.emb, 2) for u in used_tokens[:4]])
        num = _num_param_grad(st, params, name, idx, f)
        ana = g[name].reshape(-1)[idx]
        err = np.abs(num - ana).max() / max(1e-8, np.abs(num).max(), np.abs(ana).max())
        worst = max(worst, err)
        assert err < 1e-5, (loss, name, err, num, ana)
    # 独立 pass は判断経路の param（prefix 側の embedding 以外）に依存しない: sem.* と GRU と emb に勾配が出る
    assert np.abs(g["sem.wp"]).max() > 0 and np.abs(g["l0.wi"]).max() > 0 and np.abs(g["emb"]).max() > 0


def test_judge_plus_independent_is_additive_and_default_path_unchanged(tmp_path):
    sh, sem, hook, cfg, be, st, params = _mk(tmp_path, loss="infonce+rkd", M_=8)
    bt = M.make_batch(sh, np.arange(6), keys=np.random.default_rng(3).random((6, 3)), sem=sem)
    hook.step = 2
    r = st.loss_grads(bt)
    g_both = {k: v.copy() for k, v in st.get_grads().items()}
    # 独立 pass 無し（λ=0）= 判断だけ
    st.sem_lambda = 0.0
    r0 = st.loss_grads(bt)
    g_j = {k: v.copy() for k, v in st.get_grads().items()}
    st.sem_lambda = 0.7
    be.zero(st.G)
    hook.step = 2
    hook.run(st, True)
    g_s = st.get_grads()
    for k in g_both:
        assert np.abs(g_both[k] - (g_j[k] + g_s[k])).max() < 1e-12, k
    assert abs(r["loss"] - (r0["loss"] + 0.7 * r["sem"])) < 1e-12 and r["sem"] > 0 and r0["sem"] == 0.0
    # sem_ext 無し（従来の判断バッチ内 cos）は max_sem 追加前後で同じ値（独立 pass は呼ばれない）
    st.sem_ext = None
    r_old = st.loss_grads(bt)
    st2 = M.Student(NPBackend(np.float64), cfg, 6, 3, train=True, max_lp=7, max_lc=5)     # max_sem 既定 0
    st2.set_params(params)
    st2.sem_lambda = 0.7
    r_ref = st2.loss_grads(bt)
    assert r_old == r_ref
    assert all(np.array_equal(st.get_grads()[k], st2.get_grads()[k]) for k in params)


def test_buffers_grow_only_when_needed():
    cfg = M.Config(vocab=40, emb=5, hidden=6, layers=2, lp=7, lc=5, sem_dim=6)
    a = M.Student(NPBackend(np.float64), cfg, 4, 3, train=True, max_lp=7, max_lc=5)
    b = M.Student(NPBackend(np.float64), cfg, 4, 3, train=True, max_lp=7, max_lc=5, max_sem=12)
    c = M.Student(NPBackend(np.float64), cfg, 4, 3, train=True, max_lp=7, max_lc=5, max_sem=64)
    assert a.Nc == b.Nc == a.Nm == 12 and c.Nc == 64
    assert a.hs_c[0].shape == b.hs_c[0].shape and c.hs_c[0].size > a.hs_c[0].size


def test_pool_and_targets(tmp_path):
    sh, sem, hook, cfg, be, st, params = _mk(tmp_path, loss="infonce+rkd", M_=8, extra=["--sem-nce-target", "rproj"])
    pool = hook.pool
    # プールは有効位置の unique 文字列だけ。tok/len/uid が shard の初出位置と一致
    ci = sem.cand_idx
    assert set(pool.uid.tolist()) == set(ci[ci >= 0].tolist())
    for p in range(pool.n):
        r, j = [(r, j) for r, j in zip(*np.nonzero(ci == pool.uid[p]))][0]
        assert np.array_equal(pool.tok[p], sh.cand[r, j]) and pool.len[p] == sh.cand_len[r, j] and pool.source[p] == sh.extra["source"][r]
    tg = hook.targets
    assert tg.rkd.shape == (pool.n, 12) and tg.nce.shape == (pool.n, 6) and tg.pca.shape == (pool.n, 6)
    assert np.allclose(np.linalg.norm(tg.rkd, axis=1), 1, atol=1e-5)          # 中心化後に L2 正規化
    assert np.abs(tg.rkd.astype(np.float64).mean(0)).max() < 0.5              # 概ね中心化されている
    assert np.allclose(np.linalg.norm(tg.nce, axis=1), 1, atol=1e-5)


def test_build_hook_validation(tmp_path):
    import argparse
    sh = _tiny()
    sem_dir = _tiny_sem_dir(sh, tmp_path)
    sem, _ = M.load_sem_dir(sem_dir, sh.item_id)

    def mk(argv, sem_=sem, w=0.5):
        ap = argparse.ArgumentParser()
        SF.add_args(ap)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--sem-emb", default=sem_dir)
        return SF.build_hook(ap.parse_args(argv), sh, sem_, w)

    assert mk([]) is None and mk(["--sem-loss", "cos"]) is None                     # 既定は従来経路
    with pytest.raises(ValueError, match="sem-batch"):
        mk(["--sem-loss", "infonce"])
    with pytest.raises(ValueError, match="sem-batch"):
        mk(["--sem-loss", "mse"])
    with pytest.raises(ValueError, match="weights"):
        mk(["--sem-loss-weights", "cos=2"])
    with pytest.raises(ValueError, match="sem-cand-weight"):
        mk(["--sem-batch", "8"], sem_=None, w=0.0)
    with pytest.raises(ValueError, match="sem-batch"):
        mk(["--sem-batch", "1"])
    assert mk(["--sem-batch", "8"]) is not None


# ---------------------------------------------------------------------------------------
# train.py / sem_only.py 結線（np backend, fake shard + source 付き train）
# ---------------------------------------------------------------------------------------

CFG = '{"vocab": 300, "emb": 24, "hidden": 32, "layers": 2}'


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_semfix")
    F.write_fake_dataset(str(d), n_train=400, n_val=60, n_test=60, n_robust=10, vocab=300, lp=48, lc=8, kmax=5, seed=1, active=30)
    z = np.load(d / "train_L48.npz", allow_pickle=False)
    a = {k: z[k] for k in z.files}
    a["source"] = np.where(np.arange(len(a["item_id"])) % 3 == 0, "massive", "synth").astype("U8")
    np.savez(d / "train_L48.npz", **a)
    return str(d)


def write_sem_dir(fake, out, d=8, D=16, seed=0):
    sh = M.Shard(M.find_shard(fake, "train", 48))
    rng = np.random.default_rng(seed)
    tokvec = rng.normal(size=(300, D))
    index, vecs = {}, []
    ci = np.full((sh.n, sh.kmax), -1, np.int32)
    for i in range(sh.n):
        for j in range(sh.k[i]):
            key = tuple(sh.cand[i, j, :sh.cand_len[i, j]].tolist())
            if key not in index:
                index[key] = len(vecs)
                vecs.append(tokvec[list(key)].mean(0))
            ci[i, j] = index[key]
    os.makedirs(out, exist_ok=True)
    raw = np.array(vecs, np.float32)
    np.save(os.path.join(out, "emb_raw.npy"), raw)
    np.save(os.path.join(out, "emb_pca.npy"), raw[:, :d].copy())
    np.save(os.path.join(out, "cand_idx_train.npy"), ci)
    json.dump({"provider": "toy", "shard": "fake", "item_id_sha1": M.item_id_sha1(sh.item_id), "explained_variance_ratio": 1.0},
              open(os.path.join(out, "meta.json"), "w"))
    return str(out), raw.shape[0]


def run_train(fake, run, extra=()):
    argv = ["--backend", "np", "--data", fake, "--run-dir", str(run), "--config", CFG, "--batch-size", "32", "--seed", "3",
            "--log-every", "10", "--eval-every", "30", "--lr", "3e-3", "--lp", "48", "--epochs", "4", *extra]
    return T.main(argv)


def read_csv(run):
    rows = [l.split(",") for l in open(run / "metrics.csv").read().strip().splitlines()]
    return rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]


def test_train_with_independent_batch_reduces_loss_and_records(fake, tmp_path):
    sem_dir, U = write_sem_dir(fake, tmp_path / "sem")
    run = tmp_path / "run"
    run_train(fake, run, ["--epochs", "6", "--sem-cand-weight", "0.5", "--sem-emb", sem_dir, "--sem-batch", "24",
                          "--sem-loss", "infonce+rkd", "--sem-tau-nce", "0.2", "--no-final-eval"])
    _, rows = read_csv(run)
    sl = [float(r["sem_loss"]) for r in rows if r["sem_loss"]]
    assert len(sl) >= 5 and sl[-1] < 0.8 * sl[0], sl
    cfg = json.load(open(run / "config.json"))
    sf = cfg["sem"]["semfix"]
    assert sf["loss"]["terms"] == {"infonce": 1.0, "rkd": 1.0} and sf["sampler"]["M"] == 24 and sf["sampler"]["mix"] == "equal"
    assert set(sf["sampler"]["pool_by_source"]) == {"massive", "synth"} and sf["sampler"]["quota"] == {"massive": 12, "synth": 12}
    assert "独立ミニバッチ M=24 infonce+rkd" in open(run / "train.log").read()


def test_train_independent_batch_resume_is_exact_and_default_unchanged(fake, tmp_path):
    sem_dir, _ = write_sem_dir(fake, tmp_path / "sem")
    ex = ["--no-final-eval", "--sem-cand-weight", "0.5", "--sem-emb", sem_dir, "--sem-batch", "16", "--sem-loss", "mse"]
    a, b = tmp_path / "a", tmp_path / "b"
    run_train(fake, a, ["--max-steps", "30", *ex])
    run_train(fake, b, ["--max-steps", "12", *ex])
    run_train(fake, b, ["--max-steps", "30", "--resume", *ex])
    _, pa, ma, va, _ = T.load_ckpt(str(a / "ckpt" / "last.npz"))
    _, pb, mb, vb, _ = T.load_ckpt(str(b / "ckpt" / "last.npz"))
    assert all(np.array_equal(pa[k], pb[k]) for k in pa) and np.array_equal(ma, mb) and np.array_equal(va, vb)
    # 新フラグを付けない run は、独立ミニバッチ導入前と同じ従来経路（sem-batch=0 で --sem-loss cos 明示でもビット一致）
    c, d = tmp_path / "c", tmp_path / "d"
    ex0 = ["--no-final-eval", "--sem-cand-weight", "0.5", "--sem-emb", sem_dir, "--max-steps", "20"]
    run_train(fake, c, ex0)
    run_train(fake, d, [*ex0, "--sem-batch", "0", "--sem-loss", "cos"])
    _, pc, mc, vc, _ = T.load_ckpt(str(c / "ckpt" / "last.npz"))
    _, pd, md, vd, _ = T.load_ckpt(str(d / "ckpt" / "last.npz"))
    assert all(np.array_equal(pc[k], pd[k]) for k in pc) and np.array_equal(mc, md)


def test_train_semfix_arg_validation(fake, tmp_path):
    sem_dir, _ = write_sem_dir(fake, tmp_path / "sem")
    with pytest.raises(SystemExit, match="sem-batch"):
        run_train(fake, tmp_path / "x1", ["--max-steps", "2", "--sem-cand-weight", "0.5", "--sem-emb", sem_dir, "--sem-loss", "infonce"])
    with pytest.raises(SystemExit, match="sem-cand-weight"):
        run_train(fake, tmp_path / "x2", ["--max-steps", "2", "--sem-batch", "16"])


def run_sem_only(fake, run, sem_dir, extra=()):
    argv = ["--backend", "np", "--data", fake, "--lp", "48", "--run-dir", str(run), "--config", CFG, "--seed", "3", "--lr", "3e-3",
            "--sem-emb", sem_dir, "--log-every", "10", "--ckpt-every", "10", *extra]
    return SO.main(argv)


def test_sem_only_trains_resumes_and_does_not_touch_judge_head(fake, tmp_path):
    sem_dir, _ = write_sem_dir(fake, tmp_path / "sem")
    a, b = tmp_path / "a", tmp_path / "b"
    s = run_sem_only(fake, a, sem_dir, ["--max-steps", "60", "--sem-batch", "20", "--sem-loss", "infonce", "--sem-tau-nce", "0.2"])
    assert s["steps_done"] == 60 and not s["interrupted"]
    for name in ("p000", "p010", "p025", "p050", "p075", "p100", "last", "best"):
        assert (a / "ckpt" / f"{name}.npz").exists(), name
    rows = [l.split(",") for l in open(a / "metrics.csv").read().strip().splitlines()]
    sl = [float(r[1]) for r in rows[1:]]
    assert sl[-1] < sl[0]
    c0, p0, *_ = T.load_ckpt(str(a / "ckpt" / "p000.npz"))
    c1, p1, *_ = T.load_ckpt(str(a / "ckpt" / "best.npz"))
    assert c1.sem_dim == 8 and not np.array_equal(p0["sem.wp"], p1["sem.wp"]) and not np.array_equal(p0["l0.wi"], p1["l0.wi"])
    # 判断 head は勾配を受けない（AdamW の weight decay による縮小だけ: (1 - lr*wd)^steps）
    dec = (1 - 3e-3 * 0.01) ** 60
    assert np.allclose(p1["head.w1"], p0["head.w1"] * dec, rtol=1e-4, atol=1e-7) and np.allclose(p1["head.w2"], p0["head.w2"] * dec, rtol=1e-4, atol=1e-7)
    # resume が厳密
    run_sem_only(fake, b, sem_dir, ["--max-steps", "25", "--sem-batch", "20", "--sem-loss", "infonce", "--sem-tau-nce", "0.2"])
    run_sem_only(fake, b, sem_dir, ["--max-steps", "60", "--sem-batch", "20", "--sem-loss", "infonce", "--sem-tau-nce", "0.2", "--resume"])
    _, pa, ma, va, ta = T.load_ckpt(str(a / "ckpt" / "last.npz"))
    _, pb, mb, vb, tb = T.load_ckpt(str(b / "ckpt" / "last.npz"))
    assert ta["step"] == tb["step"] == 60
    assert all(np.array_equal(pa[k], pb[k]) for k in pa) and np.array_equal(ma, mb) and np.array_equal(va, vb)
    # 既存 checkpoint のある run-dir は --resume 無しでは拒否、emb_eval 配下は拒否
    with pytest.raises(SystemExit, match="既存 checkpoint"):
        run_sem_only(fake, a, sem_dir, ["--max-steps", "5"])
    ev = tmp_path / "emb_eval" / "p" / "x"
    ev.mkdir(parents=True)
    for f in os.listdir(sem_dir):
        os.link(os.path.join(sem_dir, f), ev / f)
    with pytest.raises(SystemExit, match="emb_eval"):
        run_sem_only(fake, tmp_path / "e", str(ev), ["--max-steps", "5"])


# ---------------------------------------------------------------------------------------
# SIGTERM で last checkpoint を保存して終了
# ---------------------------------------------------------------------------------------

def _wait_for(path, text, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path) and text in open(path).read():
            return True
        time.sleep(0.2)
    return False


@pytest.mark.parametrize("sem_only", [False, True])
def test_sigterm_saves_last_checkpoint_and_resumes(fake, tmp_path, sem_only):
    sem_dir, _ = write_sem_dir(fake, tmp_path / "sem")
    run = tmp_path / "run"
    if sem_only:
        cmd = [sys.executable, "-m", "tb250distill.student.sem_only", "--backend", "np", "--data", fake, "--lp", "48", "--run-dir", str(run),
               "--config", CFG, "--sem-emb", sem_dir, "--max-steps", "100000", "--log-every", "5", "--ckpt-every", "0", "--sem-batch", "64"]
        marker = "step 10/"
    else:
        cmd = [sys.executable, "-m", "tb250distill.student.train", "--backend", "np", "--data", fake, "--lp", "48", "--run-dir", str(run),
               "--config", CFG, "--epochs", "400", "--ckpt-every", "0", "--log-every", "5", "--eval-every", "100000", "--no-final-eval"]
        marker = "step 10/"
    # `&` 起動のバックグラウンド run と同じく SIGINT を無視した状態で起動する（preexec で SIG_IGN）
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    try:
        assert _wait_for(str(run / "train.log"), marker), "学習が始まらない"
        # ckpt-every 0 なので last は step 0 のものしか無い（SIGTERM 前）
        c, _, _, _, meta0 = T.load_ckpt(str(run / "ckpt" / "last.npz"))
        assert meta0["step"] == 0
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert rc == 0, rc
    log = open(run / "train.log").read()
    assert "interrupted at step" in log and "saving last checkpoint" in log
    _, _, _, _, meta = T.load_ckpt(str(run / "ckpt" / "last.npz"))
    assert meta["step"] >= 10
    if sem_only:
        assert json.load(open(run / "summary.json"))["interrupted"] is True
        assert not (run / "ckpt" / "best.npz").exists()      # 完走していないので best は作らない


def test_install_sigterm_handler_is_restored_and_raises_keyboardinterrupt():
    prev = signal.getsignal(signal.SIGTERM)
    restore = T.install_sigterm_as_interrupt()
    try:
        assert signal.getsignal(signal.SIGTERM) is not prev
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1.0)
    finally:
        restore()
    assert signal.getsignal(signal.SIGTERM) is prev


# ---------------------------------------------------------------------------------------
# np / cl 一致（OpenCL デバイスがあるときだけ）
# ---------------------------------------------------------------------------------------

DEVICE = os.environ.get("TB250_CL_DEVICE", "GT 730")


@pytest.fixture(scope="module")
def cl_be():
    pytest.importorskip("pyopencl")
    pytest.importorskip("pyclblast")
    from tb250distill.student.backend_cl import CLBackend
    try:
        return CLBackend(DEVICE)
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("loss,mix", [("cos", "equal"), ("infonce+rkd", "equal"), ("mse", "uniform")])
def test_cl_independent_pass_matches_numpy(cl_be, tmp_path, loss, mix):
    sh, sem, hook_np, cfg, be, st_np, params = _mk(tmp_path, loss=loss, M_=11, mix=mix)
    import copy
    hook_cl = copy.deepcopy(hook_np)
    st_cl = M.Student(cl_be, cfg, 6, 3, train=True, max_lp=7, max_lc=5, max_sem=11)
    st_cl.set_params({k: v.astype(np.float32) for k, v in params.items()})
    st_cl.sem_lambda = 0.7
    st_cl.sem_ext = hook_cl
    bt = M.make_batch(sh, np.arange(6), keys=np.random.default_rng(3).random((6, 3)), sem=sem)
    hook_np.step = hook_cl.step = 5
    r_np, r_cl = st_np.loss_grads(bt), st_cl.loss_grads(bt)
    assert abs(r_np["sem"] - r_cl["sem"]) < 2e-4 and abs(r_np["loss"] - r_cl["loss"]) < 2e-4
    g_np, g_cl = st_np.get_grads(), st_cl.get_grads()
    cat = lambda g: np.concatenate([g[n].ravel() for n, _ in M.param_specs(cfg)])
    assert rel(cat(g_cl), cat(g_np)) < 2e-3
    for n, _ in M.param_specs(cfg):
        floor = 1e-4 * np.linalg.norm(cat(g_np))
        assert np.linalg.norm(g_cl[n] - g_np[n]) / max(floor, np.linalg.norm(g_np[n])) < 5e-3, n
    # sem-only の 1 step 後 weight も一致
    st_np.set_params(params)
    st_cl.set_params({k: v.astype(np.float32) for k, v in params.items()})
    st_np.set_opt(np.zeros(st_np.n_params), np.zeros(st_np.n_params))
    st_cl.set_opt(np.zeros(st_cl.n_params), np.zeros(st_cl.n_params))
    a = SF.sem_only_step(st_np, 2e-3, 1)
    b = SF.sem_only_step(st_cl, 2e-3, 1)
    assert abs(a["sem"] - b["sem"]) < 2e-4
    flat = lambda s: np.concatenate([x.ravel() for x in (s.get_params()[n] for n, _ in M.param_specs(cfg))])
    gflat = cat(st_np.get_grads())        # sem-only step が作った（判断なしの）勾配
    big = np.abs(gflat) > 1e-5
    assert big.sum() > 50 and np.abs((flat(st_np) - flat(st_cl))[big]).max() < 2e-5


def test_items_sampler_matches_judge_batch_candidates(tmp_path):
    """items:B = 判断バッチ（train.py と同じ epoch_order）の候補を全て（重複あり）使う従来相当。sem-only 専用。"""
    import argparse
    sh = _tiny()
    sem_dir = _tiny_sem_dir(sh, tmp_path)
    sem, _ = M.load_sem_dir(sem_dir, sh.item_id)
    ap = argparse.ArgumentParser()
    SF.add_args(ap)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sem-emb", default=sem_dir)
    a = ap.parse_args(["--sem-batch", "8", "--sem-batch-mix", "items:6"])
    with pytest.raises(ValueError, match="sem_only"):
        SF.build_hook(a, sh, sem, 1.0)
    hook = SF.build_hook(a, sh, sem, 1.0, allow_items=True)
    assert hook.sampler.max_n == 6 * sh.kmax
    for step in (0, 3, 4, 9):
        ep, pos = divmod(step, -(-sh.n // 6))
        order, _ = T.epoch_order(0, ep, sh.n, sh.kmax)
        rows = order[pos * 6:(pos + 1) * 6]
        want = sem.cand_idx[rows].reshape(-1)
        want = want[want >= 0]
        got = hook.pool.uid[hook.sampler.indices(step)]
        assert np.array_equal(got, want)
