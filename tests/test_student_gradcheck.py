"""numpy float64 での数値勾配チェック（全パラメータ種類・長さマスク・候補マスク込み）と構造テスト。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill.student.backend_np import NPBackend  # noqa: E402
from tb250distill.student import model as M  # noqa: E402


def tiny_shard(seed=0, n=6, vocab=40, lp=9, lc=5, kmax=4):
    rng = np.random.default_rng(seed)
    d = dict(item_id=np.arange(n, dtype=np.int64), prefix=np.zeros((n, lp), np.int32),
             prefix_len=np.zeros(n, np.int32), cand=np.zeros((n, kmax, lc), np.int32),
             cand_len=np.zeros((n, kmax), np.int32), k=np.zeros(n, np.int32),
             t_logits=np.zeros((n, kmax), np.float32), gold=np.full(n, -1, np.int32))
    for i in range(n):
        pl = [lp, 1, 4, lp - 2, 3, lp][i % 6]
        d["prefix_len"][i] = pl
        d["prefix"][i, :pl] = rng.integers(5, vocab, pl)
        k = [kmax, 2, 3, kmax, 2, 3][i % 6]
        d["k"][i] = k
        for j in range(k):
            cl = int(rng.integers(1, lc + 1))
            d["cand_len"][i, j] = cl
            d["cand"][i, j, :cl] = rng.integers(5, vocab, cl)
        d["t_logits"][i, :k] = rng.normal(size=k) * 2
        d["gold"][i] = [0, -1, 1, 2, -1, 0][i % 6] % k if i % 3 != 1 else -1
    return M.Shard(d)


def build(layers, seed=1, scale=2.0, B=6, kmax=4):
    cfg = M.Config(vocab=40, emb=5, hidden=6, layers=layers, lp=9, lc=5)
    be = NPBackend(np.float64)
    st = M.Student(be, cfg, B, kmax, train=True)
    params = M.init_params(cfg, seed)
    params = {k: (v * scale).astype(np.float64) for k, v in params.items()}
    st.set_params(params)
    return cfg, be, st, params


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_numeric_gradient(layers):
    cfg, be, st, params = build(layers)
    sh = tiny_shard()
    bt = M.make_batch(sh, np.arange(6), keys=np.random.default_rng(3).random((6, 4)))
    st.loss_grads(bt)
    grads = st.get_grads()
    rng = np.random.default_rng(7)
    eps = 1e-6
    worst = 0.0
    for name, shape in M.param_specs(cfg):
        g = grads[name]
        flat_idx = np.arange(int(np.prod(shape)))
        if name == "emb":
            # 使われている token + 使われていない token（勾配 0 のはず）
            used = np.unique(np.concatenate([sh.prefix.reshape(-1), sh.cand.reshape(-1)]))
            rows = np.concatenate([rng.choice(used, 6, replace=False), [39]])
            flat_idx = np.concatenate([r * shape[1] + np.arange(shape[1]) for r in rows])
        elif len(flat_idx) > 40:
            flat_idx = rng.choice(flat_idx, 40, replace=False)
        for fi in flat_idx:
            p = {k: v.copy() for k, v in params.items()}
            p[name].reshape(-1)[fi] += eps
            st.set_params(p)
            lp_ = st.loss_grads(bt, want_grads=False)["loss"]
            p[name].reshape(-1)[fi] -= 2 * eps
            st.set_params(p)
            lm_ = st.loss_grads(bt, want_grads=False)["loss"]
            num = (lp_ - lm_) / (2 * eps)
            ana = g.reshape(-1)[fi]
            # パラメータ種類ごとの最大 |grad| で正規化（微小勾配の桁落ちに引きずられない）
            # （head.b2 は softmax のシフト不変性により解析的に勾配 0 -> 正規化の床 1e-3 を置く）
            err = abs(num - ana) / max(1e-3, np.abs(g).max())
            worst = max(worst, err)
            assert err < 1e-6, (name, fi, num, ana)
    st.set_params(params)
    print(f"layers={layers} worst normalized abs error {worst:.2e}")
    assert worst < 1e-6


def test_masks_and_equivariance():
    cfg, be, st, params = build(2)
    sh = tiny_shard()
    idx = np.arange(6)
    bt = M.make_batch(sh, idx)
    s0 = st.predict(bt)
    # 1) prefix/候補の pad 位置の token を書き換えても不変
    d = {k: getattr(sh, k).copy() for k in M.SHARD_KEYS}
    for i in range(6):
        d["prefix"][i, d["prefix_len"][i]:] = 17
        for j in range(4):
            d["cand"][i, j, d["cand_len"][i, j]:] = 23
    s1 = st.predict(M.make_batch(M.Shard(d), idx))
    valid = np.arange(4)[None, :] < sh.k[:, None]
    assert np.allclose(s0[valid], s1[valid], atol=1e-12)
    # 2) 候補 permutation 等変: 並べ替えて推論 -> 元に戻すと一致（teacher logits・gold も追従）
    keys = np.random.default_rng(5).random((6, 4))
    btp = M.make_batch(sh, idx, keys=keys)
    sp = st.predict(btp)
    for b in range(6):
        k = sh.k[b]
        assert np.allclose(sp[b, :k], s0[b, btp.perm[b, :k]], atol=1e-12)
        g0 = sh.gold[b]
        if g0 >= 0:
            assert btp.perm[b, btp.gold[b]] == g0
        assert np.allclose(btp.tlog.reshape(6, -1)[b, :k], sh.t_logits[b, btp.perm[b, :k]])
    # 3) バッチ構成に依存しない（1 件ずつ推論 == まとめて推論）
    for b in range(6):
        one = st.predict(M.make_batch(sh, [b]))
        k = sh.k[b]
        assert np.allclose(one[0, :k], s0[b, :k], atol=1e-12)


def test_param_count_and_init_determinism():
    c = M.PRESETS["common_s"]
    n = M.param_count(c)
    # 手計算: emb 8192*128 + 2層 GRU + head
    emb = 8192 * 128
    l0 = 128 * 576 + 192 * 576 + 2 * 576
    l1 = 192 * 576 + 192 * 576 + 2 * 576
    head = 192 * 192 + 192 + 192 + 1
    assert n == emb + l0 + l1 + head
    p1 = M.init_params(M.Config(vocab=50, emb=4, hidden=5, layers=2), 3)
    p2 = M.init_params(M.Config(vocab=50, emb=4, hidden=5, layers=2), 3)
    p3 = M.init_params(M.Config(vocab=50, emb=4, hidden=5, layers=2), 4)
    assert all(np.array_equal(p1[k], p2[k]) for k in p1)
    assert any(not np.array_equal(p1[k], p3[k]) for k in p1)
    assert sum(v.size for v in p1.values()) == M.param_count(M.Config(vocab=50, emb=4, hidden=5, layers=2))


def test_matches_pytorch_gru_formula():
    """GRU セルが PyTorch の式（n = tanh(W_in x + b_in + r*(W_hn h + b_hn))）と一致することを独立実装で確認。"""
    cfg, be, st, params = build(1, B=1)
    sh = tiny_shard()
    bt = M.make_batch(sh, [0])
    s = st.predict(bt)[0]
    H = cfg.hidden
    emb = params["emb"]
    wi, wh, bi, bh = (params[f"l0.{n}"] for n in ("wi", "wh", "bi", "bh"))

    def run(tokens, h):
        for tok in tokens:
            x = emb[tok]
            gi = x @ wi + bi
            gh = h @ wh + bh
            r = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
            z = 1 / (1 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h = (1 - z) * n + z * h
        return h

    hp = run(sh.prefix[0, :sh.prefix_len[0]], np.zeros(H))
    for j in range(sh.k[0]):
        h = run(sh.cand[0, j, :sh.cand_len[0, j]], hp.copy())
        zz = np.tanh(h @ params["head.w1"] + params["head.b1"])
        ref = (zz @ params["head.w2"] + params["head.b2"])[0]
        assert abs(ref - s[j]) < 1e-12


def test_no_stale_state_between_batches_of_different_size():
    """B の異なるバッチを続けて処理しても（buffer の使い回しで）結果が変わらない（h0 block の残骸バグの回帰）。"""
    for train in (True, False):
        cfg = M.Config(vocab=40, emb=5, hidden=6, layers=2, lp=9, lc=5)
        be = NPBackend(np.float64)
        st = M.Student(be, cfg, 6, 4, train=train)
        st.set_params({k: v.astype(np.float64) * 2 for k, v in M.init_params(cfg, 1).items()})
        sh = tiny_shard()
        full = M.make_batch(sh, np.arange(6))
        s_ref = st.predict(full)
        st.predict(M.make_batch(sh, [1, 2]))          # 小さいバッチで buffer を汚す
        st.predict(M.make_batch(sh, [3]))
        assert np.array_equal(st.predict(full), s_ref)
        if train:
            r1 = st.loss_grads(full)
            g1 = st.get_grads()
            st.loss_grads(M.make_batch(sh, [4, 5]))
            r2 = st.loss_grads(full)
            g2 = st.get_grads()
            assert r1 == r2 and all(np.array_equal(g1[k], g2[k]) for k in g1)


# ----------------------------------------------------------------------------------------------
# Candidate Semantic Distillation（候補を文脈なし h0=0 で GRU に通し projection head -> cosine loss）
# ----------------------------------------------------------------------------------------------

def tiny_sem(sh, d=3, U=9, seed=11, invalid_frac=0.3):
    """候補位置ごとの unique 文字列 index（一部 -1 = embedding 無し）と PCA 済み embedding [U, d]。"""
    rng = np.random.default_rng(seed)
    ci = rng.integers(0, U, size=(sh.n, sh.kmax)).astype(np.int32)
    ci[rng.random(ci.shape) < invalid_frac] = -1
    ci[np.arange(sh.kmax)[None, :] >= sh.k[:, None]] = -1   # 候補無し（pad）位置
    emb = rng.normal(size=(U, d)).astype(np.float32)
    return M.SemTargets(ci, emb)


def build_sem(layers, lam, d=3, seed=1, scale=2.0, B=6, kmax=4):
    cfg = M.Config(vocab=40, emb=5, hidden=6, layers=layers, lp=9, lc=5, sem_dim=d)
    be = NPBackend(np.float64)
    st = M.Student(be, cfg, B, kmax, train=True)
    st.sem_lambda = lam
    params = M.init_params(cfg, seed)
    params = {k: (v * scale).astype(np.float64) for k, v in params.items()}
    st.set_params(params)
    return cfg, be, st, params


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_numeric_gradient_with_sem_loss(layers):
    """sem loss 込みの全パラメータ（projection head 含む）の数値勾配。無効候補(-1)・pad 候補・長さの異なる候補を含む。"""
    lam = 0.7
    cfg, be, st, params = build_sem(layers, lam)
    sh = tiny_shard()
    sem = tiny_sem(sh)
    bt = M.make_batch(sh, np.arange(6), keys=np.random.default_rng(3).random((6, 4)), sem=sem)
    assert 0 < bt.sem_valid.sum() < bt.B * bt.K and (~bt.sem_valid).any()
    r = st.loss_grads(bt)
    assert r["sem"] > 0 and abs(r["loss"] - (st.loss_grads(bt, want_grads=False)["loss"])) < 1e-12
    grads = st.get_grads()
    assert np.abs(grads["sem.wp"]).max() > 0 and np.abs(grads["sem.bp"]).max() > 0
    rng = np.random.default_rng(7)
    eps = 1e-6
    worst = 0.0
    for name, shape in M.param_specs(cfg):
        g = grads[name]
        flat_idx = np.arange(int(np.prod(shape)))
        if name == "emb":
            used = np.unique(np.concatenate([sh.prefix.reshape(-1), sh.cand.reshape(-1)]))
            rows = np.concatenate([rng.choice(used, 6, replace=False), [39]])
            flat_idx = np.concatenate([r_ * shape[1] + np.arange(shape[1]) for r_ in rows])
        elif len(flat_idx) > 40:
            flat_idx = rng.choice(flat_idx, 40, replace=False)
        for fi in flat_idx:
            p = {k: v.copy() for k, v in params.items()}
            p[name].reshape(-1)[fi] += eps
            st.set_params(p)
            lp_ = st.loss_grads(bt, want_grads=False)["loss"]
            p[name].reshape(-1)[fi] -= 2 * eps
            st.set_params(p)
            lm_ = st.loss_grads(bt, want_grads=False)["loss"]
            num = (lp_ - lm_) / (2 * eps)
            err = abs(num - g.reshape(-1)[fi]) / max(1e-3, np.abs(g).max())
            worst = max(worst, err)
            assert err < 1e-6, (name, fi, num, g.reshape(-1)[fi])
    st.set_params(params)
    print(f"sem layers={layers} worst normalized abs error {worst:.2e}")
    assert worst < 1e-6


def test_sem_loss_matches_independent_numpy_reference():
    """sem loss の値を独立実装（文脈なし GRU を素の numpy で回して cosine）と比較。無効候補は平均に入らない。"""
    lam = 0.5
    cfg, be, st, params = build_sem(1, lam, B=6)
    sh = tiny_shard()
    sem = tiny_sem(sh)
    bt = M.make_batch(sh, np.arange(6), keys=np.random.default_rng(3).random((6, 4)), sem=sem)
    got = st.loss_grads(bt, want_grads=False)
    H = cfg.hidden
    emb = params["emb"]
    wi, wh, bi, bh = (params[f"l0.{n}"] for n in ("wi", "wh", "bi", "bh"))
    vals = []
    for b in range(6):
        for j in range(bt.K):
            if not bt.sem_valid[b * bt.K + j]:
                continue
            h = np.zeros(H)
            clen = int(bt.ints[bt.T * bt.B + bt.Tc * bt.B * bt.K + bt.B + b * bt.K + j])
            toks = bt.ints[bt.T * bt.B:bt.T * bt.B + bt.Tc * bt.B * bt.K].reshape(bt.Tc, -1)[:clen, b * bt.K + j]
            for tok in toks:
                gi = emb[tok] @ wi + bi
                gh = h @ wh + bh
                r_ = 1 / (1 + np.exp(-(gi[:H] + gh[:H])))
                z_ = 1 / (1 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
                n_ = np.tanh(gi[2 * H:] + r_ * gh[2 * H:])
                h = (1 - z_) * n_ + z_ * h
            z = h @ params["sem.wp"] + params["sem.bp"]
            t = bt.sem_t[b * bt.K + j].astype(np.float64)
            vals.append(1 - z @ t / (np.linalg.norm(z) * np.linalg.norm(t)))
    assert len(vals) == int(bt.sem_valid.sum())
    assert abs(got["sem"] - np.mean(vals)) < 1e-9
    base = M.Student(NPBackend(np.float64), cfg, 6, 4, train=True)
    base.set_params(params)
    assert abs(got["loss"] - (base.loss_grads(bt, want_grads=False)["loss"] + lam * np.mean(vals))) < 1e-9


def test_sem_lambda_zero_is_bit_identical_to_no_sem():
    """λ=0（sem_dim の有無に依らず）: 追加 forward を行わず、loss・勾配・1 step 後の weight が sem 無し構成とビット一致。
    projection head 以外の init も同一（sem は別 stream の init）。float32 numpy backend。"""
    cfg0 = M.Config(vocab=40, emb=5, hidden=6, layers=2, lp=9, lc=5)
    cfg1 = M.Config(vocab=40, emb=5, hidden=6, layers=2, lp=9, lc=5, sem_dim=3)
    p0, p1 = M.init_params(cfg0, 4), M.init_params(cfg1, 4)
    assert all(np.array_equal(p0[k], p1[k]) for k in p0) and set(p1) - set(p0) == {"sem.wp", "sem.bp"}
    sh = tiny_shard()
    sem = tiny_sem(sh)
    s0 = M.Student(NPBackend(np.float32), cfg0, 6, 4, train=True)
    s1 = M.Student(NPBackend(np.float32), cfg1, 6, 4, train=True)
    calls = {"n": 0}
    orig = s1._sem_pass
    s1._sem_pass = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
    s0.set_params(p0)
    s1.set_params(p1)
    s1.sem_lambda = 0.0
    for step in range(1, 4):
        idx = np.random.default_rng(step).permutation(6)[:5]
        keys = np.random.default_rng(step + 9).random((6, 4))[idx]
        r0 = s0.train_step(M.make_batch(sh, idx, keys=keys), 2e-3, step)
        r1 = s1.train_step(M.make_batch(sh, idx, keys=keys, sem=sem), 2e-3, step)
        assert r0["loss"] == r1["loss"] and r0["gnorm"] == r1["gnorm"] and r1["sem"] == 0.0
    assert calls["n"] == 0
    a, b = s0.get_params(), s1.get_params()
    assert all(np.array_equal(a[k], b[k]) for k in a)
    # λ>0 では sem 経路が実際に動き、base params の勾配が変わる
    s1.sem_lambda = 0.5
    r = s1.train_step(M.make_batch(sh, np.arange(6), sem=sem), 2e-3, 4)
    assert calls["n"] == 1 and r["sem"] > 0


def test_sem_pass_does_not_touch_predict_path():
    """sem 追加 pass は学習時のみ。predict（判断用 score）は sem_dim/λ の有無で同一 weight なら同一値。"""
    cfg, be, st, params = build_sem(2, 0.5)
    cfg0 = M.Config(vocab=40, emb=5, hidden=6, layers=2, lp=9, lc=5)
    s0 = M.Student(NPBackend(np.float64), cfg0, 6, 4, train=False)
    s0.set_params({k: v for k, v in params.items() if not k.startswith("sem.")})
    sh = tiny_shard()
    sem = tiny_sem(sh)
    bt = M.make_batch(sh, np.arange(6), sem=sem)
    ref = st.predict(bt)
    st.loss_grads(bt)
    assert np.array_equal(st.predict(bt), ref)
    assert np.array_equal(s0.predict(bt), ref)


def test_sem_requires_targets_and_train_mode():
    cfg, be, st, params = build_sem(1, 0.5)
    sh = tiny_shard()
    with pytest.raises(ValueError):
        st.loss_grads(M.make_batch(sh, np.arange(6)))   # sem target 無し
    ev = M.Student(NPBackend(np.float64), cfg, 6, 4, train=False)
    ev.sem_lambda = 0.5
    with pytest.raises(ValueError):
        ev.loss_grads(M.make_batch(sh, np.arange(6), sem=tiny_sem(sh)), want_grads=False)
