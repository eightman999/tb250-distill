"""合成＋公開データの混合学習の準備機能: tokenize_data の --max-per-source（train のみ・決定的抽出）と
train の --source-loss（source 別の損失重み。np float64 の数値勾配で検証、cl 一致は test_student_cl.py）。"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill import replay, tokenize_data as TD  # noqa: E402
from tb250distill.student import model as M, evaluate as E, train as T, fake_data as F  # noqa: E402
from tb250distill.student.backend_np import NPBackend  # noqa: E402
from tests.test_student_gradcheck import tiny_shard  # noqa: E402

# --------------------------------------------------------------------------- tokenize_data --max-per-source


def test_parse_max_per_source():
    assert TD.parse_max_per_source(None) == {} and TD.parse_max_per_source("") == {}
    assert TD.parse_max_per_source("synth=30000") == {"synth": 30000}
    assert TD.parse_max_per_source(["synth=30000,jnli=5", "wrime=7"]) == {"synth": 30000, "jnli": 5, "wrime": 7}
    for bad in ("synth", "synth=0", "synth=-1", "synth=x", "=3", "a=1,a=2"):
        with pytest.raises(ValueError):
            TD.parse_max_per_source(bad)


def _rows(n_synth=200, n_jnli=30):
    rows = [{"item_id": i, "source": "synth"} for i in range(1, n_synth + 1)]
    rows += [{"item_id": 1000 + i, "source": "jnli"} for i in range(n_jnli)]
    return rows


def test_cap_per_source_deterministic_and_counts():
    rows = _rows()
    a = TD.cap_per_source(rows, {"synth": 50}, seed=7)
    b = TD.cap_per_source(list(reversed(rows)), {"synth": 50}, seed=7)
    assert [r["item_id"] for r in a] == sorted(r["item_id"] for r in b)  # 行の並びに依らず同じ集合
    assert [r["item_id"] for r in a] == [r["item_id"] for r in TD.cap_per_source(rows, {"synth": 50}, seed=7)]
    assert sum(r["source"] == "synth" for r in a) == 50 and sum(r["source"] == "jnli" for r in a) == 30  # 指定外は全件
    ids = [r["item_id"] for r in a]
    assert ids == sorted(ids)  # 元の並び（item_id 昇順）を保つ
    assert ids[:50] != list(range(1, 51))  # 先頭 N 件ではなくランダム抽出
    c = TD.cap_per_source(rows, {"synth": 50}, seed=8)
    assert [r["item_id"] for r in c] != ids  # seed が変われば別の抽出
    assert len(TD.cap_per_source(rows, {"synth": 500}, seed=7)) == len(rows)  # 上限が件数以上なら全件
    assert TD.cap_per_source(rows, {}, seed=7) is rows  # 指定なしは無加工
    # 追加の item が増えても、残った item は（cap 内なら）hash 順位で決まるため集合の包含関係が安定
    more = rows + [{"item_id": 5000, "source": "synth"}]
    assert {r["item_id"] for r in TD.cap_per_source(more, {"synth": 50}, seed=7)} - {5000} <= {r["item_id"] for r in a} | {5000}


def test_build_shard_cap_train_only_and_meta_counts(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    conn = replay.connect(str(tmp_path / "r.sqlite"))
    rows = []
    iid = 0
    for split in ("train", "val"):
        for src, n in (("synth", 40), ("jnli", 10)):
            for _ in range(n):
                iid += 1
                rows.append(dict(item_id=iid, split=split, category="intent", lang="ja", context=f"電子機器の話 {iid} " * 3,
                                 question="どれ", candidates=[f"候補{j}" for j in range(3)], gold=iid % 3, variant_of=None,
                                 variant=None, gen_seed=iid, source=src))
    replay.insert_items(conn, rows)
    for r in rows:
        replay.write_teacher(conn, r["item_id"], "m", "x", [0.1, 0.5, 0.2], [0.3, 0.4, 0.3], None, commit=False)
    conn.commit()
    conn.close()
    ro = replay.connect(str(tmp_path / "r.sqlite"), readonly=True)
    out = tmp_path / "tok"
    TD.train_spm(ro, out, vocab=300)
    sp = spm.SentencePieceProcessor(model_file=str(out / "spm.model"))
    caps = {"synth": 12}
    a1, s1 = TD.build_shard(ro, sp, "train", 32, 8, 5, 1.0, max_per_source=caps, sample_seed=3)
    a2, s2 = TD.build_shard(ro, sp, "train", 32, 8, 5, 1.0, max_per_source=caps, sample_seed=3)
    assert (a1["item_id"] == a2["item_id"]).all() and s1["n"] == 22
    assert s1["source_counts"] == {"jnli": 10, "synth": 12} and s1["source_counts_before_cap"] == {"jnli": 10, "synth": 40}
    assert s1["max_per_source"] == caps and s1["by_source"]["synth"]["n"] == 12
    # val は上限を無視（全件）
    av, sv = TD.build_shard(ro, sp, "val", 32, 8, 5, 1.0, max_per_source=caps, sample_seed=3)
    assert sv["n"] == 50 and sv["source_counts"] == {"jnli": 10, "synth": 40} and "max_per_source" not in sv
    # 既定（指定なし）は従来どおり全件
    a0, s0 = TD.build_shard(ro, sp, "train", 32, 8, 5, 1.0)
    assert s0["n"] == 50 and "max_per_source" not in s0
    # CLI: meta.json に source 別件数と上限が残る
    rc = TD.main(["--db", str(tmp_path / "r.sqlite"), "--name", "t", "--out-root", str(tmp_path / "tok2"), "--vocab", "300",
                  "--splits", "train,val", "--max-per-source", "synth=12", "--lp", "32", "--lc", "8"])
    assert rc == 0
    meta = json.load(open(tmp_path / "tok2" / "t" / "meta.json"))
    assert meta["max_per_source"] == {"synth": 12} and meta["splits"]["train"]["source_counts"] == {"jnli": 10, "synth": 12}
    assert meta["splits"]["val"]["source_counts"] == {"jnli": 10, "synth": 40}
    z = np.load(tmp_path / "tok2" / "t" / "train_L32.npz")
    assert z["source"].shape == (22,)


# --------------------------------------------------------------------------- --source-loss


def test_parse_source_loss():
    assert M.parse_source_loss(None) == {} and M.parse_source_loss([]) == {}
    assert M.parse_source_loss("jnli:kd=0.2,ce=0.8") == {"jnli": (0.2, 0.8)}
    assert M.parse_source_loss(["jnli:kd=0.2,ce=0.8;wrime:ce=0.5,kd=0.5", "x:kd=1,ce=0"]) == \
        {"jnli": (0.2, 0.8), "wrime": (0.5, 0.5), "x": (1.0, 0.0)}
    for bad in ("jnli", "jnli:kd=0.2", "jnli:kd=a,ce=1", "jnli:kd=-1,ce=1", "jnli:kd=nan,ce=1", "jnli:kd=1,ce=1,kd=2",
                "jnli:kd=1,ce=1;jnli:kd=0,ce=1", ":kd=1,ce=1", "jnli:kd=1,xx=1"):
        with pytest.raises(ValueError):
            M.parse_source_loss(bad)


def with_sources(sh, names):
    d = {k: getattr(sh, k) for k in M.SHARD_KEYS}
    d["source"] = np.array(names, dtype="U16")
    return M.Shard(d)


def build_student(seed=1, layers=2):
    cfg = M.Config(vocab=40, emb=5, hidden=6, layers=layers, lp=9, lc=5)
    be = NPBackend(np.float64)
    st = M.Student(be, cfg, 6, 4, train=True)
    params = {k: (v * 2.0).astype(np.float64) for k, v in M.init_params(cfg, seed).items()}
    st.set_params(params)
    return cfg, st, params


SRC = ["synth", "jnli", "synth", "jnli", "wrime", "jnli"]   # tiny_shard の gold: i=0,2,3,5 が gold 有り（1,4 は無し）
SL = {"jnli": (0.2, 0.8), "wrime": (0.0, 1.5)}


def test_source_weights_assignment():
    sh = with_sources(tiny_shard(), SRC)
    w_kd, w_ce = M.source_weights(sh, np.arange(6), SL)
    assert w_kd.tolist() == [M.W_KD_GOLD, 0.2, M.W_KD_GOLD, 0.2, 0.0, 0.2]
    assert w_ce.tolist() == [M.W_CE_GOLD, 0.8, M.W_CE_GOLD, 0.8, 1.5, 0.8]
    assert M.source_weights(sh, np.arange(6), {}) is None
    assert M.source_weights(tiny_shard(), np.arange(6), SL) is None   # source キーの無い shard は無効
    assert M.make_batch(tiny_shard(), np.arange(6), source_loss=SL).w_kd is None
    sub = sh.subset(np.array([4, 1]))
    assert M.source_weights(sub, np.arange(2), SL)[1].tolist() == [1.5, 0.8]


def test_source_loss_numeric_gradient_and_loss_value():
    cfg, st, params = build_student()
    sh = with_sources(tiny_shard(), SRC)
    keys = np.random.default_rng(3).random((6, 4))
    bt = M.make_batch(sh, np.arange(6), keys=keys, source_loss=SL)
    assert bt.w_kd is not None
    r = st.loss_grads(bt)
    grads = st.get_grads()
    # loss 値: evaluate.composite_loss を sample ごとの重みで組み直したものと一致
    scores = st.predict(bt)
    tl = bt.tlog.reshape(6, -1).astype(np.float64)
    _, kd, ce = E.composite_loss(scores, tl, bt.kcnt, bt.gold)
    has = bt.gold >= 0
    tot = np.where(has, bt.w_kd * kd + bt.w_ce * ce, kd)
    assert abs(r["loss"] - tot.mean()) < 1e-12
    base = np.where(has, M.W_KD_GOLD * kd + M.W_CE_GOLD * ce, kd).mean()
    assert abs(r["loss"] - base) > 1e-3  # 重みが実際に効いている
    # 数値勾配
    rng = np.random.default_rng(7)
    eps = 1e-6
    worst = 0.0
    for name, shape in M.param_specs(cfg):
        g = grads[name]
        flat = np.arange(int(np.prod(shape)))
        if len(flat) > 25:
            flat = rng.choice(flat, 25, replace=False)
        for fi in flat:
            p = {k: v.copy() for k, v in params.items()}
            p[name].reshape(-1)[fi] += eps
            st.set_params(p)
            lp_ = st.loss_grads(bt, want_grads=False)["loss"]
            p[name].reshape(-1)[fi] -= 2 * eps
            st.set_params(p)
            lm_ = st.loss_grads(bt, want_grads=False)["loss"]
            err = abs((lp_ - lm_) / (2 * eps) - g.reshape(-1)[fi]) / max(1e-3, np.abs(g).max())
            worst = max(worst, err)
    st.set_params(params)
    assert worst < 1e-6, worst


def test_no_source_loss_is_identical_to_before():
    cfg, st, params = build_student()
    base_sh = tiny_shard()
    src_sh = with_sources(base_sh, SRC)
    keys = np.random.default_rng(3).random((6, 4))
    idx = np.arange(6)

    def run(sh, sl):
        st.set_params(params)
        r = st.loss_grads(M.make_batch(sh, idx, keys=keys, source_loss=sl))
        return r, st.get_grads()

    r0, g0 = run(base_sh, None)
    # 1) 指定なし（source キー有りの shard でも）
    r1, g1 = run(src_sh, None)
    # 2) 指定あり・source キー無しの shard
    r2, g2 = run(base_sh, SL)
    # 3) 指定あり・該当 source が無い（配列経路で既定重み）
    r3, g3 = run(src_sh, {"nonexistent": (0.1, 0.9)})
    for r, g in ((r1, g1), (r2, g2), (r3, g3)):
        assert r == r0
        assert all(np.array_equal(g[k], g0[k]) for k in g0)
    # 従来の定義（既定重み）と一致
    scores = st.predict(M.make_batch(base_sh, idx, keys=keys))
    bt = M.make_batch(base_sh, idx, keys=keys)
    loss, _, _ = E.composite_loss(scores, bt.tlog.reshape(6, -1).astype(np.float64), bt.kcnt, bt.gold)
    assert abs(r0["loss"] - loss.mean()) < 1e-12
    # 同じ st で source 別重みの batch の後に、指定なしの batch が汚染されない
    run(src_sh, SL)
    r4, g4 = run(base_sh, None)
    assert r4 == r0 and all(np.array_equal(g4[k], g0[k]) for k in g0)


def test_kd_loss_backend_array_weights_match_scalar():
    rng = np.random.default_rng(2)
    be = NPBackend(np.float64)
    B, K = 9, 5
    sc, tl = rng.normal(size=(B, K)) * 2, rng.normal(size=(B, K)) * 2
    k = rng.integers(1, K + 1, B).astype(np.int32)
    gold = np.where(rng.random(B) < 0.6, rng.integers(0, 10, B) % k, -1).astype(np.int32)

    def call(wk, wc):
        ds, ls = np.zeros(B * K), np.zeros(B * 3)
        be.kd_loss(sc.ravel(), tl.ravel(), k, gold, ds, ls, B, K, 2.0, wk, wc, 1.0 / B)
        return ds, ls

    a = call(0.8, 0.2)
    b = call(np.full(B, 0.8), np.full(B, 0.2))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    wk, wc = rng.random(B), rng.random(B)
    ds, ls = call(wk, wc)
    for i in range(B):   # sample ごとに、その重みのスカラー呼び出しと一致
        ds1, ls1 = call(wk[i], wc[i])
        assert np.array_equal(ds[i * K:(i + 1) * K], ds1[i * K:(i + 1) * K]) and np.array_equal(ls[i * 3:(i + 1) * 3], ls1[i * 3:(i + 1) * 3])


# --------------------------------------------------------------------------- train 統合（np backend）

CFG = '{"vocab": 300, "emb": 16, "hidden": 24, "layers": 2}'


@pytest.fixture(scope="module")
def fake_src(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_src")
    F.write_fake_dataset(str(d), n_train=200, n_val=60, n_test=40, n_robust=20, vocab=300, lp=48, lc=8, kmax=5, seed=1, active=30)
    return str(d)


def add_source(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        p = os.path.join(src_dir, f)
        if f.endswith(".npz") and not f.startswith("robust"):
            z = dict(np.load(p, allow_pickle=False))
            n = z["item_id"].shape[0]
            z["source"] = np.where(np.arange(n) % 2 == 0, "synth", "jnli").astype("U16")
            np.savez(os.path.join(dst_dir, f), **z)
        elif os.path.isfile(p):
            open(os.path.join(dst_dir, f), "wb").write(open(p, "rb").read())


def run_train(data, run, extra=()):
    return T.main(["--backend", "np", "--data", data, "--run-dir", str(run), "--config", CFG, "--batch-size", "16",
                   "--seed", "3", "--log-every", "5", "--eval-every", "10", "--epochs", "1", "--max-steps", "6", *extra])


def losses(run):
    rows = [l.split(",") for l in open(run / "metrics.csv").read().strip().splitlines()]
    h = rows[0]
    return [r[h.index("train_loss")] for r in rows[1:] if r[h.index("train_loss")]]


def test_train_source_loss_end_to_end(fake_src, tmp_path):
    data = str(tmp_path / "with_src")
    add_source(fake_src, data)
    plain, withsl, nosrc_sl, nosrc = (tmp_path / n for n in ("plain", "withsl", "nosrc_sl", "nosrc"))
    run_train(data, plain)
    run_train(data, withsl, ["--source-loss", "jnli:kd=0.2,ce=0.8"])
    # 指定なし == 従来（source キー有り/無しに依らず）
    run_train(fake_src, nosrc)
    run_train(fake_src, nosrc_sl, ["--source-loss", "jnli:kd=0.2,ce=0.8"])  # source キー無し -> 警告して無視
    assert losses(plain) == losses(nosrc) == losses(nosrc_sl)
    assert losses(plain) != losses(withsl)
    c = json.load(open(withsl / "config.json"))
    assert c["loss"]["source_loss"] == {"jnli": {"kd": 0.2, "ce": 0.8}} and c["loss"]["source_loss_active"] is True
    assert c["args"]["source_loss"] == ["jnli:kd=0.2,ce=0.8"]
    c2 = json.load(open(plain / "config.json"))
    assert c2["loss"]["source_loss"] == {} and c2["loss"]["source_loss_active"] is False
    assert json.load(open(nosrc_sl / "config.json"))["loss"]["source_loss_active"] is False
    # by_source が最終評価・eval.json に載る
    ev = json.load(open(withsl / "eval.json"))
    assert set(ev["by_source"]["val"]) == {"synth", "jnli"} and set(ev["by_source"]["test"]) == {"synth", "jnli"}
    assert ev["by_source"]["val"]["jnli"]["n"] + ev["by_source"]["val"]["synth"]["n"] == ev["splits"]["val"]["n"]
    assert "FINAL val/jnli" in open(withsl / "train.log").read()
    assert "by_source" not in json.load(open(nosrc / "eval.json"))
    with pytest.raises(SystemExit):
        run_train(data, tmp_path / "bad", ["--source-loss", "jnli:kd=0.2"])
