"""Candidate Semantic Distillation: embed.py（リーク防止・PCA・再開・出力契約）と train.py 結線（np backend）。"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill import replay  # noqa: E402
from tb250distill.teacher import embed as EM  # noqa: E402
from tb250distill.student import model as M, train as T, evaluate as E, fake_data as F  # noqa: E402


# ---------------------------------------------------------------------------------------
# embed.py
# ---------------------------------------------------------------------------------------

class FakeProvider(EM.EmbeddingProvider):
    """文字列の sha で決まる D 次元ベクトル。呼ばれた文字列を全部記録する（リーク検証用）。"""
    name = "fake"

    def __init__(self, D=24, fail_after=None):
        self.D = D
        self.seen = []
        self.calls = 0
        self.fail_after = fail_after

    def embed(self, texts):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("boom")
        self.calls += 1
        self.seen += list(texts)
        out = []
        for t in texts:
            r = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:12], 16))
            out.append(r.normal(size=self.D) * (1 + (len(t) % 5)))
        return np.array(out, np.float32)


def make_db(path, n_train=40, n_val=12, n_test=12, n_robust=6, seed=0):
    """train/val/test/robust の item。val/test/robust には train に無い文字列（候補）が含まれ、一部は train と同じ文字列。"""
    rng = np.random.default_rng(seed)
    conn = replay.connect(str(path))
    rows, iid = [], 0
    spec = [("train", n_train, 0), ("val", n_val, 100000), ("test", n_test, 200000), ("robust", n_robust, 300000)]
    for split, n, base in spec:
        for i in range(n):
            k = int(rng.integers(2, 5))
            cands = []
            for j in range(k):
                if split != "train" and j == 0:
                    cands.append(f"shared-{rng.integers(0, 6)}")        # train にも出る文字列（train 側の集合に入ってよい）
                elif split == "train":
                    cands.append(f"tr-{rng.integers(0, 15)}-{j}")
                else:
                    cands.append(f"{split}-ONLY-{rng.integers(0, 1000)}-{j}")  # held-out にだけ出る文字列
            if split == "train":
                cands[0] = f"shared-{rng.integers(0, 6)}"
            cands = list(dict.fromkeys(cands))
            rows.append(dict(item_id=base + i, split=split, category="nli", lang="en", context=f"c{i}", question="q",
                             candidates=cands, gold=0, gen_seed=1))
    replay.insert_items(conn, rows)
    return conn


def make_shard(shard_dir, conn, splits=("train",), lp=8, kmax=4):
    os.makedirs(shard_dir, exist_ok=True)
    for sp in splits:
        r = conn.execute("SELECT item_id, candidates FROM items WHERE split=? ORDER BY item_id", (sp,)).fetchall()
        ids = np.array([x[0] for x in r], np.int64)
        k = np.array([len(json.loads(x[1])) for x in r], np.int32)
        np.savez(os.path.join(shard_dir, f"{sp}_L{lp}.npz"), item_id=ids, k=k)
    return shard_dir


def test_extract_only_train_strings_and_never_embeds_heldout(tmp_path):
    conn = make_db(tmp_path / "r.sqlite")
    shard = make_shard(tmp_path / "tok" / "toyA", conn, splits=("train",))
    # val/test/robust の shard が置いてあっても読まない（壊れたファイルを置いておき、開かれたら失敗する）
    for sp in ("val", "test", "robust"):
        (tmp_path / "tok" / "toyA" / f"{sp}_L8.npz").write_bytes(b"corrupt")
    prov = FakeProvider()
    meta = EM.extract(prov, shard, str(tmp_path / "r.sqlite"), str(tmp_path / "emb"), provider_name="fake", lp=8, dim=8, chunk=7,
                      log=lambda *_: None)
    out = tmp_path / "emb" / "fake" / "toyA"
    strings = json.loads((out / "strings.json").read_text())
    held = {c for sp in ("val", "test", "robust")
            for (cj,) in conn.execute("SELECT candidates FROM items WHERE split=?", (sp,)) for c in json.loads(cj)}
    train = {c for (cj,) in conn.execute("SELECT candidates FROM items WHERE split='train'") for c in json.loads(cj)}
    held_only = held - train
    assert held_only and any("ONLY" in c for c in held_only)
    assert set(strings) == train and len(strings) == len(set(strings))
    assert not (set(strings) & held_only)
    assert set(prov.seen) == train and len(prov.seen) == len(train)       # provider に渡った文字列も train の unique 集合だけ（重複送信なし）
    assert not (set(prov.seen) & held_only)
    assert meta["split"] == "train" and meta["U"] == len(train)
    # 出力ファイル契約
    for f in ("strings.json", "emb_raw.npy", "pca.npz", "emb_pca.npy", "cand_idx_train.npy", "meta.json"):
        assert (out / f).exists(), f
    raw = np.load(out / "emb_raw.npy")
    ci = np.load(out / "cand_idx_train.npy")
    assert raw.dtype == np.float32 and raw.shape == (len(strings), 24) and ci.dtype == np.int32
    # cand_idx: 行 × 候補位置 -> unique index。候補の無い位置は -1
    for r, (iid,) in enumerate(conn.execute("SELECT item_id FROM items WHERE split='train' ORDER BY item_id")):
        cands = json.loads(conn.execute("SELECT candidates FROM items WHERE item_id=?", (int(iid),)).fetchone()[0])
        assert [strings[u] for u in ci[r, :len(cands)]] == cands and (ci[r, len(cands):] == -1).all()


def test_leak_guard_rejects_non_train_items_in_shard(tmp_path):
    conn = make_db(tmp_path / "r.sqlite")
    d = tmp_path / "tok" / "bad"
    os.makedirs(d)
    tr = [r[0] for r in conn.execute("SELECT item_id FROM items WHERE split='train' ORDER BY item_id LIMIT 3")]
    va = [r[0] for r in conn.execute("SELECT item_id FROM items WHERE split='val' ORDER BY item_id LIMIT 2")]
    ids = np.array(tr + va, np.int64)
    k = np.array([len(json.loads(conn.execute("SELECT candidates FROM items WHERE item_id=?", (int(i),)).fetchone()[0])) for i in ids], np.int32)
    np.savez(d / "train_L8.npz", item_id=ids, k=k)
    prov = FakeProvider()
    with pytest.raises(EM.LeakError):
        EM.extract(prov, d, str(tmp_path / "r.sqlite"), str(tmp_path / "emb"), provider_name="fake", lp=8, log=lambda *_: None)
    assert prov.seen == []   # embedding を 1 件も計算していない


def test_extract_is_resumable_and_deterministic(tmp_path):
    conn = make_db(tmp_path / "r.sqlite")
    shard = make_shard(tmp_path / "tok" / "toyA", conn)
    db = str(tmp_path / "r.sqlite")
    ref = EM.extract(FakeProvider(), shard, db, str(tmp_path / "ref"), provider_name="fake", lp=8, dim=8, chunk=5, log=lambda *_: None)
    p1 = FakeProvider(fail_after=3)
    with pytest.raises(RuntimeError):
        EM.extract(p1, shard, db, str(tmp_path / "emb"), provider_name="fake", lp=8, dim=8, chunk=5, log=lambda *_: None)
    assert len(os.listdir(tmp_path / "emb" / "fake" / "toyA" / "parts")) == 3
    p2 = FakeProvider()
    meta = EM.extract(p2, shard, db, str(tmp_path / "emb"), provider_name="fake", lp=8, dim=8, chunk=5, log=lambda *_: None)
    assert meta["resumed_chunks"] == 3 and len(p2.seen) == ref["U"] - 15
    a, b = tmp_path / "ref" / "fake" / "toyA", tmp_path / "emb" / "fake" / "toyA"
    for f in ("emb_raw.npy", "emb_pca.npy", "cand_idx_train.npy"):
        assert np.array_equal(np.load(a / f), np.load(b / f)), f
    za, zb = np.load(a / "pca.npz"), np.load(b / "pca.npz")
    assert all(np.array_equal(za[k], zb[k]) for k in za.files)


def test_pca_contract():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 6)) * np.array([5, 3, 2, 1, .5, .1])
    x = base @ rng.normal(size=(6, 40)) + 15.0 * rng.normal(size=(1, 40))   # 平均方向が支配的
    pca, ep = EM.fit_pca(x, 6)
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert np.allclose(pca["mean"], xn.mean(0))                      # L2 正規化後の平均（train unique 集合のみ）
    C = pca["components"]
    assert C.shape == (6, 40) and np.allclose(C @ C.T, np.eye(6), atol=1e-9)
    assert np.allclose(ep, (xn - xn.mean(0)) @ C.T, atol=1e-5)
    assert np.allclose(ep.var(axis=0, ddof=1), pca["explained_variance"], rtol=1e-4)
    assert 0.9 < pca["explained_variance_ratio"].sum() <= 1.0 + 1e-9
    assert (np.diff(pca["explained_variance"]) <= 1e-12).all()
    # 決定性・符号固定
    pca2, ep2 = EM.fit_pca(x, 6)
    assert np.array_equal(pca["components"], pca2["components"]) and np.array_equal(ep, ep2)
    # d が大きすぎるときは U-1 / D で頭打ち
    p3, e3 = EM.fit_pca(x[:5], 128)
    assert p3["components"].shape[0] == 4
    # 平均を引いた cosine は、生の cosine（全部 ~1）より情報を持つ
    raw_cos = (xn @ xn.T)[np.triu_indices(300, 1)]
    ec = ep / np.linalg.norm(ep, axis=1, keepdims=True)
    pc = (ec @ ec.T)[np.triu_indices(300, 1)]
    assert raw_cos.mean() > 0.5 and abs(pc.mean()) < 0.2


def test_presets_and_server_command():
    assert set(EM.PRESETS) >= {"qwen3_1p7b", "qwen3_emb_0p6b"}
    assert EM.PRESETS["qwen3_1p7b"].pooling == "mean" and EM.PRESETS["qwen3_emb_0p6b"].pooling == "last"
    for sp in EM.PRESETS.values():
        cmd = EM.server_cmd(sp)
        assert "--embeddings" in cmd and sp.port != EM.TEACHER_PORT
        assert cmd[cmd.index("--pooling") + 1] == sp.pooling
    with pytest.raises(SystemExit, match="18080"):   # 既存 teacher のポートでは起動しない
        EM.serve_start("qwen3_1p7b", port=EM.TEACHER_PORT)
    p = EM.make_provider("qwen3_emb_0p6b", url="http://127.0.0.1:1", batch=8)
    assert p.pooling == "last" and p.batch == 8 and p.describe()["preset"] == "qwen3_emb_0p6b"


# ---------------------------------------------------------------------------------------
# train.py 結線（np backend, fake shard）
# ---------------------------------------------------------------------------------------

CFG = '{"vocab": 300, "emb": 24, "hidden": 32, "layers": 2}'


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_sem")
    F.write_fake_dataset(str(d), n_train=600, n_val=100, n_test=100, n_robust=20, vocab=300, lp=48, lc=8, kmax=5, seed=1, active=30)
    return str(d)


def write_sem_dir(fake, out, d=8, seed=0, break_sha=False):
    """候補トークン列から決まる target（学習可能）: 各 token に d 次元ベクトル、候補 = 平均。unique は token 列の完全一致。"""
    sh = M.Shard(M.find_shard(fake, "train", 48))
    rng = np.random.default_rng(seed)
    tokvec = rng.normal(size=(300, d))
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
    emb = np.array(vecs, np.float32)
    np.save(os.path.join(out, "emb_pca.npy"), emb)
    np.save(os.path.join(out, "cand_idx_train.npy"), ci)
    sha = M.item_id_sha1(sh.item_id if not break_sha else sh.item_id[::-1])
    json.dump({"provider": "toy", "shard": "fake", "item_id_sha1": sha, "explained_variance_ratio": 1.0},
              open(os.path.join(out, "meta.json"), "w"))
    return str(out), emb.shape[0]


def run_train(fake, run, extra=()):
    argv = ["--backend", "np", "--data", fake, "--run-dir", str(run), "--config", CFG, "--batch-size", "32", "--seed", "3",
            "--log-every", "10", "--eval-every", "30", "--lr", "3e-3", "--lp", "48", "--epochs", "4", *extra]
    return T.main(argv)


def read_csv(run):
    rows = [l.split(",") for l in open(run / "metrics.csv").read().strip().splitlines()]
    return rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]


def test_train_with_sem_reduces_sem_loss_and_records(fake, tmp_path):
    sem_dir, U = write_sem_dir(fake, tmp_path / "sem")
    run = tmp_path / "run"
    run_train(fake, run, ["--epochs", "6", "--sem-cand-weight", "0.5", "--sem-emb", sem_dir, "--no-final-eval"])
    hdr, rows = read_csv(run)
    assert hdr[-1] == "sem_loss" and hdr == T.CSV_COLS
    sl = [float(r["sem_loss"]) for r in rows if r["sem_loss"]]
    assert len(sl) >= 5 and sl[-1] < 0.5 * sl[0], sl
    cfg = json.load(open(run / "config.json"))
    assert cfg["sem"]["enabled"] and cfg["sem"]["cand_weight"] == 0.5 and cfg["sem"]["sem_dim"] == 8
    assert cfg["model"]["sem_dim"] == 8 and cfg["n_params_breakdown"]["sem_head"] == 32 * 8 + 8
    assert cfg["sem"]["info"]["n_strings"] == U
    # checkpoint に projection head が入っている
    c, p, *_ = T.load_ckpt(str(run / "ckpt" / "last.npz"))
    assert c.sem_dim == 8 and p["sem.wp"].shape == (32, 8) and p["sem.bp"].shape == (8,)
    assert "sem_loss" in open(run / "train.log").read()
    # sem head 入り checkpoint でも評価 CLI が動く（head は推論で使わない）。同じ weight の head 無し構成と score が一致
    out = tmp_path / "ev.json"
    E.main(["--backend", "np", "--data", fake, "--lp", "48", "--ckpt", str(run / "ckpt" / "last.npz"), "--out", str(out),
            "--no-latency", "--batch-size", "32"])
    ev = json.load(open(out))
    assert ev["meta"]["config"]["sem_dim"] == 8 and ev["key_metrics"]["massive_test_agreement"] is None
    cfg_s, p_s, _ = M.load_params_npz(str(run / "ckpt" / "last.npz"))
    cfg_0 = M.Config.from_dict({**cfg_s.to_dict(), "sem_dim": 0})
    sh = M.Shard(M.find_shard(fake, "val", 48))
    be = T.M.make_backend("np")
    ms = M.Student(be, cfg_s, 32, sh.kmax, train=False, max_lp=sh.lp, max_lc=sh.lc)
    m0 = M.Student(be, cfg_0, 32, sh.kmax, train=False, max_lp=sh.lp, max_lc=sh.lc)
    ms.set_params(p_s)
    m0.set_params({k: v for k, v in p_s.items() if not k.startswith("sem.")})
    assert np.array_equal(E.predict_scores(ms, sh), E.predict_scores(m0, sh))


def test_sem_off_is_bit_identical_to_baseline_run(fake, tmp_path):
    """--sem-cand-weight 無し / 0（--sem-emb を付けても）の run は、従来の run と weight・optimizer state がビット一致、sem_loss 列は空。"""
    sem_dir, _ = write_sem_dir(fake, tmp_path / "sem")
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    run_train(fake, a, ["--max-steps", "25", "--no-final-eval"])
    run_train(fake, b, ["--max-steps", "25", "--no-final-eval", "--sem-cand-weight", "0", "--sem-emb", sem_dir])
    ca, pa, ma, va, _ = T.load_ckpt(str(a / "ckpt" / "last.npz"))
    cb, pb, mb, vb, _ = T.load_ckpt(str(b / "ckpt" / "last.npz"))
    assert cb.sem_dim == 0 and set(pa) == set(pb)
    assert all(np.array_equal(pa[k], pb[k]) for k in pa) and np.array_equal(ma, mb) and np.array_equal(va, vb)
    _, rows = read_csv(b)
    assert all(r["sem_loss"] == "" for r in rows)


def test_sem_resume_is_exact(fake, tmp_path):
    sem_dir, _ = write_sem_dir(fake, tmp_path / "sem")
    ex = ["--no-final-eval", "--sem-cand-weight", "0.5", "--sem-emb", sem_dir]
    a, b = tmp_path / "a", tmp_path / "b"
    run_train(fake, a, ["--max-steps", "30", *ex])
    run_train(fake, b, ["--max-steps", "12", *ex])
    run_train(fake, b, ["--max-steps", "30", "--resume", *ex])
    ca, pa, ma, va, meta_a = T.load_ckpt(str(a / "ckpt" / "last.npz"))
    cb, pb, mb, vb, meta_b = T.load_ckpt(str(b / "ckpt" / "last.npz"))
    assert meta_a["step"] == meta_b["step"] == 30
    assert all(np.array_equal(pa[k], pb[k]) for k in pa) and "sem.wp" in pa
    assert np.array_equal(ma, mb) and np.array_equal(va, vb)
    # 構成が違う resume は拒否
    with pytest.raises(SystemExit):
        run_train(fake, b, ["--max-steps", "40", "--no-final-eval", "--resume"])


def test_sem_arg_validation(fake, tmp_path):
    sem_dir, _ = write_sem_dir(fake, tmp_path / "sem")
    with pytest.raises(SystemExit, match="未実装"):
        run_train(fake, tmp_path / "x1", ["--max-steps", "2", "--sem-ctx-weight", "0.1"])
    with pytest.raises(SystemExit, match="--sem-emb"):
        run_train(fake, tmp_path / "x2", ["--max-steps", "2", "--sem-cand-weight", "0.5"])
    bad, _ = write_sem_dir(fake, tmp_path / "sem_bad", break_sha=True)
    with pytest.raises(SystemExit, match="item_id"):
        run_train(fake, tmp_path / "x3", ["--max-steps", "2", "--sem-cand-weight", "0.5", "--sem-emb", bad])
    with pytest.raises(SystemExit, match="sem-dim"):
        run_train(fake, tmp_path / "x4", ["--max-steps", "2", "--sem-cand-weight", "0.5", "--sem-emb", sem_dir, "--sem-dim", "99"])
    # --sem-dim は PCA 次元以下ならその先頭成分
    run_train(fake, tmp_path / "x5", ["--max-steps", "3", "--no-final-eval", "--sem-cand-weight", "0.5", "--sem-emb", sem_dir,
                                      "--sem-dim", "4"])
    assert T.load_ckpt(str(tmp_path / "x5" / "ckpt" / "last.npz"))[0].sem_dim == 4


def test_sem_with_init_from_baseline_checkpoint(fake, tmp_path):
    """sem 無しで学習した checkpoint から --init で始めると、base weight はそのまま・projection head は seed 決定的に初期化。"""
    sem_dir, _ = write_sem_dir(fake, tmp_path / "sem")
    base = tmp_path / "base"
    run_train(fake, base, ["--max-steps", "5", "--no-final-eval"])
    init = str(base / "ckpt" / "p000.npz")
    r = tmp_path / "r"
    run_train(fake, r, ["--max-steps", "1", "--no-final-eval", "--init", init, "--sem-cand-weight", "0.5", "--sem-emb", sem_dir])
    p0 = M.load_params_npz(os.path.join(r, "ckpt", "p000.npz"))[1]
    pi = M.load_params_npz(init)[1]
    assert all(np.array_equal(p0[k], pi[k]) for k in pi)
    c0 = M.load_params_npz(os.path.join(r, "ckpt", "p000.npz"))[0]
    exp = M.init_sem_params(c0, 3)
    assert all(np.array_equal(p0[k], exp[k]) for k in exp)


def test_key_metrics_massive_test_agreement(fake, tmp_path):
    """source キーのある shard では eval.json top-level key_metrics.massive_test_agreement と train.log FINAL 行に出る。"""
    d = tmp_path / "srcshard"
    os.makedirs(d)
    for sp in ("train", "val", "test", "robust"):
        z = dict(np.load(os.path.join(fake, f"{sp}_L48.npz")))
        n = len(z["item_id"])
        z["source"] = np.where(np.arange(n) % 2 == 0, "massive", "synth").astype("U16")
        np.savez(d / f"{sp}_L48.npz", **z)
    run = tmp_path / "run"
    run_train(str(d), run, ["--max-steps", "20", "--eval-every", "10"])
    ev = json.load(open(run / "eval.json"))
    km = ev["key_metrics"]
    assert km["massive_test_agreement"] == ev["by_source"]["test"]["massive"]["agreement"]
    assert km["massive_test_n"] == 50 and 0 <= km["massive_test_agreement"] <= 1
    assert "FINAL key_metrics: massive_test_agreement" in open(run / "train.log").read()
    # source 無し shard では None（best 選択は val_kl のまま）
    run2 = tmp_path / "run2"
    run_train(fake, run2, ["--max-steps", "5"])
    assert json.load(open(run2 / "eval.json"))["key_metrics"]["massive_test_agreement"] is None
    assert json.load(open(run2 / "config.json"))["plan"]["best_metric"].startswith("val_kl")


# ---------------------------------------------------------------------------------------
# --exclude-truncated（切り詰め候補を意味表現蒸留 loss から除外）
# ---------------------------------------------------------------------------------------

LC_TOY = 6
SHORT = ["a b", "hello", "ok go", "yes", "no way"]
LONG = ["the quick brown fox jumps over the lazy dog again and again today", "one two three four five six seven eight nine ten eleven twelve"]


def make_spm(tmp_path):
    import sentencepiece as spm
    corpus = tmp_path / "spm_corpus.txt"
    words = "the quick brown fox jumps over lazy dog again and today one two three four five six seven eight nine ten eleven twelve a b hello ok go yes no way".split()
    rng = np.random.default_rng(0)
    corpus.write_text("\n".join(" ".join(rng.choice(words, size=int(rng.integers(2, 12)))) for _ in range(800)))
    pre = str(tmp_path / "spm_toy")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=pre, vocab_size=300, model_type="unigram", byte_fallback=True,
                                   character_coverage=1.0, pad_id=0, unk_id=1, bos_id=2, eos_id=3, user_defined_symbols=["<sep>"],
                                   minloglevel=2)
    return pre + ".model"


def make_trunc_fixture(tmp_path):
    """source 3 種（A: 短い候補だけ / B: 短い + 長い / C: 長い中心）の train。base extract を作って返す。"""
    conn = replay.connect(str(tmp_path / "r.sqlite"))
    rng = np.random.default_rng(1)
    rows, srcs = [], []
    for i in range(60):
        sname = ["srcA", "srcB", "srcC"][i % 3]
        k = int(rng.integers(2, 4))
        pool = SHORT if sname == "srcA" else (SHORT + LONG if sname == "srcB" else LONG + SHORT[:1])
        cands = list(rng.choice(pool, size=k, replace=False))
        rows.append(dict(item_id=i, split="train", category="nli", lang="en", context="c", question="q", candidates=cands, gold=0,
                         gen_seed=1))
        srcs.append(sname)
    replay.insert_items(conn, rows)
    shard = tmp_path / "tok" / "toyA"
    os.makedirs(shard)
    ids = np.arange(60, dtype=np.int64)
    k = np.array([len(r["candidates"]) for r in rows], np.int32)
    np.savez(shard / "train_L8.npz", item_id=ids, k=k, cand=np.zeros((60, 3, LC_TOY), np.int32), source=np.array(srcs))
    spm_path = make_spm(tmp_path)
    EM.extract(FakeProvider(), str(shard), str(tmp_path / "r.sqlite"), str(tmp_path / "emb"), provider_name="fake", lp=8, dim=8,
               log=lambda *_: None)
    return conn, rows, srcs, str(shard), spm_path, tmp_path / "emb" / "fake"


def test_exclude_truncated_masks_only_long_candidates_and_keeps_base(tmp_path):
    import sentencepiece as spm
    conn, rows, srcs, shard, spm_path, emb_root = make_trunc_fixture(tmp_path)
    base = emb_root / "toyA"
    snap = {f: (base / f).read_bytes() for f in os.listdir(base)}
    sources, lc = EM.load_train_shard_sources(shard, 8)
    assert lc == LC_TOY and list(sources[:3]) == ["srcA", "srcB", "srcC"]
    meta = EM.make_notrunc_variant(base, emb_root / "toyA_notrunc", spm_path, lc, sources, log=lambda *_: None)
    out = emb_root / "toyA_notrunc"
    # 元は 1 byte も変わらない
    assert {f: (base / f).read_bytes() for f in os.listdir(base)} == snap
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    assert all(len(sp.encode(t)) > LC_TOY for t in LONG) and all(len(sp.encode(t)) <= LC_TOY for t in SHORT)
    ci0, ci1 = np.load(base / "cand_idx_train.npy"), np.load(out / "cand_idx_train.npy")
    strings = json.loads((base / "strings.json").read_text())
    n_ex = 0
    per = {}
    for r, row in enumerate(rows):
        for j in range(ci0.shape[1]):
            if j >= len(row["candidates"]):
                assert ci0[r, j] == -1 and ci1[r, j] == -1
                continue
            long_ = len(sp.encode(row["candidates"][j]) or [1]) > LC_TOY
            assert ci0[r, j] >= 0
            assert ci1[r, j] == (-1 if long_ else ci0[r, j]), (r, j)
            n_ex += long_
            d = per.setdefault(srcs[r], [0, 0])
            d[0] += long_
            d[1] += 1
    assert n_ex > 0 and any(per[s][0] == 0 for s in per) and any(per[s][0] > 0 for s in per)   # 除外される source / されない source が両方ある
    ex = meta["exclude_truncated"]
    assert ex["applied"] and ex["lc"] == LC_TOY and ex["excluded_slots"] == n_ex
    for s_, (e, c) in per.items():
        assert ex["excluded_by_source"][s_]["excluded_slots"] == e and ex["excluded_by_source"][s_]["candidate_slots"] == c
    assert ex["excluded_by_source"]["srcA"]["excluded_slots"] == 0
    # 埋め込み本体は再抽出せず元を参照（ハードリンク）。strings / emb は同一
    for f in ("strings.json", "emb_raw.npy", "pca.npz", "emb_pca.npy"):
        assert os.path.samefile(base / f, out / f), f
    m2 = json.loads((out / "meta.json").read_text())
    assert m2["item_id_sha1"] == json.loads((base / "meta.json").read_text())["item_id_sha1"] and m2["variant"] == "notrunc"
    # 冪等。内容の違う既存出力は上書きしない
    EM.make_notrunc_variant(base, out, spm_path, lc, sources, log=lambda *_: None)
    with pytest.raises(SystemExit, match="上書きしない"):
        EM.make_notrunc_variant(base, out, spm_path, 2, sources, log=lambda *_: None)
    with pytest.raises(SystemExit, match="元は上書きしない"):
        EM.make_notrunc_variant(base, base, spm_path, lc, sources, log=lambda *_: None)
    with pytest.raises(SystemExit, match="既に"):
        EM.make_notrunc_variant(out, emb_root / "again", spm_path, lc, sources, log=lambda *_: None)


def test_notrunc_loads_for_training_and_does_not_change_judgement_batch(tmp_path):
    conn, rows, srcs, shard, spm_path, emb_root = make_trunc_fixture(tmp_path)
    sources, lc = EM.load_train_shard_sources(shard, 8)
    base, out = emb_root / "toyA", emb_root / "toyA_notrunc"
    EM.make_notrunc_variant(base, out, spm_path, lc, sources, log=lambda *_: None)
    z = np.load(os.path.join(shard, "train_L8.npz"))
    n, kmax = len(z["item_id"]), z["cand"].shape[1]
    rng = np.random.default_rng(0)
    sh = M.Shard(dict(item_id=z["item_id"], prefix=np.ones((n, 8), np.int32), prefix_len=np.full(n, 8, np.int32), cand=z["cand"] + 5,
                      cand_len=np.where(np.arange(kmax)[None, :] < z["k"][:, None], 3, 0).astype(np.int32), k=z["k"],
                      t_logits=rng.normal(size=(n, kmax)).astype(np.float32), gold=np.zeros(n, np.int32)))
    s0, i0 = M.load_sem_dir(str(base), sh.item_id)
    s1, i1 = M.load_sem_dir(str(out), sh.item_id)
    s0.check_against(sh)
    s1.check_against(sh)                              # 一部 -1 でも meta で宣言されているので通る
    assert s1.allow_masked and not s0.allow_masked and i1["exclude_truncated"]["excluded_slots"] > 0
    assert (s1.cand_idx >= 0).sum() < (s0.cand_idx >= 0).sum()
    # 宣言の無い SemTargets に同じ cand_idx を渡すと拒否される（誤用防止）。pad 位置に有効 index があれば宣言ありでも拒否
    with pytest.raises(ValueError, match="有効位置"):
        M.SemTargets(s1.cand_idx, s1.emb).check_against(sh)
    bad = s1.cand_idx.copy()
    bad[0, kmax - 1 if sh.k[0] < kmax else 0] = 0
    if sh.k[0] < kmax:
        with pytest.raises(ValueError, match="j >= k"):
            M.SemTargets(bad, s1.emb, allow_masked=True).check_against(sh)
    # 判断用の入力（候補・teacher logits・gold・kcnt）は notrunc でも完全に同じ。sem の有効位置だけが減る
    idx = np.arange(n)
    b0 = M.make_batch(sh, idx, sem=s0)
    b1 = M.make_batch(sh, idx, sem=s1)
    assert np.array_equal(b0.ints, b1.ints) and np.array_equal(b0.tlog, b1.tlog) and np.array_equal(b0.gold, b1.gold)
    assert b1.sem_valid.sum() < b0.sem_valid.sum()
    assert np.all(b1.sem_t[~b1.sem_valid] == 0) and np.array_equal(b1.sem_valid, b0.sem_valid & b1.sem_valid)
    sub = s1.subset(np.arange(10))
    assert sub.allow_masked
    sub.check_against(sh.subset(np.arange(10)))


def test_exclude_truncated_cli_uses_existing_base_without_provider(tmp_path):
    conn, rows, srcs, shard, spm_path, emb_root = make_trunc_fixture(tmp_path)
    import shutil
    shutil.copy(spm_path, os.path.join(shard, "spm.model"))
    base = emb_root / "toyA"
    before = {f: (base / f).stat().st_mtime_ns for f in os.listdir(base)}
    # provider 名は実在の preset でなければ argparse が拒否するため、base を preset 名の下へ置く
    root = tmp_path / "emb2"
    os.makedirs(root / "qwen3_emb_0p6b")
    shutil.copytree(base, root / "qwen3_emb_0p6b" / "toyA")
    rc = EM.main(["extract", "--provider", "qwen3_emb_0p6b", "--shard", shard, "--db", str(tmp_path / "r.sqlite"), "--lp", "8",
                  "--out-root", str(root), "--exclude-truncated"])   # 再抽出はしない（provider へ接続しない）
    assert rc == 0
    out = root / "qwen3_emb_0p6b" / "toyA_notrunc"
    assert (out / "cand_idx_train.npy").exists() and json.loads((out / "meta.json").read_text())["exclude_truncated"]["lc"] == LC_TOY
    assert {f: (base / f).stat().st_mtime_ns for f in os.listdir(base)} == before


# ---------------------------------------------------------------------------------------
# data/emb_eval（評価専用 Teacher 埋め込み）を学習コードから読めないようにするガード
# ---------------------------------------------------------------------------------------

def test_emb_eval_dir_is_rejected_by_load_sem_dir_and_train(fake, tmp_path):
    ok, _ = write_sem_dir(fake, tmp_path / "emb" / "toy" / "pubA")
    sh = M.Shard(M.find_shard(fake, "train", 48))
    M.load_sem_dir(ok, sh.item_id)                                     # 通常の data/emb/... は読める
    ev, _ = write_sem_dir(fake, tmp_path / "emb_eval" / "toy" / "pubA")
    with pytest.raises(ValueError, match="評価専用"):
        M.load_sem_dir(ev, sh.item_id)                                 # 内容が学習用と同形式でも、emb_eval 配下は拒否
    link = tmp_path / "innocent_name"
    os.symlink(ev, link)
    with pytest.raises(ValueError, match="評価専用"):
        M.load_sem_dir(str(link), sh.item_id)                          # symlink 経由でも realpath で拒否
    marked, _ = write_sem_dir(fake, tmp_path / "emb" / "toy" / "marked")
    open(os.path.join(marked, "EVAL_ONLY"), "w").write("")
    with pytest.raises(ValueError, match="EVAL_ONLY"):
        M.load_sem_dir(marked, sh.item_id)
    flagged, _ = write_sem_dir(fake, tmp_path / "emb" / "toy" / "flagged")
    mj = os.path.join(flagged, "meta.json")
    meta = json.load(open(mj))
    meta["eval_only"] = True
    json.dump(meta, open(mj, "w"))
    with pytest.raises(ValueError, match="eval_only"):
        M.load_sem_dir(flagged, sh.item_id)
    # train.py: --sem-emb が emb_eval なら（λ>0 でも 0 でも）起動前に拒否
    for lam in ("0.5", "0"):
        with pytest.raises(SystemExit, match="評価専用"):
            run_train(fake, tmp_path / f"rx{lam}", ["--max-steps", "2", "--no-final-eval", "--sem-cand-weight", lam, "--sem-emb", ev])
    assert not list((tmp_path / "rx0.5").glob("**/*.npz"))   # 学習は始まっていない（checkpoint 無し）
