"""Wikipedia LM 事前学習: np float64 数値勾配 / chunk 不変 / resume 厳密一致 / export → train.py --init / wiki.py 前処理。
cl テスト（np/cl 一致）は pyopencl + デバイスがあるときだけ（TB250_CL_DEVICE、既定 "WX 2100"）。"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill.student.backend_np import NPBackend  # noqa: E402
from tb250distill.student import model as M, pretrain_lm as P, train as T, fake_data as F  # noqa: E402
from tb250distill.data import wiki  # noqa: E402

DEVICE = os.environ.get("TB250_CL_DEVICE", "WX 2100")


def tiny_cfg(layers=2, vocab=40, emb=5, hidden=6):
    return M.Config(vocab=vocab, emb=emb, hidden=hidden, layers=layers, lp=8, lc=5)


def tiny_rows(n=4, T1=9, vocab=40, seed=0):
    return np.random.default_rng(seed).integers(0, vocab, size=(n, T1)).astype(np.uint16)


def build(cfg, B=4, T=8, chunk=7, dtype=np.float64, seed=1, scale=1.5):
    be = NPBackend(dtype)
    lm = P.LMModel(be, cfg, B, T, chunk_rows=chunk)
    params = {k: (v * scale).astype(dtype) for k, v in P.init_lm_params(cfg, seed).items()}
    params["lm.b"] = np.random.default_rng(5).normal(size=params["lm.b"].shape).astype(dtype) * 0.3
    lm.set_params(params)
    return be, lm, params


# ---------------------------------------------------------------------------------------
# op / 勾配
# ---------------------------------------------------------------------------------------

def test_softmax_ce_op():
    rng = np.random.default_rng(0)
    N, V = 9, 17
    x = rng.normal(size=(N, V)) * 3
    tgt = rng.integers(0, V, N).astype(np.int32)
    be = NPBackend(np.float64)
    xa = x.copy()
    losses = np.zeros(N)
    be.softmax_ce(xa, tgt, losses, N, V, 0.25)
    lse = np.log(np.exp(x - x.max(1, keepdims=True)).sum(1)) + x.max(1)
    assert np.allclose(losses, lse - x[np.arange(N), tgt])
    p = np.exp(x - lse[:, None])
    p[np.arange(N), tgt] -= 1
    assert np.allclose(xa, p * 0.25)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_numeric_gradient(layers):
    cfg = tiny_cfg(layers)
    be, lm, params = build(cfg)
    rows = tiny_rows()
    lm.loss_grads(rows)
    grads = lm.get_grads()
    rng = np.random.default_rng(7)
    eps = 1e-6
    for name, shape in P.lm_param_specs(cfg):
        g = grads[name]
        flat_idx = np.arange(int(np.prod(shape)))
        if name == "emb":
            used = np.unique(rows[:, :-1])
            rws = np.concatenate([rng.choice(used, 6, replace=False), [int(np.setdiff1d(np.arange(40), used)[0])]])
            flat_idx = np.concatenate([r * shape[1] + np.arange(shape[1]) for r in rws])
        elif len(flat_idx) > 40:
            flat_idx = rng.choice(flat_idx, 40, replace=False)
        for fi in flat_idx:
            p = {k: v.copy() for k, v in params.items()}
            p[name].reshape(-1)[fi] += eps
            lm.set_params(p)
            lp_ = lm.loss_grads(rows, want_grads=False)["loss"]
            p[name].reshape(-1)[fi] -= 2 * eps
            lm.set_params(p)
            lm_ = lm.loss_grads(rows, want_grads=False)["loss"]
            num = (lp_ - lm_) / (2 * eps)
            err = abs(num - g.reshape(-1)[fi]) / max(1e-3, np.abs(g).max())
            assert err < 1e-6, (name, fi, num, g.reshape(-1)[fi])
    lm.set_params(params)


def test_chunk_invariance_and_partial_batch():
    cfg = tiny_cfg(2)
    rows = tiny_rows(4)
    res = []
    for chunk in (3, 7, 1000):
        be, lm, _ = build(cfg, chunk=chunk)
        st = lm.loss_grads(rows)
        res.append((st["loss"], lm.get_grads()))
    for loss, gr in res[1:]:
        assert abs(loss - res[0][0]) < 1e-12
        for k in gr:
            assert np.allclose(gr[k], res[0][1][k], atol=1e-12), k
    # 部分バッチ（b < B）: 同じ行を B=4 の model で b=2 として流した損失 = B=2 の model の損失
    be, lm4, params = build(cfg, B=4)
    be2 = NPBackend(np.float64)
    lm2 = P.LMModel(be2, cfg, 2, 8, chunk_rows=5)
    lm2.set_params(params)
    assert abs(lm4.loss_grads(rows[:2], want_grads=False)["loss"] - lm2.loss_grads(rows[:2], want_grads=False)["loss"]) < 1e-12
    # eval_loss はトークン重み付き平均
    full = lm4.eval_loss(rows, 3)
    parts = [lm4.loss_grads(rows[i:i + 3], want_grads=False) for i in (0, 3)]
    assert abs(full - sum(p["loss"] * p["ntok"] for p in parts) / sum(p["ntok"] for p in parts)) < 1e-12


def test_lr_schedule():
    assert P.lr_at(1, 100, 1.0, 10, 0.1) == pytest.approx(0.1)
    assert P.lr_at(10, 100, 1.0, 10, 0.1) == pytest.approx(1.0)
    assert P.lr_at(100, 100, 1.0, 10, 0.1) == pytest.approx(0.1)
    assert P.lr_at(5, 100, 1.0, 0, 0.1) < 1.0


# ---------------------------------------------------------------------------------------
# 学習 CLI / resume / export
# ---------------------------------------------------------------------------------------

CFG = '{"vocab": 300, "emb": 16, "hidden": 24, "layers": 2}'


@pytest.fixture(scope="module")
def wiki_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("wiki")
    rng = np.random.default_rng(0)
    # 学習可能な構造: 次トークンが現トークンから決まる 1 次マルコフ連鎖（低エントロピー）
    nxt = rng.integers(5, 300, size=300)
    def gen(n_rows, T1=17):
        out = np.zeros((n_rows, T1), np.uint16)
        for i in range(n_rows):
            t = int(rng.integers(5, 300))
            for j in range(T1):
                out[i, j] = t
                t = int(nxt[t]) if rng.random() < 0.9 else int(rng.integers(5, 300))
        return out
    np.save(d / "train_00000.npy", gen(400))
    np.save(d / "val_00000.npy", gen(40))
    (d / "meta.json").write_text(json.dumps({"name": "tiny"}))
    return str(d)


def run_lm(data, run, extra=()):
    argv = ["--backend", "np", "--data", data, "--run-dir", str(run), "--config", CFG, "--batch-size", "16", "--seed", "3",
            "--log-every", "5", "--eval-every", "20", "--ckpt-every", "5", "--lr", "5e-3", "--warmup", "3",
            "--chunk-rows", "50", *extra]
    return P.main(argv)


def test_train_reduces_ppl_and_artifacts(wiki_dir, tmp_path):
    run = tmp_path / "run"
    s = run_lm(wiki_dir, run, ["--epochs", "8"])   # 25 steps/epoch * 8
    rows = [l.split(",") for l in open(run / "metrics.csv").read().strip().splitlines()]
    hdr = rows[0]
    assert hdr == P.CSV_COLS
    vp = [float(r[hdr.index("val_ppl")]) for r in rows[1:] if r[hdr.index("val_ppl")]]
    assert vp[0] > 250 and vp[-1] < 0.5 * vp[0], vp          # 初期は ~vocab、学習で大きく下がる
    for f in ("config.json", "hardware.json", "environment.txt", "metrics.csv", "train.log", "summary.json"):
        assert (run / f).exists(), f
    for pth in ("p025", "p050", "p075", "p100", "best", "last"):
        assert (run / "ckpt" / f"{pth}.npz").exists(), pth
    assert s["steps_done"] == 200


def test_resume_is_exact(wiki_dir, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_lm(wiki_dir, a, ["--max-steps", "30", "--epochs", "3"])
    run_lm(wiki_dir, b, ["--max-steps", "30", "--epochs", "3", "--stop-after", "12"])
    run_lm(wiki_dir, b, ["--max-steps", "30", "--epochs", "3", "--resume"])
    ca, pa, oa, ma = P.load_lm_ckpt(str(a / "ckpt" / "last.npz"))
    cb, pb, ob, mb = P.load_lm_ckpt(str(b / "ckpt" / "last.npz"))
    assert ma["step"] == mb["step"] == 30
    for k in pa:
        assert np.array_equal(pa[k], pb[k]), k
    assert np.array_equal(oa[0], ob[0]) and np.array_equal(oa[1], ob[1])


def test_export_and_student_init(wiki_dir, tmp_path):
    run = tmp_path / "run"
    run_lm(wiki_dir, run, ["--max-steps", "10", "--epochs", "1"])
    cfg = M.get_config(CFG)
    base = tmp_path / "base.npz"
    M.save_params_npz(str(base), cfg, M.init_params(cfg, 0), {"seed": 0})
    out = tmp_path / "init.npz"
    info = P.main(["export", "--ckpt", str(run / "ckpt" / "last.npz"), "--base-init", str(base), "--out", str(out)])
    assert info == 0
    cfg2, p2, meta = M.load_params_npz(str(out))
    _, plm, _, _ = P.load_lm_ckpt(str(run / "ckpt" / "last.npz"))
    _, pb, _ = M.load_params_npz(str(base))
    for k in ("emb", "l0.wi", "l1.wh", "l1.bh"):
        assert np.array_equal(p2[k], plm[k]) and not np.array_equal(p2[k], pb[k]), k
    for k in ("head.w1", "head.b1", "head.w2", "head.b2"):
        assert np.array_equal(p2[k], pb[k]), k
    assert set(p2) == {n for n, _ in M.param_specs(cfg2)} and meta["source"] == "wikipedia LM pretrain"
    # train.py --init で読める（共通契約）
    d = tmp_path / "fake"
    F.write_fake_dataset(str(d), n_train=300, n_val=40, n_test=40, n_robust=10, vocab=300, lp=24, lc=6, kmax=4, seed=1, active=20)
    s = T.main(["--backend", "np", "--data", str(d), "--run-dir", str(tmp_path / "st"), "--config", CFG, "--init", str(out),
                "--batch-size", "16", "--max-steps", "3", "--no-final-eval", "--eval-every", "3"])
    assert s["steps_done"] == 3
    z = T.load_ckpt(str(tmp_path / "st" / "ckpt" / "p000.npz"))   # step 0 の重み = init
    assert np.array_equal(z[1]["emb"], p2["emb"]) and np.array_equal(z[1]["head.w1"], pb["head.w1"])


# ---------------------------------------------------------------------------------------
# wiki.py（pyarrow 不要な部分）
# ---------------------------------------------------------------------------------------

def test_clean_paragraphs():
    text = ("アンパサンド（&, ）は、記号である。\n\n語源 \n\n英語で教育を行う学校でアルファベットを復唱する場合、その文字自体が単語となる。\n"
            "* 1 2 3 4 5\nReferences: Smith 2001 Journal of Foo 12(3) 45-67 and more text here to be long enough\n"
            "短い見出し\n田中（、）は人名で、 多空白  です。")
    p = wiki.clean_paragraphs(text)
    assert p[0] == "アンパサンド（&, ）は、記号である。"
    assert "語源" not in " ".join(p) and "References" not in " ".join(p) and "短い見出し" not in p
    assert p[-1] == "田中は人名で、 多空白 です。"


def test_val_split_and_order_deterministic():
    ids = [str(i) for i in range(5000)]
    v = [wiki.is_val_article(i, 1) for i in ids]
    assert 20 < sum(v) < 90 and v == [wiki.is_val_article(i, 1) for i in ids]
    assert wiki.order_key("12") == wiki.order_key("12") != wiki.order_key("13")


class FakeSP:
    def encode(self, paras):
        return [[5 + (ord(c) % 100) for c in p] for p in paras]


def test_article_tokens_cap():
    ids, tr, used = wiki.article_tokens(FakeSP(), ["あ" * 10, "い" * 10, "う" * 10], 25)
    assert len(ids) == 20 and tr and used == 2
    ids, tr, used = wiki.article_tokens(FakeSP(), ["あ" * 40, "い" * 10], 25)
    assert len(ids) == 25 and tr and used == 1
    ids, tr, used = wiki.article_tokens(FakeSP(), ["あ" * 5, "い" * 5], 25)
    assert len(ids) == 10 and not tr and used == 2


def test_row_writer(tmp_path):
    w = wiki.RowWriter(tmp_path, "train", T=8, shard_rows=3)
    rng = np.random.default_rng(0)
    allt = []
    for _ in range(11):
        n = int(rng.integers(5, 30))
        ids = rng.integers(0, 8192, n).tolist()
        allt += ids
        w.add(ids)
    w.close()
    rows = wiki.load_rows(str(tmp_path), "train", mmap=False)
    got = np.concatenate(rows).reshape(-1)
    assert w.tokens == got.size == (len(allt) // 8) * 8 and w.shards == len(rows)
    assert np.array_equal(got, np.array(allt[:got.size], np.uint16))
    assert all(r.shape[1] == 8 and r.shape[0] <= 3 for r in rows)


def test_build_end_to_end(tmp_path):
    pa = pytest.importorskip("pyarrow")
    spm = pytest.importorskip("sentencepiece")
    import pyarrow.parquet as pq
    src = tmp_path / "src"
    src.mkdir()
    arts = []
    for i in range(60):
        body = "\n".join(f"これは記事{i}の第{j}段落です。日本語の文章がここに入ります。とても長い文章を書いて十分な長さにします。" for j in range(6))
        arts.append({"id": str(i), "title": f"t{i}", "text": f"記事{i}\n\n{body}\n\n参考文献\n"})
    pq.write_table(pa.Table.from_pylist(arts), src / "subset-00000.parquet")
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n".join(a["text"] for a in arts), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "spm"), vocab_size=700, model_type="unigram",
                                   byte_fallback=True, pad_id=0, unk_id=1, bos_id=2, eos_id=3, user_defined_symbols=["<sep>"],
                                   hard_vocab_limit=False)
    out = tmp_path / "out"
    vs = spm.SentencePieceProcessor(model_file=str(tmp_path / "spm.model")).get_piece_size()
    rc = wiki.main(["build", "--src", str(src), "--out", str(out), "--spm", str(tmp_path / "spm.model"), "--vocab", str(vs),
                    "--seq-len", "16", "--target-tokens", "2000", "--val-pct", "10", "--shard-rows", "50"])
    assert rc == 0
    meta = json.load(open(out / "meta.json"))
    assert meta["train"]["tokens"] >= 1900 and meta["val"]["rows"] > 0 and meta["unk_token_frac"] == 0.0
    tr = np.concatenate(wiki.load_rows(str(out), "train", mmap=False))
    assert tr.shape[1] == 16 and tr.max() < vs and (tr == 2).sum() > 0 and (tr == 3).sum() > 0   # bos / eos が入る


# ---------------------------------------------------------------------------------------
# cl（np/cl 一致）
# ---------------------------------------------------------------------------------------

def rel(a, b):
    return float(np.linalg.norm(a - b) / max(1e-30, np.linalg.norm(b)))


@pytest.fixture(scope="module")
def cl_be():
    pytest.importorskip("pyopencl")
    pytest.importorskip("pyclblast")
    from tb250distill.student.backend_cl import CLBackend
    try:
        return CLBackend(DEVICE)
    except RuntimeError as e:
        pytest.skip(str(e))


def test_cl_softmax_ce_vs_np(cl_be):
    rng = np.random.default_rng(1)
    for (N, V) in [(5, 17), (37, 300), (64, 8192)]:
        x = (rng.normal(size=(N, V)) * 3).astype(np.float32)
        tgt = rng.integers(0, V, N).astype(np.int32)
        tgt[0] = V - 1
        xd = cl_be.alloc((N, V))
        cl_be.upload(xd, x)
        td = cl_be.alloc((N,), "i")
        cl_be.upload(td, tgt)
        ld = cl_be.alloc((N,))
        cl_be.softmax_ce(xd, td, ld, N, V, 1.0 / N)
        xn = x.astype(np.float64)
        ln = np.zeros(N)
        NPBackend(np.float64).softmax_ce(xn, tgt, ln, N, V, 1.0 / N)
        assert np.abs(cl_be.download(ld) - ln).max() < 1e-4 * max(1.0, np.abs(ln).max())
        got = cl_be.download(xd)
        assert np.abs(got - xn).max() < 1e-5, np.abs(got - xn).max()
    # 部分 view（offset 付き）
    N, V = 6, 50
    x = rng.normal(size=(N, V)).astype(np.float32)
    tgt = rng.integers(0, V, N + 3).astype(np.int32)
    big = cl_be.alloc((N * V + 7,))
    cl_be.upload(big, np.concatenate([np.zeros(7, np.float32), x.ravel()]))
    tb = cl_be.alloc((N + 3,), "i")
    cl_be.upload(tb, tgt)
    lb = cl_be.alloc((N + 2,))
    cl_be.softmax_ce(cl_be.view(big, 7, (N, V)), cl_be.view(tb, 3, (N,)), cl_be.view(lb, 2, (N,)), N, V, 1.0)
    xn = x.astype(np.float64)
    ln = np.zeros(N)
    NPBackend(np.float64).softmax_ce(xn, tgt[3:], ln, N, V, 1.0)
    assert np.abs(cl_be.download(lb)[2:] - ln).max() < 1e-4
    assert np.abs(cl_be.download(big)[7:] - xn.ravel()).max() < 1e-5


def test_cl_lm_step_vs_np(cl_be):
    cfg = M.Config(vocab=300, emb=16, hidden=24, layers=2, lp=12, lc=5)
    rows = np.random.default_rng(3).integers(0, 300, size=(8, 13)).astype(np.uint16)
    params = P.init_lm_params(cfg, 2)
    params["lm.b"] = np.random.default_rng(5).normal(size=300).astype(np.float32) * 0.1
    lm_np = P.LMModel(NPBackend(np.float64), cfg, 8, 12, chunk_rows=50)
    lm_np.set_params({k: v.astype(np.float64) for k, v in params.items()})
    lm_cl = P.LMModel(cl_be, cfg, 8, 12, chunk_rows=50)
    lm_cl.set_params(params)
    a = lm_np.loss_grads(rows)
    b = lm_cl.loss_grads(rows)
    assert abs(a["loss"] - b["loss"]) < 1e-4
    gn, gc = lm_np.get_grads(), lm_cl.get_grads()
    tot = np.sqrt(sum(float((gn[k] ** 2).sum()) for k in gn))
    err = np.sqrt(sum(float(((gn[k] - gc[k]) ** 2).sum()) for k in gn))
    assert err / tot < 2e-3, err / tot
    for k in gn:
        assert rel(gc[k], gn[k]) < 5e-3, (k, rel(gc[k], gn[k]))
    # 1 step 後（AdamW + clip）
    lm_np.train_step(rows, 1e-3, 1)
    lm_cl.train_step(rows, 1e-3, 1)
    pn, pc = lm_np.get_params(), lm_cl.get_params()
    gnorm_mask = {k: np.abs(gn[k]) > 1e-5 for k in gn}
    for k in pn:
        m = gnorm_mask[k]
        assert np.abs(pc[k][m] - pn[k][m]).max() < 2e-5 if m.any() else True, k
