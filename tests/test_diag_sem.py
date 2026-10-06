"""diag_sem（Candidate Semantic Distillation の診断）: 数値部品、Student の候補 hidden が学習時の sem pass と一致すること、
MASSIVE 文字列 -> intent の対応検証、評価専用埋め込みの隔離、end-to-end（fake provider / fake DB / toy spm）。"""
import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill import diag_sem as D  # noqa: E402
from tb250distill import replay  # noqa: E402
from tb250distill.data import public as P  # noqa: E402
from tb250distill.student import model as M  # noqa: E402
from tb250distill.teacher import embed as EM  # noqa: E402


# ---------------------------------------------------------------------------------------
# 数値部品
# ---------------------------------------------------------------------------------------

def test_auc_and_recall_and_topk():
    assert D.auc_pos_neg([3, 4], [1, 2, 3.5]) == pytest.approx(5 / 6)
    assert D.auc_pos_neg([1, 1], [1, 1]) == pytest.approx(0.5)            # 同値は 0.5
    assert D.auc_pos_neg([], [1]) is None
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 40), rng.normal(0, 1, 50)
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert D.auc_pos_neg(pos, neg) == pytest.approx(brute)
    sim = rng.normal(size=(7, 30))
    sim[np.arange(7), np.arange(7)] = -np.inf
    r = D.recall_at_k(sim, sim)
    assert all(v == 1.0 for v in r.values())
    r2 = D.recall_at_k(sim, -np.where(np.isinf(sim), 0, sim) + np.where(np.isinf(sim), -np.inf, 0))   # 順位を反転
    assert r2[1] < 0.5 and r2[10] < 0.5
    top = D.topk_idx(sim, 5)
    for i in range(7):
        assert set(top[i]) == set(np.argsort(-sim[i])[:5])


def test_ridge_crossfit_recovers_linear_map_and_groups_do_not_leak():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 10))
    W = rng.normal(size=(10, 6))
    Y = X @ W + 0.01 * rng.normal(size=(300, 6)) + 3.0
    groups = [i // 5 for i in range(300)]
    pred, model, alpha, scores = D.ridge_crossfit(X, Y, groups, folds=5)
    assert D.cos_rows(pred, Y).mean() > 0.99 and alpha in scores
    fold = D.group_folds(groups, 5)
    for g in set(groups):
        assert len({fold[i] for i in range(300) if groups[i] == g}) == 1      # 同じ group は同じ fold
    assert set(fold.tolist()) == set(range(5))
    assert np.allclose(D.ridge_predict(model, X[:3]), pred[:3] + 0, atol=0.2)  # 全データ fit も cross-fit も同じ写像に近い
    # ノイズだけの target は cross-fit で学習できない（in-sample でなく cross-fit を使っている）
    Yn = rng.normal(size=(300, 6))
    pn, _, _, _ = D.ridge_crossfit(rng.normal(size=(300, 40)), Yn, groups, folds=5)
    assert D.cos_rows(pn, Yn).mean() < 0.3


def make_roles(n_intent=4, per=3):
    """seen: intent × lang(en) × phrase(per) 、unseen: intent × 2。小さな Sets 用。"""
    roles, strings = [], []
    for it in range(n_intent):
        for p in range(per):
            strings.append(f"train-{it}-{p}")
            roles.append({"sets": ["massive_train"], "source": "massive", "in_train": True, "intent": f"i{it}", "lang": "en", "tier": 0,
                          "phrase": p % 2, "wrapper": p})
    for it in range(n_intent):
        for p in range(2):
            strings.append(f"test-{it}-{p}")
            roles.append({"sets": ["massive_test", "massive_val"], "source": "massive", "in_train": False, "intent": f"i{it}", "lang": "en",
                          "tier": 1, "phrase": 0, "wrapper": p})
    for i in range(5):
        strings.append(f"other-{i}")
        roles.append({"sets": ["other_synth"], "source": "synth", "in_train": True, "intent": None, "lang": None, "tier": None,
                      "phrase": None, "wrapper": None})
    return strings, roles


def test_paraphrase_metrics_and_teacher_metrics():
    strings, roles = make_roles()
    sets = D.Sets(roles)
    assert len(sets.seen_massive) == 12 and len(sets.unseen) == 8 and len(sets.other["synth"]) == 5 and sets.n_unseen_in_train == 0
    n = len(strings)
    # intent ごとの one-hot（+ 小ノイズ）: 完全に intent を分離する空間
    rng = np.random.default_rng(0)
    X = np.zeros((n, 6))
    for i, r in enumerate(roles):
        if r["intent"] is not None:
            X[i, int(r["intent"][1:])] = 1.0
    X += 0.01 * rng.normal(size=X.shape)
    pm = D.paraphrase_metrics(X, sets)
    assert pm["intent_acc"] == 1.0 and pm["intent_chance"] == pytest.approx(1 / 4) and pm["auc_samelang"] == 1.0
    assert pm["margin"] > 0.9 and pm["cos_diff_intent"] < 0.1
    R = rng.normal(size=(n, 6))                        # intent と無関係な空間
    pr = D.paraphrase_metrics(R, sets)
    assert pr["margin_samelang"] < 0.5 and 0.2 < pr["auc_samelang"] < 0.8
    tm = D.teacher_cosine_metrics(X, X, sets)
    assert tm["seen_massive"]["cos"] == pytest.approx(1.0, abs=1e-6) and tm["unseen"]["cos"] == pytest.approx(1.0, abs=1e-6)
    assert tm["seen_massive"]["chance"] < 0.5
    assert tm["unseen"]["specificity"] > 0.9 and abs(tm["unseen"]["cos_other_intent_samelang"]) < 0.1   # 自分の Teacher 埋め込みにだけ近い
    sp_ = D.seen_paraphrase_metrics(X, sets)
    assert sp_["intent_acc"] == 1.0 and sp_["n"] == 12 and 0 < sp_["intent_chance"] < 0.5    # 兄弟（同 phrase）を除いても同 intent の別 phrase が残る
    assert D.seen_paraphrase_metrics(R, sets)["intent_acc"] < 0.9
    # 共通成分だけの空間（全文字列が同じ方向 + 小ノイズ）: cos は高いが specificity は低い
    C = np.tile(rng.normal(size=(1, 6)), (n, 1)) * 5 + 0.1 * rng.normal(size=(n, 6))
    tc_ = D.teacher_cosine_metrics(C, C, sets)
    assert tc_["unseen"]["cos"] > 0.99 and tc_["unseen"]["specificity"] < 0.01
    nb = D.neighbor_recall(X, X, sets)
    assert nb["recall"]["@1"] == 1.0 and nb["pool"] == 20
    v = D.verdict(D.space_metrics(X, X, sets, True), D.paraphrase_metrics(X, sets))
    assert v["ratio_to_teacher"] == 1.0 and v["cond_intent_ge_70pct_teacher"] and v["cond_cos_unseen_ge_80pct_seen"]
    assert v["x_chance"] == pytest.approx(4.0) and not v["cond_intent_ge_5x_chance"] and not v["close_enough"]   # 4 intent しかない toy では chance の 5 倍に届かない
    vr = D.verdict(D.space_metrics(R, X, sets, True), D.paraphrase_metrics(X, sets))
    assert not vr["close_enough"]


# ---------------------------------------------------------------------------------------
# Student の候補 hidden = 学習時の sem pass（np backend）と一致
# ---------------------------------------------------------------------------------------

def test_candidate_hidden_matches_training_sem_pass():
    cfg = M.Config(vocab=60, emb=8, hidden=12, layers=2, lp=8, lc=5, sem_dim=4)
    params = M.init_params(cfg, 0)
    rng = np.random.default_rng(0)
    B, K, Lc = 4, 3, 5
    cand = rng.integers(5, 60, size=(B, K, Lc)).astype(np.int32)
    clen = rng.integers(1, Lc + 1, size=(B, K)).astype(np.int32)
    for b in range(B):
        for j in range(K):
            cand[b, j, clen[b, j]:] = 0
    sh = M.Shard(dict(item_id=np.arange(B), prefix=rng.integers(5, 60, size=(B, 8)).astype(np.int32), prefix_len=np.full(B, 6, np.int32),
                      cand=cand, cand_len=clen, k=np.full(B, K, np.int32), t_logits=rng.normal(size=(B, K)).astype(np.float32),
                      gold=np.zeros(B, np.int32)))
    emb = rng.normal(size=(B * K, 4)).astype(np.float32)
    sem = M.SemTargets(np.arange(B * K).reshape(B, K), emb)
    sem.check_against(sh)
    bt = M.make_batch(sh, np.arange(B), sem=sem)
    be = M.make_backend("np")
    st = M.Student(be, cfg, B, K, train=True, max_lp=8, max_lc=Lc)
    st.set_params(params)
    st.sem_lambda = 1.0
    ref = st.loss_grads(bt, want_grads=False)["sem"]
    H = D.candidate_hidden(params, cfg, cand.reshape(B * K, Lc), clen.reshape(-1))
    Z = D.head_project(params, H)
    mine = float((1.0 - D.cos_rows(Z, emb)).mean())
    assert mine == pytest.approx(ref, abs=2e-4)
    # 文脈なし = 候補の hidden は prefix に依存しない。長さ 0 の pad は初期 hidden（0）のまま
    H0 = D.candidate_hidden(params, cfg, np.zeros((2, Lc), np.int32), np.zeros(2, np.int32))
    assert np.array_equal(H0, np.zeros_like(H0))
    # バッチ分割に依らない
    Hb = D.candidate_hidden(params, cfg, cand.reshape(B * K, Lc), clen.reshape(-1), batch=5)
    assert np.allclose(H, Hb, atol=1e-6)


# ---------------------------------------------------------------------------------------
# MASSIVE: 文字列 -> intent
# ---------------------------------------------------------------------------------------

def massive_rows(n_train=120, n_val=20, n_test=40):
    rows = []
    intents = P.MASSIVE_INTENTS
    iid = 0
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for i in range(n):
            intent = intents[(i * 7 + iid) % len(intents)]
            loc = "ja-JP" if i % 2 else "en-US"
            it = P.conv_massive({"id": f"{split}{i}", "locale": loc, "partition": split, "scenario": P._scenario_of(intent),
                                 "intent": intent, "utt": f"utterance {split} {i}"}, split, split)
            it["item_id"] = iid
            iid += 1
            rows.append(it)
    return rows


def test_massive_string_table_matches_conv_massive(tmp_path):
    tbl = D.massive_string_table()
    assert all(len(v) == 1 for v in tbl.values()) and len(tbl) == 60 * 2 * (2 * 5 + 1 * 2)   # 全文字列が一意に intent へ引ける
    conn = replay.connect(str(tmp_path / "r.sqlite"))
    rows = massive_rows()
    replay.insert_items(conn, rows)
    v = D.verify_massive_table(conn, tbl)
    assert v["n_mismatches"] == 0 and v["items"] == len(rows) and v["ambiguous_candidates"] == 0
    # 壊れた DB（別 tier の文字列に差し替え）は検出される
    bad = dict(rows[0], item_id=9999, candidates=[next(s for s, e in tbl.items() if e[0]["tier"] == 1 and e[0]["lang"] == rows[0]["lang"])] + rows[0]["candidates"][1:])
    replay.insert_items(conn, [bad])
    assert D.verify_massive_table(conn, tbl)["n_mismatches"] > 0
    # 言い回し表の train 用と評価用は重ならない（unseen の前提）
    tr = {s for s, e in tbl.items() if e[0]["tier"] == 0}
    ev = {s for s, e in tbl.items() if e[0]["tier"] == 1}
    assert not (tr & ev)


# ---------------------------------------------------------------------------------------
# end-to-end（fake provider / fake DB / toy spm）
# ---------------------------------------------------------------------------------------

class FakeProvider(EM.EmbeddingProvider):
    name = "fake"

    def __init__(self, D_=24):
        self.D = D_
        self.seen = []

    def embed(self, texts):
        import hashlib
        self.seen += list(texts)
        out = []
        for t in texts:
            r = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:12], 16))
            out.append(r.normal(size=self.D))
        return np.array(out, np.float32)


def make_spm(tmp_path, strings):
    import sentencepiece as spm
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(strings))
    pre = str(tmp_path / "spm_toy")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=pre, vocab_size=1200, model_type="unigram", byte_fallback=True,
                                   character_coverage=1.0, pad_id=0, unk_id=1, bos_id=2, eos_id=3, user_defined_symbols=["<sep>"],
                                   minloglevel=2)
    return pre + ".model"


@pytest.fixture()
def world(tmp_path):
    conn = replay.connect(str(tmp_path / "r.sqlite"))
    rows = massive_rows(n_train=240, n_val=30, n_test=60)
    nxt = len(rows)
    for i in range(80):
        rows.append(dict(item_id=nxt + i, split="train", category="nli", lang="en", context="c", question="q",
                         candidates=[f"alpha beta {i % 40}", f"gamma delta {i % 30}"], gold=0, gen_seed=1, source="synth"))
    replay.insert_items(conn, rows)
    shard = tmp_path / "tok" / "toyA"
    os.makedirs(shard)
    tr = conn.execute("SELECT item_id, candidates FROM items WHERE split='train' ORDER BY item_id").fetchall()
    ids = np.array([r[0] for r in tr], np.int64)
    k = np.array([len(json.loads(r[1])) for r in tr], np.int32)
    np.savez(shard / "train_L8.npz", item_id=ids, k=k, cand=np.zeros((len(ids), 5, 8), np.int32))
    EM.extract(FakeProvider(), str(shard), str(tmp_path / "r.sqlite"), str(tmp_path / "emb"), provider_name="fake", lp=8, dim=8,
               log=lambda *_: None)
    allstr = sorted({c for (cj,) in conn.execute("SELECT candidates FROM items") for c in json.loads(cj)})
    spm_path = make_spm(tmp_path, allstr + ["utterance train 1 what is"])
    shutil.copy(spm_path, shard / "spm.model")
    import hashlib
    json.dump({"name": "toyA", "lc": 8, "spm_sha1": hashlib.sha1(open(spm_path, "rb").read()).hexdigest()}, open(shard / "meta.json", "w"))
    return {"tmp": tmp_path, "conn": conn, "shard": str(shard), "db": str(tmp_path / "r.sqlite"), "emb_root": str(tmp_path / "emb"),
            "eval_root": str(tmp_path / "data" / "emb_eval"), "spm": spm_path}


def snapshot(d):
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_prepare_eval_isolation_and_guard(world):
    w = world
    before = snapshot(w["tmp"] / "emb")
    prov = FakeProvider()
    out = D.prepare_eval(w["conn"], w["shard"], ["fake"], {"fake": prov}, w["emb_root"], w["eval_root"], other_per_source=10, seed=0,
                         log=lambda *_: None)
    assert snapshot(w["tmp"] / "emb") == before                         # 学習用 data/emb には何も書かない
    d = out["fake"]
    assert "emb_eval" in d.split(os.sep) and os.path.exists(os.path.join(d, "EVAL_ONLY"))
    strings, roles, T, meta = D.load_eval_dir(d)
    assert meta["eval_only"] and meta["consistency_with_train_embedding"]["cos_mean"] == pytest.approx(1.0, abs=1e-5)
    assert T.shape == (len(strings), 8) and len(roles) == len(strings) and len(set(strings)) == len(strings)
    assert len(prov.seen) == len(strings)
    test_cands = {c for (cj,) in w["conn"].execute("SELECT candidates FROM items WHERE split='test' AND source='massive'") for c in json.loads(cj)}
    sets = D.Sets(roles)
    assert {strings[i] for i in sets.unseen} == test_cands and sets.n_unseen_in_train == 0
    assert all(roles[i]["intent"] and not roles[i]["in_train"] for i in sets.unseen)
    train_strings = set(json.loads((w["tmp"] / "emb" / "fake" / "toyA" / "strings.json").read_text()))
    assert all(strings[i] in train_strings for i in sets.seen_all) and not (test_cands & train_strings)
    assert len(sets.other["synth"]) == 10
    # PCA は学習用のものをそのまま適用（eval 文字列は PCA に影響しない）
    pca = np.load(w["tmp"] / "emb" / "fake" / "toyA" / "pca.npz")
    assert np.allclose(T, EM.apply_pca(np.load(os.path.join(d, "emb_raw.npy")), pca), atol=1e-5)
    # 学習コードは読めない
    sh_ids = np.load(os.path.join(w["shard"], "train_L8.npz"))["item_id"]
    with pytest.raises(ValueError, match="評価専用"):
        M.load_sem_dir(d, sh_ids)
    # 再実行は再抽出しない / eval-root に emb_eval を含まないパスは拒否
    n0 = len(prov.seen)
    D.prepare_eval(w["conn"], w["shard"], ["fake"], {"fake": prov}, w["emb_root"], w["eval_root"], other_per_source=10, seed=0, log=lambda *_: None)
    assert len(prov.seen) == n0
    with pytest.raises(SystemExit, match="emb_eval"):
        D.prepare_eval(w["conn"], w["shard"], ["fake"], {"fake": prov}, w["emb_root"], str(w["tmp"] / "plain"), log=lambda *_: None)


def make_ckpt(path, sem_dim, seed=0, vocab=1200):
    cfg = M.Config(vocab=vocab, emb=16, hidden=24, layers=2, lp=8, lc=8, sem_dim=sem_dim)
    M.save_params_npz(str(path), cfg, M.init_params(cfg, seed), extra={"step": 7})
    return cfg


def write_preds(path, conn, seed=0):
    rng = np.random.default_rng(seed)
    items = D.load_test_items(conn)
    n = len(items)
    probs = rng.dirichlet(np.ones(5), size=n)
    tp = rng.dirichlet(np.ones(5), size=n)
    k = np.array([len(i["candidates"]) for i in items], np.int32)
    for a in (probs, tp):
        a[np.arange(5)[None, :] >= k[:, None]] = 0
        a /= a.sum(1, keepdims=True)
    np.savez(path, test_item_id=np.array([i["item_id"] for i in items]), test_k=k, test_probs=probs, test_t_probs=tp,
             test_gold=np.zeros(n, np.int32), __meta__=np.array("{}"))


def test_end_to_end_run_diag_summary(world):
    w = world
    D.prepare_eval(w["conn"], w["shard"], ["fake"], {"fake": FakeProvider()}, w["emb_root"], w["eval_root"], other_per_source=20, log=lambda *_: None)
    out = w["tmp"] / "diag"
    results = []
    for name, sem_dim in (("sem_toy_a", 8), ("baseA_toy_b", 0)):
        rd = w["tmp"] / "runs" / name / "common" / "g"
        os.makedirs(rd / "ckpt")
        make_ckpt(rd / "ckpt" / "best.npz", sem_dim, seed=1 if sem_dim else 2)
        write_preds(rd / "preds.npz", w["conn"])
        if sem_dim:
            json.dump({"sem": {"info": {"provider": "fake"}}}, open(rd / "config.json", "w"))
        res = D.run_diag(str(rd), D.run_name_from_dir(rd), None, ["fake"], w["eval_root"], w["shard"], w["db"], str(out), w["emb_root"],
                         log=lambda *_: None)
        results.append(res)
    a, b = results
    pa, pb = a["providers"]["fake"], b["providers"]["fake"]
    assert set(pa["spaces"]) == {"head", "probe", "h"} and set(pb["spaces"]) == {"probe", "h"}      # head が無い run に head 行は出ない
    assert pa["primary_space"] == "head" and pb["primary_space"] == "probe" and pa["has_head"] and not pb["has_head"]
    assert pa["sets"]["unseen"] == len(D.Sets(D.load_eval_dir(pa["eval_dir"])[1]).unseen) > 0 and pa["sets"]["unseen_also_in_train"] == 0
    for p in (pa, pb):
        for sp, m in p["spaces"].items():
            assert 0 <= m["neighbors"]["recall"]["@10"] <= 1 and 0 <= m["paraphrase"]["intent_acc"] <= 1
            if sp != "h":
                assert -1 <= m["teacher_cos"]["unseen"]["cos"] <= 1 and "seen_massive" in m["teacher_cos"] and "seen_synth" in m["teacher_cos"]
        assert isinstance(p["verdict_primary"]["close_enough"], bool)
        jr = p["judgement_relation"]["teacher_top1"]
        assert jr["n_items"] == 60 and 0 <= jr["rep_acc"] <= 1 and sum(jr["table"].values()) == 60
        assert len(p["examples"]) <= 4 and all(e["spaces"]["teacher"]["nearest"] for e in p["examples"])
    assert "judgement_relation_probe" in pa and "judgement_relation_probe" not in pb
    # JSON に書かれ、再実行で同じ結果（決定的）
    names = sorted(p.name for p in out.glob("*.json"))
    assert len(names) == 2
    j = json.loads((out / names[0]).read_text())
    again = D.run_diag(str(w["tmp"] / "runs" / "sem_toy_a" / "common" / "g"), "sem_toy_a_g", None, ["fake"], w["eval_root"], w["shard"], w["db"],
                       str(out), w["emb_root"], log=lambda *_: None)
    assert again["providers"]["fake"]["spaces"] == pa["spaces"]
    md = D.render_summary([json.loads(p.read_text()) for p in sorted(out.glob("*.json"))])
    assert "Teacher = fake" in md and "sem_toy_a" in md and "baseA_toy_b" in md and "判定" in md and "head" in md
    # CLI の summary
    assert D.main(["summary", "--out", str(out), "--docs", str(w["tmp"] / "DIAG.md")]) == 0
    assert (out / "SUMMARY.md").read_text() == (w["tmp"] / "DIAG.md").read_text()


def test_sem_run_uses_only_its_training_provider(world):
    rd = world["tmp"] / "r"
    os.makedirs(rd)
    assert D.run_providers(str(rd), None, ["a", "b"]) == ["a", "b"]
    json.dump({"sem": {"info": {"provider": "b"}}}, open(rd / "config.json", "w"))
    assert D.run_providers(str(rd), None, ["a", "b"]) == ["b"]
    assert D.run_name_from_dir("runs/sem06/common/gt710") == "sem06_gt710"
