"""tb250distill.data.public（公開データ → items 変換）と、source/extra 対応（replay・produce・tokenize・evaluate）のテスト。
小さな fixture だけを使い、ネットワーク・pyarrow は不要。"""
import json
import sqlite3

import numpy as np
import pytest

from tb250distill import replay
from tb250distill.data import public as P
from tb250distill.teacher import produce


# --------------------------------------------------------------------------- fixtures

def jcqa_row(i=1, label=2):
    return {"q_id": i, "question": "電子機器で使用される最も主要な電子回路基板の事をなんと言う？",
            "choice0": "掲示板", "choice1": "パソコン", "choice2": "マザーボード", "choice3": "ハードディスク", "choice4": "まな板", "label": label}


def jnli_row(i="1", label="entailment"):
    return {"sentence_pair_id": i, "yjcaptions_id": "10125-1", "sentence1": "時計がついている場所に看板が設置されています。",
            "sentence2": "屋根の上に看板があり時計もついています。", "label": label}


def massive_row(i="1", intent="alarm_set", locale="ja-JP"):
    return {"id": i, "locale": locale, "partition": "train", "scenario": "alarm", "intent": intent,
            "utt": "金曜日の午前九時に起こしてください" if locale.startswith("ja") else "wake me up at nine on friday"}


def wrime_row(idx=0, writer=(0, 1, 0, 0, 0, 2, 1, 0), readers=((0, 0, 0, 0, 0, 1, 0, 0), (0, 1, 0, 0, 0, 2, 0, 0), (0, 2, 0, 0, 0, 1, 0, 0)),
              wsent=0, rsents=(-2, -1, -1), text="表情筋が衰えてきてる。まずいな…"):
    r = {"_idx": idx, "Sentence": text, "UserID": "1", "Train/Dev/Test": "train", "Writer_Sentiment": str(wsent)}
    for c, v in zip(P.EMOTION_COLS, writer):
        r[f"Writer_{c}"] = str(v)
    for j, (rv, rs) in enumerate(zip(readers, rsents), start=1):
        for c, v in zip(P.EMOTION_COLS, rv):
            r[f"Reader{j}_{c}"] = str(v)
        r[f"Reader{j}_Sentiment"] = str(rs)
    return r


TOOLS = [json.dumps({"name": "get_weather", "description": "Fetches the weather for a city.",
                     "parameters": {"type": "dict", "properties": {"city": {"type": "str"}, "days": {"type": "int"}}, "required": ["city"]}})]


def w2c_pref_row(chosen="<TOOLCALL>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}]</TOOLCALL>",
                 rejected="Which city do you mean?", user="What is the weather in Tokyo?"):
    return {"tools": TOOLS, "messages": [{"role": "user", "content": user}],
            "chosen_response": {"role": "assistant", "content": chosen}, "rejected_response": {"role": "assistant", "content": rejected}}


def w2c_mcq_row(correct="tool_call", uuid="u1"):
    return {"uuid": uuid, "source": "BFCL", "source_id": "live_0-0-0", "question": "weather in Hanoi please", "correct_answer": correct,
            "answers": {"direct": "It is 28C in Hanoi.", "tool_call": json.dumps({"name": "get_weather", "arguments": {"city": "Hanoi"}}),
                        "request_for_info": "Could you tell me which date?", "cannot_answer": "I'm sorry, I can't check the weather."},
            "tools": TOOLS}


def route_row(score=5, prompt="Write a haiku about spring."):
    return {"prompt": prompt, "source": ["sharegpt"], "gpt4_response": "x", "mixtral_response": "y", "mixtral_score": score}


ALL_PHRASES_JNLI_TRAIN = {p for v in P.JNLI_PHRASES.values() for p in v[0]}
ALL_PHRASES_JNLI_EVAL = {p for v in P.JNLI_PHRASES.values() for p in v[1]}


# --------------------------------------------------------------------------- 変換

def test_jcqa_uses_original_choices_and_gold():
    it = P.conv_jcqa(jcqa_row(), "train", "train")
    assert it["candidates"] == ["掲示板", "パソコン", "マザーボード", "ハードディスク", "まな板"]
    assert it["gold"] == 2 and it["category"] == "commonsense" and it["lang"] == "ja" and it["source"] == "jcqa"
    assert it["context"].startswith("電子機器") and it["question"] in P.Q_JCQA
    assert it["extra"]["orig_id"] == 1 and it["extra"]["label"] == 2
    assert P.conv_jcqa(jcqa_row(), "train", "train") == it  # 決定的
    bad = jcqa_row()
    bad["choice1"] = bad["choice0"]
    assert P.conv_jcqa(bad, "train", "train") is None   # 重複候補は捨てる


def test_jnli_phrases_gold_and_tier_split():
    seen_k = set()
    for i in range(60):
        for split, pool in (("train", ALL_PHRASES_JNLI_TRAIN), ("val", ALL_PHRASES_JNLI_EVAL), ("test", ALL_PHRASES_JNLI_EVAL)):
            lab = ["entailment", "contradiction", "neutral"][i % 3]
            it = P.conv_jnli(jnli_row(str(i), lab), split, "x")
            seen_k.add(len(it["candidates"]))
            assert set(it["candidates"]) <= pool          # train と val/test の言い回しは別集合
            assert it["candidates"][it["gold"]] in P.JNLI_PHRASES[lab][0 if split == "train" else 1]
            assert it["context"] == jnli_row()["sentence1"] and jnli_row()["sentence2"] in it["question"]
            assert it["category"] == "nli" and it["extra"]["label"] == lab
    assert seen_k == {2, 3}
    assert not (ALL_PHRASES_JNLI_TRAIN & ALL_PHRASES_JNLI_EVAL)
    assert P.conv_jnli(jnli_row(label="bogus"), "train", "x") is None


def test_massive_hard_negatives_and_eval_phrases():
    train_strings = set()
    for i, ph in P.MASSIVE_INTENT_PHRASES.items():
        for lang in ("ja", "en"):
            descs, wraps = ph[lang][0], P.MASSIVE_WRAP[lang][0]
            train_strings |= {w.format(d=d) for d in descs for w in wraps}
    ks = set()
    for n in range(80):
        lang = "ja-JP" if n % 2 == 0 else "en-US"
        it = P.conv_massive(massive_row(str(n), "alarm_set", lang), "val", "validation")
        ks.add(len(it["candidates"]))
        assert 3 <= len(it["candidates"]) <= 5 and len(set(it["candidates"])) == len(it["candidates"])
        assert it["gold"] is not None and it["lang"] == ("ja" if n % 2 == 0 else "en")
        assert not (set(it["candidates"]) & train_strings)           # 評価用は学習で見ていない文言
        shown = it["extra"]["intents_shown"]
        assert shown[0] == "alarm_set" and len(shown) == len(it["candidates"])
        assert any(s.startswith("alarm_") for s in shown[1:])        # 同 scenario の hard negative が入る
        assert it["category"] == "intent" and it["extra"]["scenario"] == "alarm"
    assert ks == {3, 4, 5}
    tr = P.conv_massive(massive_row("1", "news_query", "en-US") | {"scenario": "news"}, "train", "train")
    assert len(tr["candidates"]) >= 3                                   # news は同 scenario 無し → 他 scenario から補う


def test_massive_phrase_table_complete():
    assert len(P.MASSIVE_INTENTS) == 60
    for i, ph in P.MASSIVE_INTENT_PHRASES.items():
        for lang in ("ja", "en"):
            assert len(ph[lang][0]) == 2 and len(ph[lang][1]) == 1


def test_wrime_emotion_gold_soft_and_tie():
    # reader 平均の首位 = fear(1.33) > sadness(1.0)、差 >= 0.34 → gold = fear
    seen = 0
    for idx in range(60):
        it = P.conv_wrime(wrime_row(idx), "train", "train")
        e = it["extra"]
        assert abs(sum(e["soft"]) - 1) < 1e-3 and it["category"] == "sentiment"
        assert 2 <= len(it["candidates"]) <= 5 and len(set(it["candidates"])) == len(it["candidates"])
        if e["task"] == "emotion" and e["subject"] == "reader":
            seen += 1
            assert e["label"] == "fear"
            assert it["candidates"][it["gold"]] in P.EMOTION_PHRASES["fear"][0]
            assert "anger" in e["labels_shown"]            # 逆の感情が hard negative に入る
        if e["task"] == "polarity" and e["subject"] == "reader":
            assert e["label"] == "negative" and it["candidates"][it["gold"]] in P.POLARITY_PHRASES["negative"][0]
    assert seen > 0
    # 同点首位（writer: joy=2, trust=2）→ gold 無し、同点の 2 感情が候補に入る
    tie = wrime_row(1, writer=(2, 0, 0, 0, 0, 0, 0, 2))
    found = False
    for idx in range(200):
        tie["_idx"] = idx
        it = P.conv_wrime(tie, "train", "train")
        if it["extra"]["task"] == "emotion" and it["extra"]["subject"] == "writer":
            found = True
            assert it["gold"] is None and {"joy", "trust"} <= set(it["extra"]["labels_shown"])
            break
    assert found
    # 強い感情なし → 'none' が gold
    flat = wrime_row(2, writer=(0,) * 8, readers=((0,) * 8,) * 3, wsent=0, rsents=(0, 0, 0))
    for idx in range(200):
        flat["_idx"] = idx
        it = P.conv_wrime(flat, "val", "dev")
        if it["extra"]["task"] == "emotion":
            assert it["extra"]["label"] == "none" and it["candidates"][it["gold"]] in P.EMOTION_PHRASES["none"][1]
            break


def test_w2c_pref_context_tail_and_gold():
    it = P.conv_w2c_pref(w2c_pref_row(), 7, "train")
    assert it["category"] == "agent_gate" and it["source"] == "when2call" and it["lang"] == "en"
    assert it["context"].startswith("Available tools:\n- get_weather(city*, days)") and it["context"].endswith("User request: What is the weather in Tokyo?")
    assert len(it["candidates"]) == 2 and it["candidates"][it["gold"]].startswith("<TOOLCALL>")
    assert it["extra"]["origin"] == "pref" and sorted(it["extra"]["types_guess"]) == ["request_for_info", "tool_call"]
    assert P.conv_w2c_pref(w2c_pref_row(rejected="x" * 5, chosen="x" * 5), 8, "train") is None   # 同一応答は捨てる


def test_w2c_mcq_all_four_options_and_toolcall_normalized():
    ks = set()
    for n in range(40):
        it = P.conv_w2c_mcq(w2c_mcq_row("request_for_info", f"u{n}"), "val")
        ks.add(len(it["candidates"]))
        assert it["extra"]["types"][it["gold"]] == "request_for_info" and it["candidates"][it["gold"]].startswith("Could you")
        tc = [c for c in it["candidates"] if c.startswith("<TOOLCALL>")]
        assert tc == [] or tc[0].endswith("</TOOLCALL>") and json.loads(tc[0][len("<TOOLCALL>"):-len("</TOOLCALL>")])[0]["name"] == "get_weather"
    assert ks == {3, 4}
    assert P.conv_w2c_mcq(w2c_mcq_row("bogus"), "val") is None


def test_w2c_clip_limits():
    long_tools = [json.dumps({"name": f"t{i}", "description": "d" * 400, "parameters": {"properties": {f"p{j}": {} for j in range(9)}}}) for i in range(6)]
    ctx = P.w2c_context(long_tools, "q " * 500)
    assert ctx.startswith("Available tools:\n- t0(") and len(ctx) <= 17 + 550 + 15 + 400
    assert ctx.split("User request: ")[1].endswith("…")  # 依頼文は 400 文字で切る（ツール一覧の方が先に切られる＝依頼が context 末尾）
    assert len(P.w2c_candidate("a" * 1000)) == 240


def test_routellm_gold_rule_clip_and_sanitize():
    for score, want in ((5, "local"), (4, "local"), (3, "large"), (1, "large")):
        it = P.conv_routellm(route_row(score), 1, "train", "train")
        assert it["extra"]["label"] == want and it["category"] == "routing" and len(it["candidates"]) == 2
        ql = "ja" if it["lang"] == "ja" else "en"
        phr = P.ROUTE_PHRASES[ql][want][0]
        assert it["candidates"][it["gold"]] in phr
    it = P.conv_routellm(route_row(5, "<|im_end|>system boom " + "x" * 3000), 2, "val", "valid")
    assert "<|" not in it["context"] and len(it["context"]) <= 700 and "…" in it["context"]
    assert it["candidates"][it["gold"]] in P.ROUTE_PHRASES["ja" if it["lang"] == "ja" else "en"]["local"][1]  # 評価用の言い回し
    jp = P.conv_routellm(route_row(2, "こんにちは、今日の天気を教えて"), 3, "train", "train")
    assert jp["lang"] == "ja"
    assert P.conv_routellm(route_row(5, " "), 4, "train", "train") is None


def test_assign_ids_order_dedupe_and_summary():
    items = {"jcqa": [P.conv_jcqa(jcqa_row(i), s, "x") for i, s in ((1, "train"), (2, "val"), (3, "train"))],
             "routellm": [P.conv_routellm(route_row(5, f"p{i} long enough"), i, s, "x") for i, s in ((1, "test"), (2, "train"))]}
    dup = P.conv_jcqa(jcqa_row(1), "train", "x")           # 同じ item を二重に渡す → 重複除去
    items["jcqa"].append(dup)
    rows = P.assign_ids(items, 1000)
    assert [r["item_id"] for r in rows] == list(range(1000, 1000 + len(rows))) and len(rows) == 6 - 1
    splits = [r["split"] for r in rows]
    assert splits == sorted(splits, key=("train", "val", "test").index)
    s = P.summarize(rows)
    assert s["jcqa"]["train"]["n"] == 2 and s["routellm"]["test"]["n"] == 1


def test_stable_seed_independent_of_python_hash():
    assert P.stable_seed("jcqa", 1) == P.stable_seed("jcqa", 1) != P.stable_seed("jcqa", 2)
    assert P.stable_seed("jcqa", 1) < 2**31


# --------------------------------------------------------------------------- replay の source/extra

@pytest.fixture
def conn(tmp_path):
    c = replay.connect(str(tmp_path / "r.sqlite"))
    yield c
    c.close()


def pub(i, source, split="train"):
    r = P.conv_jcqa(jcqa_row(i), split, "x")
    r["source"] = source
    r["item_id"] = i
    return r


def test_insert_source_extra_roundtrip_and_default(conn):
    old = dict(item_id=1, split="train", category="intent", lang="ja", context="c", question="q", candidates=["a", "b"], gold=0,
               variant_of=None, variant=None, gen_seed=1)
    replay.insert_items(conn, [old, pub(2, "jcqa")])
    got = {i["item_id"]: i for i in replay.get_items(conn, [1, 2])}
    assert got[1]["source"] == "synth" and got[1]["extra"] is None
    assert got[2]["source"] == "jcqa" and got[2]["extra"]["orig_id"] == 2
    # 旧形式（11 列タプル）も受け付ける
    replay.insert_items(conn, [(3, "train", "intent", "ja", "c", "q", json.dumps(["a", "b"]), 0, None, None, 3)])
    assert replay.get_items(conn, [3])[0]["source"] == "synth"


def test_migration_adds_columns_to_old_db(tmp_path):
    path = str(tmp_path / "old.sqlite")
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE items(item_id INTEGER PRIMARY KEY, split TEXT NOT NULL, category TEXT NOT NULL, lang TEXT NOT NULL, context TEXT NOT NULL,
      question TEXT NOT NULL, candidates TEXT NOT NULL, gold INTEGER, variant_of INTEGER, variant TEXT, gen_seed INTEGER NOT NULL);
    INSERT INTO items VALUES (1,'train','nli','ja','c','q','["a","b"]',0,NULL,NULL,5);
    CREATE TABLE teacher(item_id INTEGER PRIMARY KEY, teacher_model TEXT NOT NULL, method TEXT NOT NULL, logits TEXT NOT NULL,
      probs TEXT NOT NULL, raw TEXT, latency_ms REAL, created_at TEXT NOT NULL);
    """)
    c.commit()
    c.close()
    ro = replay.connect(path, readonly=True)       # 旧 DB を readonly で読んでも壊れない
    assert [r["item_id"] for r in replay.iter_scored(ro)] == []
    ro.close()
    w = replay.connect(path)                        # 通常接続でマイグレーション
    cols = {r[1] for r in w.execute("PRAGMA table_info(items)")}
    assert {"source", "extra"} <= cols
    assert w.execute("SELECT source, extra FROM items WHERE item_id=1").fetchone()[:] == ("synth", None)
    replay.insert_items(w, [pub(2, "wrime")])
    w.close()
    w = replay.connect(path)                        # 二度目は no-op
    assert w.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


def test_pending_and_scored_count_by_source(conn):
    replay.insert_items(conn, [dict(pub(1, "synth"), source="synth"), pub(2, "jcqa"), pub(3, "wrime", "val"), pub(4, "synth", "val")])
    ids = lambda **kw: [i["item_id"] for i in replay.pending_items(conn, 10, ["train", "val"], **kw)]  # noqa: E731
    assert ids(sources=["public"]) == [2, 3]            # splits の優先順（train → val）→ item_id 順
    assert ids(sources=["synth"]) == [1, 4]
    assert ids(sources=["jcqa", "wrime"]) == [2, 3]
    assert ids() == [1, 2, 3, 4]
    lg = [0.0, -1.0, -2.0, -3.0, -4.0]
    replay.write_teacher(conn, 3, "m", "x", lg, _softmax(lg), None)
    assert replay.scored_count(conn, "val", ["public"]) == 1 and replay.scored_count(conn, "val", ["synth"]) == 0
    assert replay.scored_count(conn, None, ["public"]) == 1
    cs = replay.counts_by_source(conn)
    assert cs["wrime"]["val"] == {"total": 1, "scored": 1} and cs["synth"]["train"] == {"total": 1, "scored": 0}
    assert [r["source"] for r in replay.iter_scored(conn, sources=["public"])] == ["wrime"]


def _softmax(xs):
    import math

    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [x / sum(e) for x in e]


def test_parse_stages_with_sources():
    st = produce.parse_stages("val@public,test:200@public,train:50@jcqa+wrime,val,train")
    assert st == [("val", None, ["public"]), ("test", 200, ["public"]), ("train", 50, ["jcqa", "wrime"]), ("val", None, None), ("train", None, None)]
    with pytest.raises(SystemExit):
        produce.parse_stages("train@nope")
    with pytest.raises(SystemExit):
        produce.parse_stages("bogus@public")


def test_backup_db(tmp_path, conn):
    replay.insert_items(conn, [pub(1, "jcqa")])
    conn.commit()
    res = P.backup_db(str(tmp_path / "r.sqlite"), str(tmp_path / "bk.sqlite"))
    assert res["items"] == (1, 1) and res["integrity"] == "ok"
    with pytest.raises(SystemExit):
        P.backup_db(str(tmp_path / "r.sqlite"), str(tmp_path / "bk.sqlite"))   # 上書きしない


# --------------------------------------------------------------------------- tokenize / evaluate の source 対応

def test_shard_has_source_key_and_per_source_stats(tmp_path, conn):
    spm = pytest.importorskip("sentencepiece")
    from tb250distill import tokenize_data as T

    rows = []
    for i in range(1, 41):
        it = pub(i, "jcqa" if i % 2 else "wrime")
        it["context"] = f"電子機器の話 {i} " * 3
        rows.append(it)
    replay.insert_items(conn, rows)
    for i in range(1, 41):
        lg = [0.5 * (j == i % 5) for j in range(5)]
        replay.write_teacher(conn, i, "m", "x", lg, _softmax(lg), None, commit=False)
    conn.commit()
    ro = replay.connect(str(tmp_path / "r.sqlite"), readonly=True)
    out = tmp_path / "tok"
    T.train_spm(ro, out, vocab=400, max_per_source=7)
    sp = spm.SentencePieceProcessor(model_file=str(out / "spm.model"))
    arrays, stats = T.build_shard(ro, sp, "train", lp=32, lc=8, kmax=5, teacher_temp=1.0)
    assert arrays["source"].dtype.kind == "U" and set(arrays["source"].tolist()) == {"jcqa", "wrime"}
    assert stats["n"] == 40 and set(stats["by_source"]) == {"jcqa", "wrime"} and stats["by_source"]["jcqa"]["n"] == 20
    assert 0 <= stats["by_source"]["wrime"]["prefix_truncated_frac"] <= 1
    assert 0 <= stats["by_source"]["wrime"]["byte_fallback_token_frac"] <= 1
    # corpus の source ごと上限
    assert len(T.corpus_rows(ro, max_per_source=7)) == 14 and len(T.corpus_rows(ro)) == 40
    assert len(T.corpus_rows(ro, sources=["jcqa"])) == 20 and len(T.corpus_rows(ro, sources=["public"])) == 40
    assert len(T.corpus_rows(ro, sources=["synth"])) == 0
    a2, st2 = T.build_shard(ro, sp, "train", lp=32, lc=8, kmax=5, teacher_temp=1.0, sources=["wrime"])
    assert st2["n"] == 20 and set(a2["source"].tolist()) == {"wrime"}
    # npz に保存→ Shard が extra として読める（allow_pickle=False でも）
    np.savez(tmp_path / "s.npz", **arrays)
    from tb250distill.student import model as M

    sh = M.Shard(str(tmp_path / "s.npz"))
    assert sh.extra["source"].shape == (40,)
    sub = sh.subset(np.arange(5))
    assert sub.extra["source"].shape == (5,)


def test_by_source_metrics_only_when_multi_source():
    from tb250distill.student import evaluate as E
    from tb250distill.student import model as M

    n, K, lp, lc = 6, 3, 4, 2
    d = dict(item_id=np.arange(n), prefix=np.ones((n, lp), np.int32), prefix_len=np.full(n, lp, np.int32),
             cand=np.ones((n, K, lc), np.int32), cand_len=np.full((n, K), lc, np.int32), k=np.full(n, K, np.int32),
             t_logits=np.tile(np.array([2.0, 0.0, -1.0], np.float32), (n, 1)), gold=np.zeros(n, np.int32))
    scores = np.tile(np.array([1.0, 0.0, -1.0]), (n, 1))
    assert E.by_source_metrics(scores, M.Shard(dict(d))) == {}                                  # source 無し（旧 shard）→ 何も変わらない
    one = M.Shard(dict(d, source=np.array(["a"] * n)))
    assert E.by_source_metrics(scores, one) == {}                                                 # 単一 source → 不要
    two = M.Shard(dict(d, source=np.array(["a", "a", "a", "b", "b", "b"])))
    res = E.by_source_metrics(scores, two)
    assert set(res) == {"a", "b"} and res["a"]["n"] == 3 and res["b"]["agreement"] == 1.0 and "reliability_gold" not in res["a"]


def test_teacher_stats(conn):
    rows = [pub(1, "jcqa"), pub(2, "jcqa"), pub(3, "wrime", "val")]
    rows[2]["gold"] = None
    replay.insert_items(conn, rows)
    replay.write_teacher(conn, 1, "m", "x", [3.0, 0, 0, 0, 0], _softmax([3.0, 0, 0, 0, 0]))   # gold(=2) 外れ
    replay.write_teacher(conn, 2, "m", "x", [0, 0, 3.0, 0, 0], _softmax([0, 0, 3.0, 0, 0]))   # gold 当たり
    replay.write_teacher(conn, 3, "m", "x", [0, 1.0, 0, 0, 0], _softmax([0, 1.0, 0, 0, 0]))
    st = P.teacher_stats(conn)
    assert st["jcqa"]["train"]["scored"] == 2 and st["jcqa"]["train"]["teacher_gold_acc"] == 0.5
    assert st["jcqa"]["train"]["random_gold_acc"] == 0.2
    assert st["wrime"]["val"]["with_gold"] == 0 and st["wrime"]["val"]["teacher_gold_acc"] is None
