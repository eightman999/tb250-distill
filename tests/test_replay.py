import json
import math
import threading

import pytest

from tb250distill import replay


def mk(i, split="train", k=3, gold=0, **kw):
    d = dict(item_id=i, split=split, category="intent", lang="ja", context=f"ctx{i}", question="q",
             candidates=[f"cand{j}" for j in range(k)], gold=gold, variant_of=None, variant=None, gen_seed=i)
    d.update(kw)
    return d


@pytest.fixture
def conn(tmp_path):
    c = replay.connect(str(tmp_path / "r.sqlite"))
    yield c
    c.close()


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [x / sum(e) for x in e]


def test_wal_and_schema(conn):
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"items", "teacher"} <= tables


def test_insert_and_get(conn):
    assert replay.insert_items(conn, [mk(1), mk(2, k=2, gold=None)]) == 2
    got = replay.get_items(conn, [1, 2])
    assert got[0]["candidates"] == ["cand0", "cand1", "cand2"] and got[1]["gold"] is None


def test_insert_validation(conn):
    with pytest.raises(ValueError):
        replay.insert_items(conn, [mk(1, gold=5)])
    with pytest.raises(ValueError):
        replay.insert_items(conn, [mk(1, candidates=[])])
    assert replay.counts(conn) == {}  # 失敗した insert は何も残さない


def test_pending_order_and_resume(conn):
    replay.insert_items(conn, [mk(1, "train"), mk(2, "val"), mk(3, "test"), mk(4, "train"), mk(5, "robust", variant_of=3, variant="perm")])
    order = [i["item_id"] for i in replay.pending_items(conn, 10, ["val", "test", "train", "robust"])]
    assert order == [2, 3, 1, 4, 5]
    assert [i["item_id"] for i in replay.pending_items(conn, 2, ["train"])] == [1, 4]
    lg = [0.0, -1.0, -2.0]
    replay.write_teacher(conn, 2, "m.gguf", "label_logprob_perm2", lg, softmax(lg), {"x": 1}, 12.5)
    order = [i["item_id"] for i in replay.pending_items(conn, 10, ["val", "test", "train", "robust"])]
    assert order == [3, 1, 4, 5]  # 採点済みは出ない（再開可能）


def test_write_teacher_rejects_bad(conn):
    replay.insert_items(conn, [mk(1)])
    with pytest.raises(ValueError):
        replay.write_teacher(conn, 1, "m", "x", [0.0, 1.0], [0.5, 0.5, 0.0])
    with pytest.raises(ValueError):
        replay.write_teacher(conn, 1, "m", "x", [0.0, float("nan"), 1.0], [0.3, 0.3, 0.4])
    with pytest.raises(ValueError):
        replay.write_teacher(conn, 1, "m", "x", [0.0, 1.0, 2.0], [0.3, 0.3, 0.3])


def test_iter_scored_and_counts(conn):
    replay.insert_items(conn, [mk(i, "train" if i < 4 else "val") for i in range(1, 7)])
    for i in (1, 3, 5):
        lg = [float(i), 0.0, -1.0]
        replay.write_teacher(conn, i, "m.gguf", "label_logprob_perm2", lg, softmax(lg), {"perms": []}, 1.0)
    rows = list(replay.iter_scored(conn, "train"))
    assert [r["item_id"] for r in rows] == [1, 3]
    r = rows[0]
    assert r["logits"] == [1.0, 0.0, -1.0] and abs(sum(r["probs"]) - 1) < 1e-9 and r["raw"] == {"perms": []}
    assert r["teacher_model"] == "m.gguf" and r["candidates"][0] == "cand0"
    assert [r["item_id"] for r in replay.iter_scored(conn)] == [1, 3, 5]
    assert [r["item_id"] for r in replay.iter_scored(conn, ["train", "val"], batch=1)] == [1, 3, 5]
    c = replay.counts(conn)
    assert c["train"] == {"total": 3, "scored": 2, "errors": 0} and c["val"]["scored"] == 1


def test_errors_skip_after_max_attempts(conn):
    replay.insert_items(conn, [mk(1), mk(2)])
    assert replay.write_error(conn, 1, "boom") == 1
    assert [i["item_id"] for i in replay.pending_items(conn, 10, ["train"], max_attempts=2)] == [1, 2]
    assert replay.write_error(conn, 1, "boom again") == 2
    assert [i["item_id"] for i in replay.pending_items(conn, 10, ["train"], max_attempts=2)] == [2]
    assert replay.counts(conn)["train"]["errors"] == 1
    lg = [0.0, 0.0, 0.0]
    replay.write_teacher(conn, 1, "m", "x", lg, softmax(lg))   # 成功したらエラー記録は消える
    assert replay.counts(conn)["train"]["errors"] == 0
    assert [i["item_id"] for i in replay.pending_items(conn, 10, ["train"], exclude=[2])] == []


def test_concurrent_reader_while_writing(tmp_path):
    path = str(tmp_path / "r.sqlite")
    w = replay.connect(path)
    replay.insert_items(w, [mk(i) for i in range(1, 201)])
    stop = threading.Event()
    seen = []

    def reader():
        r = replay.connect(path, readonly=True)
        while not stop.is_set():
            seen.append(sum(1 for _ in replay.iter_scored(r, "train")))
        r.close()

    t = threading.Thread(target=reader)
    t.start()
    for i in range(1, 201):
        lg = [0.0, -1.0, -2.0]
        replay.write_teacher(w, i, "m", "x", lg, softmax(lg), commit=(i % 10 == 0))
    w.commit()
    stop.set()
    t.join()
    assert seen and max(seen) <= 200 and seen == sorted(seen)
    assert replay.scored_count(w, "train") == 200


def test_existing_keys_and_max_id(conn):
    replay.insert_items(conn, [mk(7)])
    assert replay.max_item_id(conn) == 7
    assert ("ctx7", "q", json.dumps(["cand0", "cand1", "cand2"])) in replay.existing_keys(conn)
