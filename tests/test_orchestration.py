"""coordinator -> eval -> cascade -> report の通し検証（Mac、np backend、fake shard、GPU 不使用）。"""
import json
import os
import re
import sqlite3
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill import coordinator as C, cascade as CA, report as R  # noqa: E402
from tb250distill.student import fake_data as F  # noqa: E402

GPUS = "GT 430,GT 710,GT 730"
STEPS = "25"


def wait_done(runs_dir, plan, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = C.status_rows(runs_dir, plan)
        if rows and all(r["state"] != "running" for r in rows):
            return rows
        time.sleep(0.5)
    raise AssertionError("timeout: " + json.dumps(C.status_rows(runs_dir, plan), indent=1))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("orch")
    data = root / "tok"
    F.write_fake_dataset(str(data), n_train=400, n_val=150, n_test=150, n_robust=30, vocab=300, lp=48, lc=8, kmax=5,
                         seed=2, active=30)
    cfgs = {}
    for name, (e, h, l) in {"t430": (12, 16, 2), "t710": (16, 20, 2), "t730": (16, 24, 3), "common": (16, 24, 2)}.items():
        p = root / f"{name}.json"
        p.write_text(json.dumps({"vocab": 300, "emb": e, "hidden": h, "layers": l, "lp": 48, "lc": 8}))
        cfgs[name] = str(p)
    # teacher latency 付き replay DB（cascade 用の最小スキーマ）
    db = root / "replay.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE teacher(item_id INTEGER PRIMARY KEY, latency_ms REAL)")
    ids = []
    for sp in ("val", "test"):
        ids += [int(i) for i in np.load(data / f"{sp}_L48.npz")["item_id"]]
    con.executemany("INSERT INTO teacher VALUES (?,?)", [(i, 800.0 + (i % 7)) for i in ids])
    con.commit()
    con.close()
    return {"root": root, "data": str(data), "runs": str(root / "runs"), "cfgs": cfgs, "db": str(db)}


def co(env, *args):
    return C.main([args[0], "--runs-dir", env["runs"], *args[1:]])


def test_commit_value(monkeypatch):
    monkeypatch.delenv("TB250_GIT_COMMIT", raising=False)
    assert re.fullmatch(r"[0-9a-f]{40}|uncommitted-[0-9a-f]{12}", C.git_commit_value())
    monkeypatch.setenv("TB250_GIT_COMMIT", "abc123")
    assert C.git_commit_value() == "abc123"


def test_init_not_overwritten(env, capsys):
    out = os.path.join(env["runs"], "common", "init.npz")
    assert co(env, "init", "--config", env["cfgs"]["common"], "--seed", "0", "--out", out) == 0
    first = open(out, "rb").read()
    assert co(env, "init", "--config", env["cfgs"]["common"], "--seed", "9", "--out", out) == 0
    assert "not overwritten" in capsys.readouterr().out
    assert open(out, "rb").read() == first


def test_dry_run_writes_nothing(env, tmp_path, capsys):
    runs = str(tmp_path / "r")
    assert C.main(["launch", "--runs-dir", runs, "--plan", "optimized", "--data", env["data"], "--dry-run",
                   "--backend", "cl", "--epochs", "3"]) == 0
    out = capsys.readouterr().out
    assert "--device 'GT 430'" in out and "--config s730" in out and "init_s710.npz" in out
    assert "--lp 256" in out   # s730 は Lp 256
    assert not os.path.exists(runs)


def test_common_pipeline(env):
    r = env["runs"]
    os.environ["TB250_GIT_COMMIT"] = "test-commit-xyz"
    try:
        rc = co(env, "launch", "--plan", "common", "--data", env["data"], "--backend", "np", "--gpus", GPUS,
                "--common-config", env["cfgs"]["common"], "--max-steps", STEPS, "--eval-every", "10", "--log-every", "5",
                "--batch-size", "16", "--seed", "3", "--lr", "3e-3")
    finally:
        os.environ.pop("TB250_GIT_COMMIT", None)
    assert rc == 0
    plan = json.load(open(os.path.join(r, "coordinator", "common.json")))
    assert set(plan["runs"]) == {"gt430", "gt710", "gt730"}
    pids = {k: v["pid"] for k, v in plan["runs"].items()}
    assert len(set(pids.values())) == 3 and all(v["started_at"] and v["cmdline"] for v in plan["runs"].values())
    assert plan["git_commit"] == "test-commit-xyz"
    # 同一 init を使っている
    assert {v["init"] for v in plan["runs"].values()} == {os.path.join(r, "common", "init.npz")}
    # 実行中の再 launch は拒否される（他の run には触らない）か、既に終わっている場合は ckpt 検出で skip
    rows = wait_done(r, "common")
    assert [x["state"] for x in rows] == ["finished"] * 3, rows
    for x in rows:
        assert os.path.exists(os.path.join(x["run_dir"], "exit_code"))
        assert os.path.getsize(os.path.join(x["run_dir"], "stdout.log")) > 0
        cfg = json.load(open(os.path.join(x["run_dir"], "config.json")))
        assert cfg["git_commit"] == "test-commit-xyz"          # 環境変数が子に渡った
        assert cfg["args"]["seed"] == 3 and cfg["args"]["batch_size"] == 16
        assert x["step"] == "25" and x["val_kl"] and x["samples_per_s"]
    # 同一条件: step 0 の val_kl（同一 init・同一 val）が 3 run で一致
    kl0 = [open(os.path.join(x["run_dir"], "metrics.csv")).read().splitlines()[1].split(",")[6] for x in rows]
    assert len(set(kl0)) == 1
    # 既存 checkpoint のある run dir は --resume 無しだと起動拒否（他 run は影響なし）
    rc = co(env, "launch", "--plan", "common", "--data", env["data"], "--backend", "np", "--gpus", GPUS,
            "--common-config", env["cfgs"]["common"], "--max-steps", STEPS)
    assert rc == 1
    plan2 = json.load(open(os.path.join(r, "coordinator", "common.json")))
    assert {v["pid"] for v in plan2["runs"].values()} == set(pids.values())   # 前回記録が保持される
    assert all("last_launch_error" in v for v in plan2["runs"].values())


def test_eval_cascade_report(env):
    r = env["runs"]
    rc = co(env, "eval", "--plan", "common", "--split", "test", "--robust", "--replay-db", env["db"])
    assert rc == 0
    for g in ("gt430", "gt710", "gt730"):
        d = os.path.join(r, "common", g)
        ev = json.load(open(os.path.join(d, "eval.json")))
        assert ev["meta"]["ckpt"].endswith("best.npz") and {"val", "test", "robust"} <= set(ev["splits"])
        assert os.path.exists(os.path.join(d, "eval_train_final.json"))
        z = np.load(os.path.join(d, "preds.npz"))
        assert z["val_probs"].shape[0] == 150 and z["test_latency_ms"].shape == (150,)
        assert np.isfinite(z["test_latency_ms"]).all()
        assert np.allclose(z["val_probs"].sum(1), 1.0)
    cj = os.path.join(r, "cascade", "cascade_common.json")
    CA.main(["--plan", "common", "--runs-dir", r, "--replay-db", env["db"], "--tau-grid", "0.5:0.99:0.07"])
    res = json.load(open(cj))
    v = res["splits"]["val"]["cascade"]
    t = res["splits"]["test"]["cascade"]
    assert v["agreement"] >= 0.99 - 1e-9                      # 選択基準の制約
    assert t["energy_per_decision_j"] is None and t["energy_per_decision_j_estimated_tdp"] > 0
    assert t["mean_latency_ms"] > 700                          # teacher latency（~800ms）が含まれる
    assert set(res["splits"]["test"]["single"]) == {"S", "M", "L", "Teacher"}
    assert res["splits"]["test"]["single"]["Teacher"]["agreement"] == 1.0
    assert "not measured" in res["assumptions"]["energy_note"] or "実測ではない" in res["assumptions"]["energy_note"]
    md = os.path.join(env["root"], "REPORT_TABLE.md")
    assert R.main(["--runs-dir", r, "--out-md", md]) == 0
    text = open(md).read()
    for lab in ("C430", "C710", "C730"):
        assert f"| {lab} |" in text
    assert "Fermi" in text and "Kepler" in text and "Cascade" in text and "perm" in text
    rj = json.load(open(os.path.join(r, "report.json")))
    c430 = next(x for x in rj["runs"] if x["run"] == "C430")
    assert c430["eval_split"] == "test" and c430["samples_per_s"] > 0 and c430["agreement"] is not None
    assert c430["vram_mb"] is None                             # np backend は VRAM 不明 -> n/a


def test_optimized_failure_isolated(env):
    r = env["runs"]
    bad = os.path.join(env["root"], "bad.json")
    json.dump({"vocab": 100, "emb": 8, "hidden": 8, "layers": 1, "lp": 48, "lc": 8}, open(bad, "w"))   # shard の token id より小さい vocab
    presets = f"gt430={env['cfgs']['t430']},gt710={bad},gt730={env['cfgs']['t730']}"
    rc = co(env, "launch", "--plan", "optimized", "--data", env["data"], "--backend", "np", "--gpus", GPUS,
            "--presets", presets, "--max-steps", "10", "--eval-every", "5", "--batch-size", "16")
    assert rc == 0
    rows = {x["key"]: x for x in wait_done(r, "optimized")}
    assert rows["gt430"]["state"] == "finished" and rows["gt730"]["state"] == "finished"
    assert rows["gt710"]["state"].startswith("failed")         # 失敗しても他は完走
    assert "最大 token id" in (rows["gt710"].get("stderr_tail") or "")
    assert os.path.exists(os.path.join(r, "optimized", "init_t430.npz"))
    # eval は失敗 run（checkpoint 無し）を SKIP して他だけ動く
    assert co(env, "eval", "--plan", "optimized", "--split", "test") == 0
    assert os.path.exists(os.path.join(r, "optimized", "gt430", "preds.npz"))
    assert not os.path.exists(os.path.join(r, "optimized", "gt710", "preds.npz"))


def _write_preds(path, probs_fn, lat):
    """合成 preds.npz。k=2、teacher は常に候補 0。items 0-99 easy / 100-199 medium / 200-299 hard。"""
    n = 300
    ids = np.arange(n, dtype=np.int64) + 1000
    probs = np.zeros((n, 5))
    for i in range(n):
        p_correct, correct = probs_fn(i)          # 候補 0 が teacher/gold
        top = p_correct
        if correct:
            probs[i, :2] = [top, 1 - top]
        else:
            probs[i, :2] = [1 - top, top]
    tp = np.zeros((n, 5))
    tp[:, 0], tp[:, 1] = 0.9, 0.1
    z = {}
    for sp in ("val", "test"):
        z.update({f"{sp}_item_id": ids, f"{sp}_k": np.full(n, 2, np.int32), f"{sp}_probs": probs, f"{sp}_t_probs": tp,
                  f"{sp}_gold": np.zeros(n, np.int32), f"{sp}_latency_ms": np.full(n, float(lat))})
    z["__meta__"] = np.array(json.dumps({"device": "synthetic"}))
    np.savez(path, **z)


def test_cascade_threshold_selection(tmp_path):
    def s_fn(i):   # easy: 確信して正解 / medium: 0.6 で不正解 / hard: 0.55 で不正解
        return (0.95, True) if i < 100 else ((0.6, False) if i < 200 else (0.55, False))

    def m_fn(i):   # medium: 0.9 で正解 / hard: 0.55 不正解
        return (0.95, True) if i < 100 else ((0.9, True) if i < 200 else (0.55, False))

    def l_fn(i):   # hard: 0.8 で不正解（確信した誤り）
        return (0.95, True) if i < 200 else (0.8, False)

    for nm, fn, lat in (("S", s_fn, 1.0), ("M", m_fn, 5.0), ("L", l_fn, 20.0)):
        _write_preds(str(tmp_path / f"{nm}.npz"), fn, lat)
    out = str(tmp_path / "c.json")
    res = CA.main(["--stage", f"S={tmp_path/'S.npz'}", "--stage", f"M={tmp_path/'M.npz'}", "--stage", f"L={tmp_path/'L.npz'}",
                   "--replay-db", "/nonexistent", "--teacher-latency-ms", "500", "--out", out,
                   "--tdp", "S=10", "--tdp", "teacher=50"])
    th = res["thresholds"]
    assert 0.6 < th["S"] <= 0.95 and 0.55 < th["M"] <= 0.9
    assert th["L"] is None or th["L"] > 0.8                    # L は確信した誤りがあるので hard を採用してはいけない
    v = res["splits"]["val"]["cascade"]
    assert abs(v["teacher_escalation_rate"] - 1 / 3) < 1e-9   # hard の 100 件だけ Teacher へ
    assert v["agreement"] == 1.0 and v["accuracy_gold"] == 1.0
    assert abs(v["stages"][0]["escalation_rate"] - 2 / 3) < 1e-9
    # latency: easy=1, medium=1+5, hard=1+5+(L が有効なら 20)+500 -> L を使わない方が小さいので L は skipped
    assert v["stages"][2].get("skipped") is True
    assert abs(v["mean_latency_ms"] - (100 * 1 + 100 * 6 + 100 * 506) / 300) < 1e-6
    # 単独比較: S のみは medium/hard を誤る（agreement 1/3）
    assert abs(res["splits"]["test"]["single"]["S"]["agreement"] - 1 / 3) < 1e-9
    assert res["splits"]["test"]["single"]["Teacher"]["mean_latency_ms"] == 500.0
    # 推定 energy = TDP x latency（S=10W, teacher=50W。M/L は TDP 既定 19/30）
    e = res["splits"]["test"]["cascade"]["energy_per_decision_j_estimated_tdp"]
    exp = (100 * 10 * 1 + 100 * (10 * 1 + 19 * 5) + 100 * (10 * 1 + 19 * 5 + 50 * 500)) / 300 / 1e3
    assert abs(e - exp) < 1e-9
    assert res["splits"]["test"]["cascade"]["energy_per_decision_j"] is None
