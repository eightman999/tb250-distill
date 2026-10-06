"""fake shard での学習・resume・評価（np backend）。"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill.student import model as M, evaluate as E, train as T, fake_data as F  # noqa: E402
from tb250distill.student.backend_np import NPBackend  # noqa: E402

CFG = '{"vocab": 300, "emb": 24, "hidden": 32, "layers": 2}'


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake")
    F.write_fake_dataset(str(d), n_train=6000, n_val=200, n_test=200, n_robust=40, vocab=300, lp=48, lc=8, kmax=5,
                         seed=1, active=30)
    return str(d)


def test_shard_contract(fake):
    sh = M.Shard(M.find_shard(fake, "train", 48))
    assert sh.prefix.shape == (6000, 48) and sh.cand.shape == (6000, 5, 8)
    assert sh.t_logits.dtype == np.float32 and (sh.k >= 2).all()
    pad = np.arange(5)[None, :] >= sh.k[:, None]
    assert (sh.t_logits[pad] == 0).all() and (sh.cand_len[pad] == 0).all()
    rb = M.Shard(M.find_shard(fake, "robust", 48))
    assert set(np.unique(rb.extra["variant"])) == {"perm", "irrelevant_ctx", "ambiguous"}


def run_train(fake, run, extra=()):
    argv = ["--backend", "np", "--data", fake, "--run-dir", str(run), "--config", CFG, "--batch-size", "32",
            "--seed", "3", "--log-every", "10", "--eval-every", "40", "--lr", "3e-3", *extra]
    return T.main(argv)


def test_train_reduces_kl_and_writes_artifacts(fake, tmp_path):
    run = tmp_path / "run"
    s = run_train(fake, run, ["--epochs", "4"])  # 188 steps/epoch * 4 = 752 step
    rows = [l.split(",") for l in open(run / "metrics.csv").read().strip().splitlines()]
    hdr = rows[0]
    assert hdr == T.CSV_COLS
    kl = [float(r[hdr.index("val_kl")]) for r in rows[1:] if r[hdr.index("val_kl")]]
    assert kl[-1] < 0.7 * kl[0], kl
    ag = [float(r[hdr.index("val_agree")]) for r in rows[1:] if r[hdr.index("val_agree")]]
    assert ag[-1] > ag[0] + 0.1
    for f in ("config.json", "hardware.json", "environment.txt", "metrics.csv", "train.log", "eval.json"):
        assert (run / f).exists(), f
    for p in ("p000", "p010", "p025", "p050", "p075", "p100", "best", "last"):
        assert (run / "ckpt" / f"{p}.npz").exists() and (run / "ckpt" / f"{p}.json").exists(), p
    ev = json.load(open(run / "eval.json"))
    assert ev["splits"]["val"]["agreement"] > ev["splits"]["val"]["random_baseline"] + 0.1
    assert ev["permutation_test"]["max_abs_prob_diff"] < 1e-5   # float32 の丸め誤差のみ
    assert ev["robust"]["available"] and set(ev["robust"]["variants"]) == {"perm", "irrelevant_ctx", "ambiguous"}
    assert ev["robust"]["variants"]["perm"]["n_paired"] > 0
    assert ev["inference"]["throughput"]["items_per_s"] > 0
    cfg = json.load(open(run / "config.json"))
    assert cfg["n_params"] == M.param_count(M.Config.from_dict(cfg["model"]))
    # eval.json の agreement は同じ ckpt を evaluate CLI で読み直しても一致
    out = tmp_path / "eval2.json"
    E.main(["--backend", "np", "--data", fake, "--ckpt", str(run / "ckpt" / "last.npz"), "--out", str(out),
            "--no-latency", "--batch-size", "32"])
    ev2 = json.load(open(out))
    assert abs(ev2["splits"]["val"]["kl"] - ev["splits"]["val"]["kl"]) < 1e-9


def test_resume_is_exact(fake, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_train(fake, a, ["--max-steps", "30", "--no-final-eval"])
    run_train(fake, b, ["--max-steps", "12", "--no-final-eval"])
    run_train(fake, b, ["--max-steps", "30", "--no-final-eval", "--resume"])
    ca, pa, ma, va, meta_a = T.load_ckpt(str(a / "ckpt" / "last.npz"))
    cb, pb, mb, vb, meta_b = T.load_ckpt(str(b / "ckpt" / "last.npz"))
    assert meta_a["step"] == meta_b["step"] == 30
    for k in pa:
        assert np.array_equal(pa[k], pb[k]), k
    assert np.array_equal(ma, mb) and np.array_equal(va, vb)


def test_resume_refuses_overwrite(fake, tmp_path):
    run_train(fake, tmp_path / "x", ["--max-steps", "5", "--no-final-eval"])
    with pytest.raises(SystemExit):
        run_train(fake, tmp_path / "x", ["--max-steps", "5", "--no-final-eval"])


def test_order_is_device_independent_and_seed_determined():
    o1, k1 = T.epoch_order(3, 0, 100, 5)
    o2, k2 = T.epoch_order(3, 0, 100, 5)
    o3, _ = T.epoch_order(3, 1, 100, 5)
    assert np.array_equal(o1, o2) and np.array_equal(k1, k2) and not np.array_equal(o1, o3)


def test_host_loss_matches_kernel_loss(fake):
    cfg = M.get_config(CFG)
    cfg.lp, cfg.lc = 48, 8
    be = NPBackend(np.float64)
    st = M.Student(be, cfg, 32, 5, train=True)
    st.set_params({k: v.astype(np.float64) for k, v in M.init_params(cfg, 0).items()})
    sh = M.Shard(M.find_shard(fake, "val", 48))
    idx = np.arange(32)
    bt = M.make_batch(sh, idx)
    r = st.loss_grads(bt, want_grads=False)
    s = st.predict(bt)
    loss, kd, ce = E.composite_loss(s, bt.tlog.reshape(32, -1).astype(np.float64), bt.kcnt, bt.gold)
    assert abs(r["loss"] - loss.mean()) < 1e-12 and abs(r["kd"] - kd.mean()) < 1e-12


def test_gold_less_items_use_kd_only():
    s = np.array([[1.0, 2.0, 0.5]])
    t = np.array([[0.2, 0.1, 0.9]])
    k = np.array([3])
    l_nog, kd, ce = E.composite_loss(s, t, k, np.array([-1]))
    l_g, kd2, ce2 = E.composite_loss(s, t, k, np.array([1]))
    assert np.isclose(l_nog[0], kd[0]) and np.isclose(l_g[0], 0.8 * kd2[0] + 0.2 * ce2[0])


def test_temperature_guard_pauses_and_records(fake, tmp_path, monkeypatch):
    """温度が temp_pause 以上なら temp_resume 以下まで待機し、paused_s を metrics/ckpt に記録する。"""
    temps = iter([70, 70, 90, 85, 80, 77] + [70] * 1000)

    class FakeSmi:
        def __init__(self, be, interval=10.0):
            self.enabled = True
            self.vram = 100.0
            self.temp = None
            self.vram_max = 100.0
            self.temp_max = 0
            self.err = None
            self.n = 0

        def poll(self, force=False):
            # 通常は 1 step ごとに呼ばれる。force なし呼び出しは毎回新しい温度
            self.temp = float(next(temps))
            self.temp_max = max(self.temp_max, self.temp)

    sleeps = []
    monkeypatch.setattr(T, "SmiMonitor", FakeSmi)
    monkeypatch.setattr(T.time, "sleep", lambda s: sleeps.append(s))
    run = tmp_path / "g"
    T.main(["--backend", "np", "--data", fake, "--run-dir", str(run), "--config", CFG, "--batch-size", "32", "--seed", "3",
            "--max-steps", "10", "--no-final-eval", "--log-every", "5", "--temp-pause", "88", "--temp-resume", "78"])
    assert len(sleeps) >= 3                      # 90 -> 85 -> 80 -> 77 で再開
    log = open(run / "train.log").read()
    assert "TEMP PAUSE" in log and "TEMP RESUME" in log
    meta = json.load(open(run / "ckpt" / "last.json"))
    assert meta["pauses"] == 1 and meta["paused_s"] >= 0
    hdr = open(run / "metrics.csv").readline().strip().split(",")
    assert "paused_s" in hdr


def test_amd_power_columns_and_energy_summary(fake, tmp_path, monkeypatch):
    """AMD 監視値が metrics.csv 末尾 3 列に入り、eval.json の train_summary.energy に実測 J/sample が残る。"""
    class FakeAmdSmi:
        def __init__(self, be, interval=10.0):
            self.enabled, self.kind = True, "amdgpu"
            self.vram, self.temp, self.power, self.busy, self.sclk = 500.0, 60.0, 15.0, 80.0, 1870.0
            self.vram_max, self.temp_max, self.power_max, self.err = 500.0, 60.0, 15.0, None
            self.recording = True
            self.power_samples, self.power_integral_j, self.power_time_s = 5, 1500.0, 100.0
            self.busy_sum, self.busy_n = 400.0, 5

        def info(self):
            return {"kind": "amdgpu", "amdgpu": {"card": "card2", "method": "name_table"}}

        def power_mean(self):
            return self.power_integral_j / self.power_time_s

        def poll(self, force=False):
            pass

    monkeypatch.setattr(T, "SmiMonitor", FakeAmdSmi)
    run = tmp_path / "amd"
    s = run_train(fake, run, ["--max-steps", "40"])
    rows = [l.split(",") for l in open(run / "metrics.csv").read().strip().splitlines()]
    hdr = rows[0]
    assert hdr[-4:] == ["power_w", "gpu_busy_pct", "sclk_mhz", "sem_loss"] and hdr[:18] == T.CSV_COLS[:18]  # sem_loss は末尾追加
    last = dict(zip(hdr, rows[-1]))
    assert float(last["power_w"]) == 15.0 and float(last["gpu_busy_pct"]) == 80.0 and float(last["sclk_mhz"]) == 1870.0
    assert float(last["temp_c"]) == 60.0 and float(last["vram_mb"]) == 500.0
    e = s["energy"]
    assert e["measured"] and e["power_mean_w"] == pytest.approx(15.0) and e["samples"] == 40 * 32
    assert e["energy_per_sample_j"] == pytest.approx(15.0 * e["wall_active_s"] / (40 * 32))
    ev = json.load(open(run / "eval.json"))
    assert ev["train_summary"]["energy"]["measured"] is True
    assert json.load(open(run / "config.json"))["gpu_monitor"]["amdgpu"]["card"] == "card2"
    assert json.load(open(run / "hardware.json"))["student_gpu_monitor"]["kind"] == "amdgpu"
    assert "gpu monitor" in open(run / "train.log").read() and "ENERGY" in open(run / "train.log").read()
    assert json.load(open(run / "ckpt" / "last.json"))["samples_done"] == 40 * 32


def test_nvidia_style_monitor_leaves_new_columns_empty(fake, tmp_path, monkeypatch):
    class OldSmi:      # 旧インターフェース（power 属性なし）でも動く
        def __init__(self, be, interval=10.0):
            self.enabled, self.vram, self.temp, self.vram_max, self.temp_max, self.err = True, 100.0, 50.0, 100.0, 50.0, None

        def poll(self, force=False):
            pass

    monkeypatch.setattr(T, "SmiMonitor", OldSmi)
    run = tmp_path / "nv"
    s = run_train(fake, run, ["--max-steps", "20", "--no-final-eval"])
    rows = [l.split(",") for l in open(run / "metrics.csv").read().strip().splitlines()]
    last = dict(zip(rows[0], rows[-1]))
    assert last["power_w"] == "" and last["gpu_busy_pct"] == "" and last["sclk_mhz"] == "" and last["temp_c"] == "50"
    assert s["energy"]["measured"] is False
