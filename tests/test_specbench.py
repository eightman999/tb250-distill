"""specbench: コマンドライン生成・プラン整合・エネルギー積分・集計・実行ループ（偽 llama-server）。GPU 不要。"""
import json
import os
import socket
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill.specbench import configs as C  # noqa: E402
from tb250distill.specbench import report as R  # noqa: E402
from tb250distill.specbench import run as RUN  # noqa: E402
from tb250distill.specbench.prompts import PROMPTS  # noqa: E402


def pair(cmd, flag):
    """cmd 内の flag の直後の値。無ければ None。"""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# --------------------------------------------------------------------------------------
# build_cmd / プラン
# --------------------------------------------------------------------------------------

def test_build_cmd_baseline_has_no_draft_flags():
    cfg = C.BenchConfig("a", C.T8_Q2)
    cmd = C.build_cmd(cfg, "/x/llama", 18190)
    assert cmd[0] == "/x/llama/llama-server"
    assert pair(cmd, "-fit") == "off"
    assert pair(cmd, "-dev") == "Vulkan0" and pair(cmd, "-sm") == "none" and pair(cmd, "-ngl") == "all"
    assert pair(cmd, "-c") == "1024" and pair(cmd, "-np") == "1" and pair(cmd, "-t") == "2"
    assert pair(cmd, "--port") == "18190" and pair(cmd, "--host") == "127.0.0.1"
    for flag in ("--spec-type", "-md", "-devd", "-ngld", "-ts", "-ctk", "-ctv",
                 "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-p-min"):
        assert flag not in cmd


def test_build_cmd_spec_none_ignores_draft_fields():
    cfg = C.BenchConfig("a", C.T8_Q2, draft_model=C.D17, spec_type="none", n_max=4, p_min=0.5)
    cmd = C.build_cmd(cfg)
    assert not any(f in cmd for f in ("--spec-type", "-md", "-devd", "-ngld", "--spec-draft-n-max", "--spec-draft-p-min"))


def test_build_cmd_draft_simple():
    cfg = C.BenchConfig("a", C.T8_Q2, draft_model=C.D17, draft_device="Vulkan1", spec_type="draft-simple",
                        n_max=8, n_min=1, p_min=0.75)
    cmd = C.build_cmd(cfg)
    assert pair(cmd, "--spec-type") == "draft-simple"
    assert pair(cmd, "-md") == C.D17 and pair(cmd, "-devd") == "Vulkan1" and pair(cmd, "-ngld") == "all"
    assert pair(cmd, "--spec-draft-n-max") == "8" and pair(cmd, "--spec-draft-n-min") == "1"
    assert pair(cmd, "--spec-draft-p-min") == "0.75"
    assert "--draft-max" not in cmd and "-fit" in cmd


def test_build_cmd_ngram_has_no_draft_model_flags():
    cmd = C.build_cmd(C.BenchConfig("a", C.T8_Q2, spec_type="ngram-mod"))
    assert pair(cmd, "--spec-type") == "ngram-mod"
    assert not any(f in cmd for f in ("-md", "-devd", "-ngld", "--spec-draft-n-max"))


def test_build_cmd_split_and_cache_type():
    cfg = C.BenchConfig("a", C.T8_Q3, target_devices="Vulkan0,Vulkan1", split_mode="layer", tensor_split="3,1",
                        cache_type="q8_0")
    cmd = C.build_cmd(cfg)
    assert pair(cmd, "-dev") == "Vulkan0,Vulkan1" and pair(cmd, "-sm") == "layer" and pair(cmd, "-ts") == "3,1"
    assert pair(cmd, "-ctk") == "q8_0" and pair(cmd, "-ctv") == "q8_0"


def test_build_cmd_validation():
    with pytest.raises(ValueError):
        C.build_cmd(C.BenchConfig("a", C.T8_Q2, spec_type="draft-simple"))      # draft_model 無し
    with pytest.raises(ValueError):
        C.build_cmd(C.BenchConfig("a", C.T8_Q2, spec_type="bogus"))


def test_build_env_sets_icd_and_library_path(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/old")
    env = C.build_env("/x/llama")
    assert env["VK_ICD_FILENAMES"] == "/usr/share/vulkan/icd.d/radeon_icd.json"
    assert env["LD_LIBRARY_PATH"] == "/x/llama:/old"


@pytest.mark.parametrize("plan", sorted(C.PLANS))
def test_plan_consistency(plan):
    ref, cfgs = C.PLANS[plan]
    names = [c.name for c in cfgs]
    assert len(names) == len(set(names))
    assert ref in names
    for c in cfgs:
        cmd = C.build_cmd(c)                      # 全 config で例外なく作れる
        assert pair(cmd, "-fit") == "off"
        if c.spec_type == "draft-simple":
            assert c.draft_model
        if c.spec_type == "none":
            assert "--spec-type" not in cmd


def test_main_plan_contents():
    ref, cfgs = C.get_plan("main")
    assert ref == "t8q2_rx"
    by = {c.name: c for c in cfgs}
    expected = {"t8q2_rx", "t8q2_rx__d17wx_n2", "t8q2_rx__d17wx_n4", "t8q2_rx__d17wx_n8", "t8q2_rx__d17wx_n8_p075",
                "t8q2_rx__d17rx_n4", "t8q2_rx__d17cpu_n4", "t8q2_rx__ngram_simple", "t8q2_rx__ngram_mod",
                "t8q2_split", "t8q3_split", "t8q3_rx", "t8q3_rx__d17wx_n4"}
    assert set(by) == expected
    assert by["t8q2_rx__d17cpu_n4"].draft_device == "none"
    assert by["t8q2_rx__d17rx_n4"].draft_device == "Vulkan0"
    assert by["t8q2_rx__d17wx_n8_p075"].p_min == 0.75
    assert by["t8q3_rx"].cache_type == "q8_0"
    assert by["t8q2_split"].split_mode == "layer" and by["t8q2_split"].target_devices == "Vulkan0,Vulkan1"


def test_get_plan_only():
    ref, cfgs = C.get_plan("main", ["t8q3_split", "t8q2_rx"])
    assert [c.name for c in cfgs] == ["t8q2_rx", "t8q3_split"]          # プラン内の順序
    with pytest.raises(ValueError):
        C.get_plan("main", ["nope"])


def test_prompts_shape():
    assert len(PROMPTS) >= 8
    assert len({p["id"] for p in PROMPTS}) == len(PROMPTS)
    assert all({"id", "category", "prompt"} <= set(p) and p["prompt"] for p in PROMPTS)


# --------------------------------------------------------------------------------------
# エネルギー積分・温度ガード
# --------------------------------------------------------------------------------------

def test_integrate_power_constant_and_ramp():
    s = [(float(t), 10.0) for t in range(0, 11)]
    assert RUN.integrate_power(s, 2.0, 7.0) == pytest.approx(50.0)
    ramp = [(float(t), float(t)) for t in range(0, 11)]                 # P = t [W]
    assert RUN.integrate_power(ramp, 2.0, 6.0) == pytest.approx(16.0)   # ∫t dt = (36-4)/2
    assert RUN.integrate_power(ramp, 2.5, 3.5) == pytest.approx(3.0)    # 窓の端は補間


def test_integrate_power_edges_and_none():
    s = [(0.0, 10.0), (1.0, 10.0), (2.0, None), (3.0, 10.0)]            # None は読み飛ばして補間
    assert RUN.integrate_power(s, 0.0, 3.0) == pytest.approx(30.0)
    assert RUN.integrate_power([], 0.0, 1.0) is None
    assert RUN.integrate_power([(0.0, None)], 0.0, 1.0) is None
    assert RUN.integrate_power([(0.0, 10.0), (1.0, 10.0)], 5.0, 6.0) is None        # サンプルから遠すぎる
    assert RUN.integrate_power([(0.0, 10.0), (1.0, 10.0)], 0.5, 1.5) == pytest.approx(10.0)   # 端は最寄り値で代用
    assert RUN.integrate_power([(0.0, 10.0)], 1.0, 1.0) == 0.0


def make_tel(power=None, temp=None, **kw):
    def sampler(ddir):
        if ddir is None:
            return {"vram_mb": None, "temp_c": None, "power_w": None, "busy_pct": None, "sclk_mhz": None}
        return {"vram_mb": 100.0, "temp_c": (temp or {}).get(ddir, 50.0), "power_w": (power or {"rx": 10.0, "wx": 5.0})[ddir],
                "busy_pct": 1.0, "sclk_mhz": 500.0}
    return RUN.Telemetry({"rx": "rx", "wx": "wx"}, interval=0.02, sampler=sampler, **kw)


def test_telemetry_energy_sum_and_none():
    t = [0.0]
    tel = make_tel(clock=lambda: t[0])
    for i in range(11):
        t[0] = float(i)
        tel.sample_once()
    e = tel.energy(2.0, 6.0)
    assert e["energy"]["rx"] == pytest.approx(40.0) and e["energy"]["wx"] == pytest.approx(20.0)
    assert e["total"] == pytest.approx(60.0)
    # 1 枚が読めない（device_dir None）なら合計も None
    tel2 = RUN.Telemetry({"rx": "rx", "wx": None}, interval=0.02, clock=lambda: t[0],
                         sampler=lambda d: {"power_w": 10.0 if d else None, "temp_c": 40.0 if d else None})
    for i in range(11):
        t[0] = float(i)
        tel2.sample_once()
    e2 = tel2.energy(2.0, 6.0)
    assert e2["energy"]["rx"] == pytest.approx(40.0) and e2["energy"]["wx"] is None and e2["total"] is None


def test_temp_guard_waits_until_below_lo():
    state = {"temp": 90.0}
    tel = make_tel()
    tel.sampler = lambda d: {"temp_c": state["temp"], "power_w": 1.0}
    tel.sample_once()
    clock = [0.0]

    def sleep(dt):
        clock[0] += dt
        state["temp"] -= 5.0                     # 5 秒ごとに 5 度下がる
        tel.sample_once()

    waited = RUN.temp_guard(tel, sleep=sleep, clock=lambda: clock[0], log=lambda m: None)
    assert state["temp"] < 78.0 and waited == pytest.approx(clock[0]) and waited >= 15.0
    state["temp"] = 70.0
    tel.sample_once()
    assert RUN.temp_guard(tel, sleep=sleep, clock=lambda: clock[0], log=lambda m: None) == 0.0
    assert RUN.temp_guard(None) == 0.0


# --------------------------------------------------------------------------------------
# 起動前チェック
# --------------------------------------------------------------------------------------

GOOD_DEVICES = """ggml_vulkan: Found 2 Vulkan devices:
Available devices:
  Vulkan0: AMD Radeon RX 6400 (RADV NAVI24) (4080 MiB, 3900 MiB free)
  Vulkan1: AMD Radeon Pro WX 2100 (RADV POLARIS12) (2048 MiB, 2000 MiB free)
"""


def test_verify_devices():
    assert RUN.verify_devices(GOOD_DEVICES) == []
    swapped = GOOD_DEVICES.replace("Vulkan0: AMD Radeon RX 6400", "Vulkanx").replace("Vulkan1: AMD Radeon Pro WX 2100", "Vulkan0: AMD Radeon Pro WX 2100")
    assert len(RUN.verify_devices(swapped)) == 2
    assert RUN.verify_devices("") != []


def test_find_llama_servers_proc(tmp_path):
    for pid, comm in (("123", "llama-server"), ("456", "python3"), ("789", "llama-server")):
        d = tmp_path / pid
        d.mkdir()
        (d / "comm").write_text(comm + "\n")
    (tmp_path / "self").mkdir()
    found = RUN.find_llama_servers(str(tmp_path))
    assert [p for p, _ in found] == [123, 789]


def make_llama_dir(tmp_path):
    d = tmp_path / "llama"
    d.mkdir()
    (d / "llama-server").write_text("#!/bin/sh\n")
    m = tmp_path / "m.gguf"
    m.write_bytes(b"x")
    return d, m


def test_preflight(tmp_path):
    d, m = make_llama_dir(tmp_path)
    cfg = C.BenchConfig("a", str(m))
    hooks = RUN.Hooks(list_devices=lambda _d: GOOD_DEVICES, server_version=lambda _d: "version: 1",
                      find_servers=lambda: [], port_free=lambda _p: True)
    problems, info = RUN.preflight([cfg], str(d), 18190, hooks, check_icd=False)
    assert problems == [] and info["version"] == "version: 1" and "Vulkan0" in info["list_devices"]

    bad = RUN.Hooks(list_devices=lambda _d: "Vulkan0: NVIDIA GT 730", server_version=lambda _d: "v",
                    find_servers=lambda: [(42, "/usr/bin/llama-server")], port_free=lambda _p: False)
    problems, _ = RUN.preflight([C.BenchConfig("a", str(tmp_path / "missing.gguf"))], str(d), 18190, bad, check_icd=False)
    text = "\n".join(problems)
    assert "モデルが無い" in text and "ポート" in text and "pid 42" in text and "Vulkan0" in text and "Vulkan1" in text

    problems, _ = RUN.preflight([cfg], str(tmp_path / "nodir"), 18190, hooks, check_icd=False)
    assert any("llama-server が無い" in p for p in problems)


def test_dry_run_prints_commands_without_side_effects(tmp_path, capsys):
    out = tmp_path / "x"
    rc = RUN.main(["--plan", "main", "--out", str(out), "--dry-run"])
    txt = capsys.readouterr().out
    assert rc == 0 and not out.exists()
    assert txt.count("llama-server -m") == 13 and "-fit off" in txt and "--spec-draft-n-max 8" in txt
    rc = RUN.main(["--plan", "main", "--list", "--only", "t8q2_rx"])
    assert rc == 0
    assert RUN.main(["--plan", "main", "--only", "nope", "--dry-run"]) == 2


# --------------------------------------------------------------------------------------
# 偽 llama-server を使った run ループ
# --------------------------------------------------------------------------------------

FAKE_SERVER = textwrap.dedent('''
    import hashlib, json, sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    port, mode = int(sys.argv[1]), sys.argv[2]
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            self._send({"status": "ok"}) if self.path == "/health" else self._send({}, 404)
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert req["temperature"] == 0 and req["seed"] == 42 and req["cache_prompt"] is False and req["ignore_eos"] is True
            p, n = req["prompt"], req["n_predict"]
            content = hashlib.sha256(p.encode()).hexdigest()[:n]
            spec = mode == "spec"
            if spec and len(p) % 2 == 0:
                content += "!"                      # 一部の prompt で出力が分岐する
            t = {"cache_n": 0, "prompt_n": 5, "prompt_ms": 50.0, "prompt_per_second": 100.0, "predicted_n": n,
                 "predicted_ms": 100.0, "predicted_per_second": 40.0 if spec else 20.0}
            if spec:
                t.update(draft_n=8, draft_n_accepted=6)
            self._send({"content": content, "tokens_predicted": n, "tokens_evaluated": 5, "stop_type": "limit", "timings": t})
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')

TEST_PROMPTS = [{"id": "p_even", "category": "cat_a", "prompt": "aaaa"}, {"id": "p_odd", "category": "cat_b", "prompt": "bbb"}]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_builder(tmp_path):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER)
    calls = []

    def builder(cfg, llama_dir, port):
        calls.append(cfg.name)
        if cfg.name == "bad":
            return [sys.executable, "-c", "import sys; print('boom: out of memory'); sys.exit(3)"]
        if cfg.name == "hang":
            return [sys.executable, "-c", "import time; time.sleep(60)"]
        return [sys.executable, str(script), str(port), "spec" if cfg.spec_type != "none" else "base"]

    return builder, calls


def make_opts(tmp_path, builder, **kw):
    return RUN.RunOptions(out=tmp_path / "out", reps=1, n_predict=16, port=free_port(), load_timeout=20.0,
                          settle_s=0.0, cmd_builder=builder, env_builder=lambda d: dict(os.environ),
                          health_interval=0.05, log=lambda m: None, **kw)


TEST_CFGS = [
    C.BenchConfig("ref", "m", note="reference"),
    C.BenchConfig("spec", "m", draft_model="d", spec_type="draft-simple", n_max=4, note="draft"),
    C.BenchConfig("bad", "m", note="起動に失敗する"),
]


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def test_run_bench_loop_and_report(tmp_path):
    builder, calls = fake_builder(tmp_path)
    opts = make_opts(tmp_path, builder)
    tel = make_tel()
    tel.csv_path = opts.out / "telemetry.csv"
    summary = RUN.run_bench("test", "ref", TEST_CFGS, opts, prompts=TEST_PROMPTS, tel=tel)
    out = opts.out
    results = read_jsonl(out / "results.jsonl")
    assert len(results) == 4 and all(r["ok"] for r in results)
    assert {(r["config"], r["prompt_id"]) for r in results} == {(c, p) for c in ("ref", "spec") for p in ("p_even", "p_odd")}
    for r in results:
        assert r["rep"] == 0 and r["tokens_predicted"] == 16 and len(r["sha256"]) == 64 and r["stop_type"] == "limit"
        assert r["energy_j"] == pytest.approx(r["energy_j_rx"] + r["energy_j_wx"])
        assert r["energy_j"] == pytest.approx(15.0 * (r["t_end"] - r["t_start"]), rel=0.2, abs=0.05)
    spec_rows = {r["prompt_id"]: r for r in results if r["config"] == "spec"}
    assert spec_rows["p_odd"]["match_ref"] is True and spec_rows["p_even"]["match_ref"] is False
    assert spec_rows["p_odd"]["draft_n"] == 8 and spec_rows["p_odd"]["draft_n_accepted"] == 6
    assert all(r["match_ref"] is True for r in results if r["config"] == "ref")
    outputs = read_jsonl(out / "outputs.jsonl")
    assert len(outputs) == 4 and all(o["content"] for o in outputs)

    status = {s["config"]: s for s in read_jsonl(out / "status.jsonl")}
    assert status["ref"]["status"] == "ok" and status["spec"]["status"] == "ok"
    assert status["bad"]["status"] == "load_failed" and status["bad"]["exit_code"] == 3
    assert any("out of memory" in line for line in status["bad"]["log_tail"])
    assert calls[0] == "ref"                                              # reference が先頭

    assert (out / "env.json").exists() and (out / "plan.json").exists() and (out / "server-ref.log").exists()
    assert (out / "telemetry.csv").read_text().startswith("t,config,card,vram_mb")
    md = (out / "summary.md").read_text()
    assert "load_failed" in md and "boom" in md and "| spec |" in md and "cat_a" in md
    sj = json.loads((out / "summary.json").read_text())
    by = {c["name"]: c for c in sj["configs"]}
    assert by["spec"]["speedup"] == pytest.approx(2.0) and by["ref"]["speedup"] == pytest.approx(1.0)
    assert by["spec"]["accept_rate"] == pytest.approx(0.75) and by["spec"]["accept_len_est"] == pytest.approx(3.0)
    assert by["spec"]["match_ref_rate"] == pytest.approx(0.5) and by["bad"]["status"] == "load_failed"
    assert by["ref"]["vram_peak_mb_rx"] == pytest.approx(100.0)
    assert summary["reference"] == "ref"

    # 既存の出力先に --resume なしで再実行はしない
    with pytest.raises(RuntimeError):
        RUN.run_bench("test", "ref", TEST_CFGS, opts, prompts=TEST_PROMPTS)

    # --resume: 完了済みはスキップ、load_failed も再試行しない
    calls.clear()
    opts2 = make_opts(tmp_path, builder, resume=True)
    opts2.port = opts.port
    RUN.run_bench("test", "ref", TEST_CFGS, opts2, prompts=TEST_PROMPTS)
    assert calls == [] and len(read_jsonl(out / "results.jsonl")) == 4
    # --retry-failed なら bad だけ起動し直す
    opts3 = make_opts(tmp_path, builder, resume=True, retry_failed=True)
    RUN.run_bench("test", "ref", TEST_CFGS, opts3, prompts=TEST_PROMPTS)
    assert calls == ["bad"]


def test_run_bench_resume_fills_missing_requests(tmp_path):
    builder, calls = fake_builder(tmp_path)
    opts = make_opts(tmp_path, builder)
    cfgs = TEST_CFGS[:2]
    RUN.run_bench("test", "ref", cfgs, opts, prompts=TEST_PROMPTS)
    rp = opts.out / "results.jsonl"
    rows = read_jsonl(rp)
    keep = [r for r in rows if not (r["config"] == "spec" and r["prompt_id"] == "p_odd")]      # 1 件欠けた状態にする
    rp.write_text("".join(json.dumps(r) + "\n" for r in keep))
    calls.clear()
    opts2 = make_opts(tmp_path, builder, resume=True)
    RUN.run_bench("test", "ref", cfgs, opts2, prompts=TEST_PROMPTS)
    assert calls == ["spec"]
    rows = read_jsonl(rp)
    assert len(rows) == 4 and sum(r["config"] == "spec" and r["prompt_id"] == "p_odd" for r in rows) == 1


def test_run_bench_load_timeout_stops_server(tmp_path):
    builder, _ = fake_builder(tmp_path)
    opts = make_opts(tmp_path, builder)
    opts.load_timeout = 0.5
    RUN.run_bench("test", "hang", [C.BenchConfig("hang", "m")], opts, prompts=TEST_PROMPTS)
    st = read_jsonl(opts.out / "status.jsonl")[-1]
    assert st["status"] == "load_timeout"
    pid = st["pid"]
    with pytest.raises(ProcessLookupError):
        for _ in range(50):                                   # 停止済み（ゾンビも reap 済み）であること
            os.kill(pid, 0)
            time.sleep(0.1)


def test_run_bench_interrupt_stops_server(tmp_path):
    builder, _ = fake_builder(tmp_path)
    opts = make_opts(tmp_path, builder)
    n = {"c": 0}
    real_log = opts.log

    def log(m):
        n["c"] += 1
        if "rep0" in m:                                       # 1 件目の request 後に Ctrl-C 相当
            raise KeyboardInterrupt
        real_log(m)

    opts.log = log
    with pytest.raises(KeyboardInterrupt):
        RUN.run_bench("test", "ref", TEST_CFGS[:1], opts, prompts=TEST_PROMPTS)
    st = read_jsonl(opts.out / "status.jsonl")[-1]
    assert st["status"] == "interrupted"
    with pytest.raises(ProcessLookupError):
        for _ in range(50):
            os.kill(st["pid"], 0)
            time.sleep(0.1)
    assert (opts.out / "summary.md").exists()                  # 中断しても summary は出る


# --------------------------------------------------------------------------------------
# report の集計（偽の results）
# --------------------------------------------------------------------------------------

def rec(cfg, rep, pid, cat, gps, sha, wall=1.0, tokens=100, draft=(0, 0), e=None, **kw):
    r = {"config": cfg, "rep": rep, "prompt_id": pid, "category": cat, "ok": True, "tokens_predicted": tokens,
         "wall_s": wall, "sha256": sha,
         "timings": {"predicted_per_second": gps, "prompt_per_second": 50.0, "draft_n": draft[0], "draft_n_accepted": draft[1]}}
    if e:
        r.update(energy_j=e[0] + e[1], energy_j_rx=e[0], energy_j_wx=e[1])
    r.update(kw)
    return r


def test_report_aggregation(tmp_path):
    out = tmp_path
    (out / "plan.json").write_text(json.dumps({
        "plan": "t", "reference": "ref",
        "configs": [{"name": "ref", "spec_type": "none"}, {"name": "spec", "spec_type": "draft-simple", "n_max": 4},
                    {"name": "oom", "spec_type": "draft-simple"}]}))
    rows = [
        rec("ref", 0, "p1", "c1", 10.0, "A", e=(10.0, 2.0)),
        rec("ref", 0, "p2", "c2", 10.0, "B", e=(10.0, 2.0)),
        rec("ref", 1, "p1", "c1", 10.0, "A"),                                     # reference 自身の再現性
        rec("spec", 0, "p1", "c1", 20.0, "A", wall=0.5, draft=(8, 4), e=(5.0, 5.0)),
        rec("spec", 0, "p2", "c2", 30.0, "X", wall=0.5, draft=(8, 6), e=(5.0, 5.0)),
        {"config": "spec", "rep": 1, "prompt_id": "p1", "ok": False, "error": "timeout"},
    ]
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "status.jsonl").write_text(
        json.dumps({"config": "ref", "status": "ok", "load_s": 5.0}) + "\n"
        + json.dumps({"config": "spec", "status": "ok", "load_s": 8.0}) + "\n"
        + json.dumps({"config": "oom", "status": "load_failed", "load_s": 2.0, "log_tail": ["ggml: out of memory"]}) + "\n")
    (out / "telemetry.csv").write_text(
        "t,config,card,vram_mb,temp_c,power_w,busy_pct,sclk_mhz\n"
        "1,ref,rx,3000,60,40,90,2000\n2,ref,rx,3500,70,45,90,2000\n3,ref,wx,200,50,5,0,300\n"
        "4,spec,rx,3600,65,40,90,2000\n5,spec,wx,1800,81,20,90,1000\n")

    s = R.write_summary(out)
    by = {c["name"]: c for c in s["configs"]}
    assert by["ref"]["gen_tps"] == 10.0 and by["ref"]["speedup"] == 1.0
    assert by["spec"]["gen_tps"] == 25.0 and by["spec"]["speedup"] == pytest.approx(2.5)
    assert by["spec"]["wall_tps"] == pytest.approx(200.0) and by["ref"]["wall_tps"] == pytest.approx(100.0)
    assert by["spec"]["speedup_wall"] == pytest.approx(2.0)
    assert by["spec"]["accept_rate"] == pytest.approx(10 / 16) and by["spec"]["accept_len_est"] == pytest.approx(2.5)
    assert by["ref"]["accept_rate"] is None
    assert by["spec"]["n_ok"] == 2 and by["spec"]["n_fail"] == 1
    assert by["spec"]["match_ref_rate"] == pytest.approx(0.5) and by["spec"]["match_ref_total"] == 2
    assert by["ref"]["match_ref_rate"] == pytest.approx(1.0) and by["ref"]["match_ref_total"] == 1   # rep1 だけ比較
    assert by["spec"]["j_per_token"] == pytest.approx(20.0 / 200) and by["spec"]["j_per_token_rx"] == pytest.approx(10.0 / 200)
    assert by["ref"]["j_per_token"] == pytest.approx(24.0 / 200) and by["ref"]["j_per_token_wx"] == pytest.approx(4.0 / 200)
    assert by["ref"]["vram_peak_mb_rx"] == 3500 and by["ref"]["vram_peak_mb_wx"] == 200
    assert by["spec"]["temp_max_c_wx"] == 81 and by["spec"]["temp_max_c_rx"] == 65
    assert by["oom"]["status"] == "load_failed" and by["oom"]["gen_tps"] is None and by["oom"]["speedup"] is None
    assert s["category_gen_tps"]["spec"] == {"c1": 20.0, "c2": 30.0} and s["category_gen_tps"]["oom"] == {"c1": None, "c2": None}
    md = (out / "summary.md").read_text()
    assert "2.50x" in md and "0.62" in md and "out of memory" in md and "load_failed" in md
    assert json.loads((out / "summary.json").read_text())["reference"] == "ref"
    assert R.main([str(out)]) == 0
    assert R.main([str(out / "nope")]) == 2


def test_read_card_adds_gtt(tmp_path):
    from tb250distill.specbench import run as sbrun
    (tmp_path / "mem_info_vram_used").write_text(str(3000 * 1024 * 1024))
    (tmp_path / "mem_info_gtt_used").write_text(str(1066 * 1024 * 1024))
    v = sbrun.read_card(str(tmp_path))
    assert v["vram_mb"] == pytest.approx(3000) and v["gtt_mb"] == pytest.approx(1066)
    assert sbrun.read_card(None)["gtt_mb"] is None


def test_report_gtt_peak(tmp_path):
    from tb250distill.specbench import report as sbreport
    p = tmp_path / "telemetry.csv"
    p.write_text("t,config,card,vram_mb,temp_c,power_w,busy_pct,sclk_mhz,gtt_mb\n"
                 "1,a,rx,3363,50,30,90,2000,13\n2,a,rx,3363,50,30,90,2000,1066\n")
    peaks = sbreport.load_telemetry_peaks(p)
    assert peaks["a"]["rx"]["gtt_peak_mb"] == pytest.approx(1066)
    old = tmp_path / "old.csv"   # gtt_mb 列の無い main1 形式
    old.write_text("t,config,card,vram_mb,temp_c,power_w,busy_pct,sclk_mhz\n1,a,rx,3363,50,30,90,2000\n")
    assert sbreport.load_telemetry_peaks(old)["a"]["rx"]["gtt_peak_mb"] is None


def test_draft06_plan():
    from tb250distill.specbench import configs as sbc
    ref, cfgs = sbc.get_plan("draft06")
    names = [c.name for c in cfgs]
    assert ref == "t8q2_rx" and len(names) == len(set(names))
    for c in cfgs:
        cmd = sbc.build_cmd(c)
        if c.draft_model:
            assert "qwen3-0.6b" in Path(c.draft_model).name and "-md" in cmd
