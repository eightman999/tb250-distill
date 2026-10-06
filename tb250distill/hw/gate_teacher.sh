#!/usr/bin/env bash
# RX 6400 Teacher gate: llama-server(Vulkan0) を起動し、N 分間 /completion を叩き続けて
# 成功率・tokens/s・amdgpu 温度を記録する。tb250 上で実行する。
#
#   tb250distill/hw/gate_teacher.sh [--minutes N] [--port P] [--out DIR] [--model GGUF]
#
# 既に llama-server が動いている（RX6400 を Teacher が使用中）場合は何も起動せず、
# status=SKIPPED として記録して終了する（--force-run で無視可。他の担当の邪魔になるので通常は使わない）。
#
# 環境変数（動作確認用の上書き）:
#   LLAMA_DEVICE (既定 Vulkan0) / LLAMA_NGL (既定 99) / LLAMA_CTX (既定 1024)
#   LLAMA_EXTRA_ARGS (llama-server への追加引数) / LLAMA_BIN_DIR (既定 ~/bench/llama-bin/llama-b11384) / AMD_PCI (既定 0000:03:00.0 = RX6400)
# 結果: <out>/gate_teacher.json, gate_teacher_requests.jsonl, gate_teacher_server.log
set -uo pipefail

MINUTES=10
PORT=18090
OUT=runs/hw
MODEL="$HOME/bench/models/Qwen3-1.7B-Q4_K_M.gguf"
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --minutes) MINUTES="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --force-run) FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

BIN_DIR="${LLAMA_BIN_DIR:-$HOME/bench/llama-bin/llama-b11384}"
DEVICE="${LLAMA_DEVICE:-Vulkan0}"
NGL="${LLAMA_NGL:-99}"
CTX="${LLAMA_CTX:-1024}"
AMD_PCI="${AMD_PCI:-0000:03:00.0}"
mkdir -p "$OUT"
SUMMARY="$OUT/gate_teacher.json"
REQLOG="$OUT/gate_teacher_requests.jsonl"
SRVLOG="$OUT/gate_teacher_server.log"

# ---- 既存 llama-server があればスキップ
EXISTING="$(pgrep -a llama-server | grep -v pgrep || true)"
if [ -n "$EXISTING" ] && [ "$FORCE" != 1 ]; then
  EXISTING_JSON="$(printf '%s' "$EXISTING" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip().splitlines()))')"
  cat > "$SUMMARY" <<EOF
{"status": "SKIPPED", "reason": "llama-server already running (RX6400 in use by Teacher); not started to avoid interference",
 "running_processes": $EXISTING_JSON, "checked_at": "$(date -Iseconds)", "minutes_requested": $MINUTES}
EOF
  echo "SKIPPED: llama-server already running:"; echo "$EXISTING"
  exit 0
fi
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
  echo "port $PORT already in use" >&2
  echo "{\"status\": \"FAIL\", \"reason\": \"port $PORT in use\"}" > "$SUMMARY"
  exit 1
fi

export LD_LIBRARY_PATH="$BIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json   # NVIDIA 390 の Vulkan ICD は落ちるので必須

: > "$SRVLOG"
"$BIN_DIR/llama-server" -m "$MODEL" --device "$DEVICE" -ngl "$NGL" -c "$CTX" -np 1 \
  --host 127.0.0.1 --port "$PORT" ${LLAMA_EXTRA_ARGS:-} >> "$SRVLOG" 2>&1 &
SRV_PID=$!
cleanup() { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

export GT_PORT="$PORT" GT_MINUTES="$MINUTES" GT_SUMMARY="$SUMMARY" GT_REQLOG="$REQLOG" GT_SRVLOG="$SRVLOG" \
       GT_PID="$SRV_PID" GT_MODEL="$MODEL" GT_DEVICE="$DEVICE" GT_NGL="$NGL" GT_AMD_PCI="$AMD_PCI"

python3 - <<'PY'
import glob, json, os, time, urllib.request, urllib.error, datetime, statistics

port = int(os.environ["GT_PORT"]); minutes = float(os.environ["GT_MINUTES"])
summary = os.environ["GT_SUMMARY"]; reqlog = os.environ["GT_REQLOG"]; srvlog = os.environ["GT_SRVLOG"]
pid = int(os.environ["GT_PID"]); base = "http://127.0.0.1:%d" % port
now = lambda: datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def alive():
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False

def amd_temp():
    p = "/sys/bus/pci/devices/%s/hwmon/hwmon*/temp1_input" % os.environ["GT_AMD_PCI"]
    for f in glob.glob(p):
        try: return int(open(f).read().strip()) / 1000.0
        except Exception: pass
    return None

def amd_sclk():
    try: return open("/sys/bus/pci/devices/%s/pp_dpm_sclk" % os.environ["GT_AMD_PCI"]).read().strip().replace("\n", " | ")
    except Exception as e: return "error: %s" % e

R = {"status": "FAIL", "started": now(), "minutes_requested": minutes, "model": os.environ["GT_MODEL"],
     "device": os.environ["GT_DEVICE"], "ngl": os.environ["GT_NGL"], "port": port, "errors": []}

# ---- 起動待ち
ready = False
t0 = time.time()
while time.time() - t0 < 180:
    if not alive():
        break
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as r:
            if r.status == 200:
                ready = True; break
    except Exception:
        pass
    time.sleep(1)
R["startup_s"] = round(time.time() - t0, 1)
log = open(srvlog, errors="replace").read()
R["server_log_mentions_device"] = [l.strip() for l in log.splitlines()
    if ("Vulkan0" in l or "RX 6400" in l or "offloaded" in l)][:6]
if not ready:
    R["errors"].append("server did not become healthy (alive=%s)" % alive())
    R["finished"] = now(); json.dump(R, open(summary, "w"), indent=2); print("FAIL: server not ready"); raise SystemExit(1)

prompts = [
    "Explain in two sentences why the sky is blue.",
    "日本語で、犬と猫の違いを簡潔に説明してください。",
    "List five prime numbers and explain what a prime number is.",
    "Write a short haiku about autumn rain.",
    "What are the main differences between TCP and UDP?",
    "次の文の感情を分類してください: 「今日の発表は本当に最高だった」",
]
end = time.time() + minutes * 60
n_ok = n_fail = 0; tps = []; lat = []; temps = []; toks = 0
f = open(reqlog, "w")
i = 0
while time.time() < end:
    i += 1
    body = json.dumps({"prompt": "%s (#%d)" % (prompts[i % len(prompts)], i), "n_predict": 96, "temperature": 0.7,
                       "seed": i, "cache_prompt": False, "stream": False}).encode()
    rec = {"t": now(), "i": i}
    ts = time.time()
    try:
        req = urllib.request.Request(base + "/completion", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        dt = time.time() - ts
        tm = d.get("timings", {})
        n = tm.get("predicted_n", d.get("tokens_predicted", 0))
        sp = tm.get("predicted_per_second") or (n / dt if dt > 0 else 0)
        ok = n > 0 and bool(d.get("content", "").strip() or n > 0)
        rec.update(ok=ok, latency_s=round(dt, 3), tokens=n, tok_per_s=round(sp, 2),
                   prompt_per_s=round(tm.get("prompt_per_second", 0) or 0, 1))
        if ok: n_ok += 1; tps.append(sp); lat.append(dt); toks += n
        else: n_fail += 1
    except Exception as e:
        n_fail += 1
        rec.update(ok=False, error="%s: %s" % (type(e).__name__, str(e)[:200]))
        if not alive():
            rec["server_died"] = True
    t = amd_temp()
    rec["amd_temp_c"] = t
    if t is not None: temps.append(t)
    if i % 10 == 1: rec["amd_sclk"] = amd_sclk()
    f.write(json.dumps(rec) + "\n"); f.flush()
    if rec.get("server_died"):
        R["errors"].append("server died during test"); break
f.close()

total = n_ok + n_fail
R.update(finished=now(), requests=total, ok=n_ok, failed=n_fail,
         success_rate=round(n_ok / total, 4) if total else 0.0,
         tokens_generated=toks,
         tok_per_s_mean=round(statistics.mean(tps), 2) if tps else None,
         tok_per_s_min=round(min(tps), 2) if tps else None,
         latency_s_mean=round(statistics.mean(lat), 3) if lat else None,
         amd_temp_c_max=max(temps) if temps else None, amd_temp_c_mean=round(statistics.mean(temps), 1) if temps else None,
         server_alive_end=alive(), amd_sclk_end=amd_sclk())
if total and R["success_rate"] >= 0.99 and R["server_alive_end"] and not R["errors"]:
    R["status"] = "PASS"
json.dump(R, open(summary, "w"), indent=2, ensure_ascii=False)
print("%s: %d req, ok=%d fail=%d, %.1f tok/s mean, amd temp max=%s C" % (
    R["status"], total, n_ok, n_fail, R["tok_per_s_mean"] or 0, R["amd_temp_c_max"]))
raise SystemExit(0 if R["status"] == "PASS" else 1)
PY
RC=$?
exit $RC
