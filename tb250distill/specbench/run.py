"""llama.cpp 異種 GPU 投機的デコードベンチの実行。

  python -m tb250distill.specbench.run --plan main --out runs/specbench/<名前> [--only a,b] [--reps 2]
        [--n-predict 128] [--port 18190] [--llama-dir DIR] [--load-timeout 300] [--resume] [--dry-run]
  python -m tb250distill.specbench.run --plan main --list

条件を固定するための方針:
  - config ごとに llama-server を起動し直す（-fit off、-np 1、-c 1024）。起動は start_new_session=True、停止は
    PID 指定の SIGTERM -> 待機 -> SIGKILL（pkill -f / pgrep -f は使わない）。他のプロセスの llama-server は殺さない。
  - GPU を共有しないよう、起動前に他の llama-server プロセスが居れば中止する。デバイス名も確認する
    （Vulkan0 が RX 6400、Vulkan1 が WX 2100 でなければ中止）。
  - /completion は temperature 0・seed 42・cache_prompt false・ignore_eos true で生成長を固定する。
  - 温度が 88 ℃ 以上なら 78 ℃ 未満まで request を待つ。
  - request ごとに RX 6400 / WX 2100 の電力を台形積分して energy_j を出す（0.5 秒ごとの sysfs 読み取り）。

出力（<out>/）: plan.json, env.json, results.jsonl, outputs.jsonl, status.jsonl, telemetry.csv,
  server-<config>.log, summary.json, summary.md
必須環境変数 VK_ICD_FILENAMES は build_env が設定する。HTTP は標準ライブラリ（urllib）だけを使う。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..hw import amdgpu
from . import configs as C
from . import report as R
from .prompts import PROMPTS

CARD_NAMES = {"rx": "RX 6400", "wx": "WX 2100"}     # card key -> デバイス名（identify_card の照合に使う）
TEMP_HI = 88.0
TEMP_LO = 78.0


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, obj: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def git_commit() -> str:
    """coordinator の決め方（TB250_GIT_COMMIT -> git rev-parse -> uncommitted-<hash>）に合わせる。"""
    try:
        from ..coordinator import git_commit_value
        return git_commit_value()
    except Exception:  # noqa: BLE001
        return os.environ.get("TB250_GIT_COMMIT", "unknown")


def install_sigterm_as_interrupt():
    """SIGTERM を KeyboardInterrupt として扱う（finally でサーバを止めて終了する）。メインスレッド以外では何もしない。"""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def _handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    prev = signal.signal(signal.SIGTERM, _handler)
    return lambda: signal.signal(signal.SIGTERM, prev)


# --------------------------------------------------------------------------------------
# HTTP（標準ライブラリのみ）
# --------------------------------------------------------------------------------------

def http_json(url: str, payload: dict | None = None, timeout: float = 10.0):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def health(base_url: str, timeout: float = 3.0) -> bool:
    try:
        status, _ = http_json(base_url + "/health", timeout=timeout)
        return status == 200
    except Exception:  # noqa: BLE001（ロード中の 503 や接続拒否は未 ready 扱い）
        return False


def completion(base_url: str, prompt: str, n_predict: int, timeout: float) -> dict:
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 42, "cache_prompt": False,
               "ignore_eos": True, "stream": False}
    status, body = http_json(base_url + "/completion", payload, timeout=timeout)
    if status != 200:
        raise RuntimeError(f"/completion status {status}")
    return body


# --------------------------------------------------------------------------------------
# サーバプロセス
# --------------------------------------------------------------------------------------

class ServerProc:
    """llama-server（または差し替えた偽サーバ）の子プロセス。停止は PID 指定の SIGTERM -> 待機 -> SIGKILL。"""

    def __init__(self, cmd: list[str], env: dict, log_path):
        self.cmd, self.env, self.log_path = cmd, env, Path(log_path)
        self.proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdin=subprocess.DEVNULL, stdout=self._log,
                                     stderr=subprocess.STDOUT, start_new_session=True)

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc else None

    def poll(self) -> int | None:
        return self.proc.poll() if self.proc else None

    def stop(self, timeout: float = 20.0) -> None:
        p = self.proc
        try:
            if p is not None and p.poll() is None:
                try:
                    os.kill(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        os.kill(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None


def wait_healthy(base_url: str, server: ServerProc, timeout: float, interval: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep) -> str:
    """"ok" / "exited"（health 待ち中にプロセスが落ちた）/ "timeout"。"""
    t0 = time.monotonic()
    while True:
        if server.poll() is not None:
            return "exited"
        if health(base_url):
            return "ok"
        if time.monotonic() - t0 >= timeout:
            return "timeout"
        sleep(interval)


def tail_lines(path, n: int = 40) -> list[str]:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


# --------------------------------------------------------------------------------------
# 起動前チェック
# --------------------------------------------------------------------------------------

def parse_devices(text: str) -> dict[str, str]:
    """`llama-server --list-devices` の出力から {"Vulkan0": "AMD Radeon RX 6400 (...)"} を作る。"""
    import re
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(Vulkan\d+):\s*(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def verify_devices(text: str) -> list[str]:
    """Vulkan0 が RX 6400、Vulkan1 が WX 2100 でなければ問題を返す。"""
    devs = parse_devices(text)
    problems = []
    for key, want in (("Vulkan0", CARD_NAMES["rx"]), ("Vulkan1", CARD_NAMES["wx"])):
        if key not in devs:
            problems.append(f"{key} が --list-devices に無い（期待: {want}）")
        elif want.lower() not in devs[key].lower():
            problems.append(f"{key} は {devs[key]!r}（期待: {want}）")
    return problems


def list_devices(llama_dir: str, timeout: float = 120.0) -> str:
    r = subprocess.run([str(Path(llama_dir) / "llama-server"), "--list-devices"], env=C.build_env(llama_dir),
                       capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return (r.stdout or "") + (r.stderr or "")


def server_version(llama_dir: str, timeout: float = 60.0) -> str:
    try:
        r = subprocess.run([str(Path(llama_dir) / "llama-server"), "--version"], env=C.build_env(llama_dir),
                           capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return ((r.stdout or "") + (r.stderr or "")).strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {type(e).__name__}: {e}"


def find_llama_servers(proc_root: str = "/proc") -> list[tuple[int, str]]:
    """動作中の llama-server プロセス [(pid, 説明)]。/proc の comm / exe で判定（pgrep -f は使わない）。
    /proc が無い環境（macOS）では ps の comm で判定する。自分自身の pid は除く。"""
    me = os.getpid()
    found: list[tuple[int, str]] = []
    root = Path(proc_root)
    if root.is_dir():
        for d in root.iterdir():
            if not d.name.isdigit() or int(d.name) == me:
                continue
            comm = ""
            exe = ""
            try:
                comm = (d / "comm").read_text().strip()
            except OSError:
                pass
            try:
                exe = os.readlink(d / "exe")
            except OSError:
                pass
            if comm == "llama-server" or Path(exe.replace(" (deleted)", "")).name == "llama-server":
                found.append((int(d.name), exe or comm))
        return sorted(found)
    try:
        r = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != me and Path(parts[1]).name == "llama-server":
                found.append((int(parts[0]), parts[1]))
    except Exception:  # noqa: BLE001
        pass
    return sorted(found)


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 直前の config の接続が TIME_WAIT で残っていても llama-server は bind できる（SO_REUSEADDR）ので、同じ条件で判定する
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


@dataclass
class Hooks:
    """起動前チェックの差し替え口（テスト用）。"""
    list_devices: Callable[[str], str] = list_devices
    server_version: Callable[[str], str] = server_version
    find_servers: Callable[[], list] = find_llama_servers
    port_free: Callable[[int], bool] = port_is_free


def preflight(cfgs: list[C.BenchConfig], llama_dir: str, port: int, hooks: Hooks | None = None,
              check_icd: bool = True) -> tuple[list[str], dict]:
    """(問題のリスト, env.json に入れる情報)。問題があれば実行しない。"""
    hooks = hooks or Hooks()
    problems: list[str] = []
    info: dict = {"llama_dir": str(llama_dir)}
    if check_icd and not Path(C.ICD).exists():
        problems.append(f"Vulkan ICD が無い: {C.ICD}（NVIDIA ICD を避けるため必須）")
    exe = Path(llama_dir) / "llama-server"
    if not exe.exists():
        problems.append(f"llama-server が無い: {exe}")
    models = sorted({m for c in cfgs for m in c.models()})
    info["models"] = {}
    for m in models:
        if Path(m).exists():
            info["models"][m] = Path(m).stat().st_size
        else:
            problems.append(f"モデルが無い: {m}")
    if not hooks.port_free(port):
        problems.append(f"ポート {port} が使用中")
    others = hooks.find_servers()
    info["other_llama_servers"] = [list(o) for o in others]
    if others:
        problems.append("他の llama-server が動作中（GPU 共有を避けるため中止）: "
                        + ", ".join(f"pid {p} ({d})" for p, d in others))
    if exe.exists():
        info["version"] = hooks.server_version(str(llama_dir))
        try:
            text = hooks.list_devices(str(llama_dir))
            info["list_devices"] = text
            problems += verify_devices(text)
        except Exception as e:  # noqa: BLE001
            problems.append(f"--list-devices に失敗: {type(e).__name__}: {e}")
    return problems, info


# --------------------------------------------------------------------------------------
# テレメトリ・エネルギー
# --------------------------------------------------------------------------------------

def integrate_power(samples: list[tuple[float, float | None]], t0: float, t1: float, max_gap: float = 2.0):
    """電力サンプル [(t, W)] を [t0, t1] で台形積分して J を返す。
    窓の両端は隣り合うサンプルの線形補間で求める。端がサンプルの外側なら最寄りの値で代用し、
    最寄りサンプルまで max_gap 秒より離れていれば None（信頼できない）。サンプルが無い・窓が逆順なら None。"""
    pts = sorted((t, p) for t, p in samples if p is not None)
    if not pts or t1 < t0:
        return None
    if t1 == t0:
        return 0.0
    times = [t for t, _ in pts]

    def val(t: float):
        i = bisect_left(times, t)
        if i == 0:
            return pts[0][1] if pts[0][0] - t <= max_gap else None
        if i >= len(pts):
            return pts[-1][1] if t - pts[-1][0] <= max_gap else None
        (ta, pa), (tb, pb) = pts[i - 1], pts[i]
        return pa if tb == ta else pa + (pb - pa) * (t - ta) / (tb - ta)

    v0, v1 = val(t0), val(t1)
    if v0 is None or v1 is None:
        return None
    knots = [(t0, v0)] + [(t, p) for t, p in pts if t0 < t < t1] + [(t1, v1)]
    return sum((knots[i + 1][0] - knots[i][0]) * (knots[i + 1][1] + knots[i][1]) / 2 for i in range(len(knots) - 1))


def read_card(device_dir: str | None) -> dict:
    """read_amd に GTT 使用量（gtt_mb）を足したもの。VRAM に載らずシステムメモリへはみ出した量を見るため
    （mem_info_vram_used には含まれない。main1 の draft 同居構成で 1 GB はみ出していた）。"""
    out = amdgpu.read_amd(device_dir)
    out["gtt_mb"] = None
    if device_dir:
        try:
            out["gtt_mb"] = int(Path(device_dir, "mem_info_gtt_used").read_text().strip()) / (1024 * 1024)
        except (OSError, ValueError):
            pass
    return out


class Telemetry:
    """RX 6400 / WX 2100 を interval 秒ごとに read_card で読み、telemetry.csv に保存するバックグラウンドスレッド。
    cards: {"rx": device_dir or None, "wx": ...}。読めない値は None（CSV では空欄）。"""
    COLS = ["t", "config", "card", "vram_mb", "temp_c", "power_w", "busy_pct", "sclk_mhz", "gtt_mb"]

    def __init__(self, cards: dict[str, str | None], csv_path=None, interval: float = 0.5,
                 sampler: Callable[[str | None], dict] = read_card, clock: Callable[[], float] = time.time):
        self.cards, self.csv_path, self.interval, self.sampler, self.clock = cards, csv_path, interval, sampler, clock
        self.config = ""
        self.samples: dict[str, list[tuple[float, dict]]] = {k: [] for k in cards}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fh = None
        self._w = None

    def set_config(self, name: str) -> None:
        self.config = name

    def sample_once(self) -> None:
        t = self.clock()
        for card, ddir in self.cards.items():
            try:
                v = self.sampler(ddir)
            except Exception:  # noqa: BLE001
                v = {}
            v = {k: v.get(k) for k in self.COLS[3:]}
            with self._lock:
                self.samples[card].append((t, v))
                if self._w and any(x is not None for x in v.values()):
                    self._w.writerow([f"{t:.3f}", self.config, card] + ["" if v[k] is None else f"{v[k]:.6g}" for k in self.COLS[3:]])
        if self._fh:
            with self._lock:
                self._fh.flush()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self.csv_path:
            Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
            new = not Path(self.csv_path).exists()
            self._fh = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._w = csv.writer(self._fh)
            if new:
                self._w.writerow(self.COLS)
        self._thread = threading.Thread(target=self._loop, name="specbench-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None
                self._w = None

    def latest(self, card: str, key: str):
        with self._lock:
            for _, v in reversed(self.samples.get(card, [])):
                if v.get(key) is not None:
                    return v[key]
        return None

    def window(self, card: str, key: str, t0: float, t1: float, pad: float = 2.0) -> list[tuple[float, float | None]]:
        with self._lock:
            return [(t, v.get(key)) for t, v in self.samples.get(card, []) if t0 - pad <= t <= t1 + pad]

    def energy(self, t0: float, t1: float) -> dict:
        """[t0, t1] の消費エネルギー J（card ごと + 合計。どれかが None なら合計も None）と最高温度。"""
        e = {c: integrate_power(self.window(c, "power_w", t0, t1), t0, t1) for c in self.cards}
        total = None if any(v is None for v in e.values()) or not e else sum(e.values())
        temps = {}
        for c in self.cards:
            ts = [v for t, v in self.window(c, "temp_c", t0, t1, pad=0.0) if v is not None]
            temps[c] = max(ts) if ts else None
        return {"energy": e, "total": total, "temp_max": temps}


def temp_guard(tel: Telemetry | None, hi: float = TEMP_HI, lo: float = TEMP_LO, poll: float = 5.0,
               max_wait: float = 1800.0, sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.monotonic, log: Callable[[str], None] = print) -> float:
    """いずれかの温度が hi 以上なら lo 未満になるまで待つ。待った秒数を返す（温度が読めなければ 0）。"""
    if tel is None:
        return 0.0

    def hottest():
        ts = [tel.latest(c, "temp_c") for c in tel.cards]
        ts = [t for t in ts if t is not None]
        return max(ts) if ts else None

    t = hottest()
    if t is None or t < hi:
        return 0.0
    log(f"[temp] {t:.0f}C >= {hi:.0f}C。{lo:.0f}C 未満まで待機")
    t0 = clock()
    while True:
        sleep(poll)
        t = hottest()
        if t is None or t < lo:
            break
        if clock() - t0 > max_wait:
            log(f"[temp] {max_wait:.0f}s 待っても {t:.0f}C。続行する")
            break
    return clock() - t0


# --------------------------------------------------------------------------------------
# 実行本体
# --------------------------------------------------------------------------------------

@dataclass
class RunOptions:
    out: Path
    reps: int = 2
    n_predict: int = 128
    port: int = C.DEFAULT_PORT
    llama_dir: str = str(C.LLAMA_DIR)
    load_timeout: float = 300.0
    request_timeout: float = 900.0
    settle_s: float = 5.0                       # config 間で VRAM が解放されるのを待つ
    resume: bool = False
    retry_failed: bool = False
    cmd_builder: Callable = C.build_cmd         # (cfg, llama_dir, port) -> argv。テストで差し替える
    env_builder: Callable = C.build_env
    health_interval: float = 1.0
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = print


class RunState:
    def __init__(self, reference: str, results: JsonlWriter, outputs: JsonlWriter, statuses: JsonlWriter):
        self.reference = reference
        self.results, self.outputs, self.statuses = results, outputs, statuses
        self.done: set[tuple[str, int, str]] = set()
        self.ref_hash: dict[str, tuple[int, str]] = {}


def make_record(cfg: C.BenchConfig, rep: int, prompt: dict, n_predict: int, t0: float, t1: float, wall_s: float,
                resp: dict | None, error: str | None, tel: Telemetry | None, state: RunState, temp_wait_s: float) -> tuple[dict, str]:
    """results.jsonl の 1 行と content を作る。"""
    rec: dict = {"config": cfg.name, "rep": rep, "prompt_id": prompt["id"], "category": prompt["category"],
                 "ok": error is None, "error": error, "t_start": round(t0, 3), "t_end": round(t1, 3),
                 "wall_s": round(wall_s, 4), "n_predict": n_predict, "temp_wait_s": round(temp_wait_s, 1)}
    content = ""
    if resp is not None:
        content = resp.get("content") or ""
        timings = resp.get("timings") or {}
        tp = resp.get("tokens_predicted", timings.get("predicted_n"))
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        rec.update(tokens_predicted=tp, tokens_evaluated=resp.get("tokens_evaluated"),
                   stop_type=resp.get("stop_type"), sha256=sha, timings=timings,
                   draft_n=int(timings.get("draft_n") or 0), draft_n_accepted=int(timings.get("draft_n_accepted") or 0),
                   wall_tps=(tp / wall_s) if tp and wall_s > 0 else None)
        pid = prompt["id"]
        if cfg.name == state.reference and pid not in state.ref_hash:
            state.ref_hash[pid] = (rep, sha)
            rec["match_ref"] = True
        else:
            rec["match_ref"] = (sha == state.ref_hash[pid][1]) if pid in state.ref_hash else None
    if tel is not None and error is None:
        en = tel.energy(t0, t1)
        rec.update(energy_j_rx=en["energy"].get("rx"), energy_j_wx=en["energy"].get("wx"), energy_j=en["total"],
                   temp_max_rx=en["temp_max"].get("rx"), temp_max_wx=en["temp_max"].get("wx"))
    return rec, content


def run_config(cfg: C.BenchConfig, opts: RunOptions, prompts: list[dict], tel: Telemetry | None,
               state: RunState) -> dict:
    """1 config を実行して status 辞書を返す（status.jsonl にも書く）。サーバは必ず止める。"""
    base_url = f"http://127.0.0.1:{opts.port}"
    log_path = opts.out / f"server-{cfg.name}.log"
    cmd = opts.cmd_builder(cfg, opts.llama_dir, opts.port)
    status: dict = {"config": cfg.name, "status": None, "started_at": now_iso(), "cmd": cmd, "describe": C.describe(cfg)}
    if tel is not None:
        tel.set_config(cfg.name)
    server = ServerProc(cmd, opts.env_builder(opts.llama_dir), log_path)
    t_launch = time.monotonic()
    try:
        server.start()
        status["pid"] = server.pid
        res = wait_healthy(base_url, server, opts.load_timeout, opts.health_interval, opts.sleep)
        status["load_s"] = round(time.monotonic() - t_launch, 1)
        if res != "ok":
            status["status"] = "load_failed" if res == "exited" else "load_timeout"
            status["exit_code"] = server.poll()
            return status
        # warmup（記録しない。Vulkan のシェーダ・初回確保を済ませる）
        try:
            completion(base_url, prompts[0]["prompt"], min(32, opts.n_predict), opts.request_timeout)
        except Exception as e:  # noqa: BLE001
            opts.log(f"[{cfg.name}] warmup 失敗: {type(e).__name__}: {e}")
        consecutive_fail = 0
        status["status"] = "ok"
        for rep in range(opts.reps):
            for p in prompts:
                if (cfg.name, rep, p["id"]) in state.done:
                    continue
                if server.poll() is not None:
                    status["status"] = "crashed"
                    status["exit_code"] = server.poll()
                    return status
                waited = temp_guard(tel, sleep=opts.sleep, log=opts.log)
                t0, c0 = time.time(), time.perf_counter()
                resp = err = None
                try:
                    resp = completion(base_url, p["prompt"], opts.n_predict, opts.request_timeout)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                wall = time.perf_counter() - c0
                t1 = time.time()
                rec, content = make_record(cfg, rep, p, opts.n_predict, t0, t1, wall, resp, err, tel, state, waited)
                if err is None:
                    state.outputs.append({"config": cfg.name, "rep": rep, "prompt_id": p["id"], "content": content})
                    state.done.add((cfg.name, rep, p["id"]))
                    consecutive_fail = 0
                    tm = rec.get("timings") or {}
                    acc = f" accept {rec['draft_n_accepted']}/{rec['draft_n']}" if rec["draft_n"] else ""
                    opts.log(f"[{cfg.name}] rep{rep} {p['id']:<10} {tm.get('predicted_per_second', float('nan')):6.2f} tok/s "
                             f"wall {wall:5.1f}s{acc}")
                else:
                    consecutive_fail += 1
                    opts.log(f"[{cfg.name}] rep{rep} {p['id']} 失敗: {err}")
                state.results.append(rec)
                if consecutive_fail >= 2:
                    status["status"] = "request_failed"
                    status["error"] = err
                    return status
        return status
    except KeyboardInterrupt:
        status["status"] = "interrupted"
        raise
    except Exception as e:  # noqa: BLE001
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        return status
    finally:
        server.stop()
        status["finished_at"] = now_iso()
        if status["status"] != "ok":
            status["log_tail"] = tail_lines(log_path, 40)
        state.statuses.append(status)


def load_resume_state(state: RunState, out: Path) -> dict[str, dict]:
    """既存の results.jsonl から完了済み (config, rep, prompt) と reference の hash を、status.jsonl から最新 status を読む。"""
    rows = R.load_jsonl(out / "results.jsonl")
    for r in rows:
        if r.get("ok"):
            state.done.add((r["config"], int(r["rep"]), r["prompt_id"]))
    state.ref_hash = R.reference_hashes(rows, state.reference)
    last: dict[str, dict] = {}
    for s in R.load_jsonl(out / "status.jsonl"):
        last[s["config"]] = s
    return last


def run_bench(plan: str, reference: str, cfgs: list[C.BenchConfig], opts: RunOptions,
              prompts: list[dict] | None = None, tel: Telemetry | None = None, env_info: dict | None = None) -> dict:
    """プランを順に実行して summary を返す。reference は先頭で実行する。"""
    prompts = prompts or PROMPTS
    out = opts.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.jsonl").exists() and not opts.resume:
        raise RuntimeError(f"{out / 'results.jsonl'} が既にある。続きから再開するなら --resume、新規なら別の --out を指定する")
    state = RunState(reference, JsonlWriter(out / "results.jsonl"), JsonlWriter(out / "outputs.jsonl"),
                     JsonlWriter(out / "status.jsonl"))
    last_status = load_resume_state(state, out) if opts.resume else {}
    prior = (R.load_json(out / "plan.json", {}) or {}).get("configs", []) if opts.resume else []
    merged = {c["name"]: c for c in prior}
    merged.update({c.name: c.to_dict() for c in cfgs})          # 再開時は過去に実行した config も残す
    write_json(out / "plan.json", {
        "plan": plan, "reference": reference, "configs": list(merged.values()),
        "args": {"reps": opts.reps, "n_predict": opts.n_predict, "port": opts.port, "llama_dir": opts.llama_dir,
                 "load_timeout": opts.load_timeout, "prompts": [p["id"] for p in prompts]},
    })
    env = {"started_at": now_iso(), "host": platform.node(), "platform": platform.platform(), "python": sys.version,
           "git_commit": git_commit(), "icd": C.ICD}
    env.update(env_info or {})
    write_json(out / "env.json", env)

    ordered = sorted(cfgs, key=lambda c: c.name != reference)      # reference を先頭へ（他は元の順序）
    if reference not in {c.name for c in cfgs}:
        opts.log(f"[warn] reference {reference} が実行対象に無い。match_ref・speedup は過去の結果がある場合のみ出る")
    if tel is not None:
        tel.start()
    try:
        for i, cfg in enumerate(ordered):
            prev = last_status.get(cfg.name)
            if all((cfg.name, r, p["id"]) in state.done for r in range(opts.reps) for p in prompts):
                opts.log(f"[{cfg.name}] 完了済み。スキップ")
                continue
            if prev and prev.get("status") in ("load_failed", "load_timeout") and not opts.retry_failed:
                opts.log(f"[{cfg.name}] 前回 {prev['status']}。スキップ（--retry-failed で再試行）")
                continue
            opts.log(f"=== [{i + 1}/{len(ordered)}] {cfg.name}: {C.describe(cfg)}")
            st = run_config(cfg, opts, prompts, tel, state)
            opts.log(f"=== {cfg.name}: {st['status']}（load {st.get('load_s')}s）")
            if i + 1 < len(ordered):
                opts.sleep(opts.settle_s)               # VRAM の解放待ち
    finally:
        if tel is not None:
            tel.stop()
        R.write_summary(out, reference)
    return R.summarize(out, reference)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def print_plan(plan: str, reference: str, cfgs: list[C.BenchConfig]) -> None:
    print(f"plan {plan}（reference: {reference}、{len(cfgs)} configs）")
    for c in cfgs:
        mark = "*" if c.name == reference else " "
        print(f"{mark} {c.name:<26} {C.describe(c)}")
        if c.note:
            print(f"    {c.note}")


def print_dry_run(plan: str, reference: str, cfgs: list[C.BenchConfig], opts: RunOptions) -> None:
    print_plan(plan, reference, cfgs)
    n_req = opts.reps * len(PROMPTS)
    print(f"\n各 config: warmup 1 + {opts.reps} reps x {len(PROMPTS)} prompts = {n_req} requests（n_predict={opts.n_predict}）")
    print(f"env: VK_ICD_FILENAMES={C.ICD} LD_LIBRARY_PATH={opts.llama_dir}\n")
    for c in cfgs:
        print(f"[{c.name}]")
        print("  " + shlex.join(C.build_cmd(c, opts.llama_dir, opts.port)))


def build_telemetry(out: Path, interval: float) -> Telemetry:
    cards = {}
    for key, name in CARD_NAMES.items():
        ident = amdgpu.identify_card(name)
        cards[key] = ident["device_dir"]
        if ident["device_dir"] is None:
            print(f"[warn] {name} の sysfs card を特定できない（テレメトリは None）: {ident['reason']}")
    return Telemetry(cards, out / "telemetry.csv", interval)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="llama.cpp 異種 GPU 投機的デコードベンチ")
    ap.add_argument("--plan", default="main", choices=sorted(C.PLANS))
    ap.add_argument("--out", default=None, help="出力ディレクトリ（例 runs/specbench/<名前>）。--list / --dry-run 以外は必須")
    ap.add_argument("--only", default=None, help="カンマ区切りの config 名で絞り込む")
    ap.add_argument("--list", action="store_true", help="プランを表示して終了")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--port", type=int, default=C.DEFAULT_PORT)
    ap.add_argument("--llama-dir", default=str(C.LLAMA_DIR))
    ap.add_argument("--load-timeout", type=float, default=300.0, help="health が ok になるまでの上限秒")
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--settle", type=float, default=5.0, help="config 間の VRAM 解放待ち秒")
    ap.add_argument("--sample-interval", type=float, default=0.5, help="テレメトリの読み取り間隔秒")
    ap.add_argument("--resume", action="store_true", help="results.jsonl にある完了済み (config, rep, prompt) をスキップ")
    ap.add_argument("--retry-failed", action="store_true", help="--resume 時、前回 load_failed/load_timeout の config も再試行")
    ap.add_argument("--dry-run", action="store_true", help="各 config のコマンドラインだけ表示（チェック・起動なし）")
    args = ap.parse_args(argv)

    try:
        reference, cfgs = C.get_plan(args.plan, [s for s in (args.only or "").split(",") if s] or None)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.list:
        print_plan(args.plan, reference, cfgs)
        return 0
    opts = RunOptions(out=Path(args.out or "."), reps=args.reps, n_predict=args.n_predict, port=args.port,
                      llama_dir=args.llama_dir, load_timeout=args.load_timeout, request_timeout=args.request_timeout,
                      settle_s=args.settle, resume=args.resume, retry_failed=args.retry_failed)
    if args.dry_run:
        print_dry_run(args.plan, reference, cfgs, opts)
        return 0
    if not args.out:
        print("error: --out が必要", file=sys.stderr)
        return 2

    problems, info = preflight(cfgs, args.llama_dir, args.port)
    if problems:
        print("起動前チェックに失敗:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    restore = install_sigterm_as_interrupt()
    try:
        tel = build_telemetry(opts.out, args.sample_interval)
        summary = run_bench(args.plan, reference, cfgs, opts, tel=tel, env_info=info)
    except KeyboardInterrupt:
        print("\n中断。サーバは停止済み。--resume で続きから再開できる", file=sys.stderr)
        return 130
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        restore()
    print(R.render_md(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
