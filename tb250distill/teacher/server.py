"""RX 6400（Vulkan0）上の llama-server の起動・停止・ヘルスチェック。

必須環境変数: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
（NVIDIA 390 の Vulkan ICD が llama.cpp を stack smashing で落とすため。設定しないと起動を拒否する）。
llama-server は 127.0.0.1:18080 だけで待ち受け、setsid で親から切り離す（nohup 相当）。

CLI:
  python -m tb250distill.teacher.server start [--ctx 288] [--model PATH] [--port 18080] [--parallel 1]
  python -m tb250distill.teacher.server status | health | stop | restart
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

HOME = Path.home()
LLAMA_DIR = HOME / "bench" / "llama-bin" / "llama-b11384"
DEFAULT_MODEL = HOME / "bench" / "models" / "Qwen3-1.7B-Q4_K_M.gguf"
ICD = "/usr/share/vulkan/icd.d/radeon_icd.json"
DEFAULT_PORT = 18080
RUN_DIR = Path(os.environ.get("TB250_RUNS", Path(__file__).resolve().parents[2] / "runs")) / "teacher"


@dataclass
class ServerConfig:
    model: str = str(DEFAULT_MODEL)
    port: int = DEFAULT_PORT
    host: str = "127.0.0.1"
    device: str = "Vulkan0"
    ngl: int = 99
    max_prompt_tokens: int = 256          # scorer 側の prompt 上限（256 → 512 → 1024 へ拡張可能）
    ctx_margin: int = 32                  # n_predict=1 とテンプレートの余裕
    parallel: int = 1
    threads: int = 2
    batch: int = 512
    llama_dir: str = str(LLAMA_DIR)

    @property
    def ctx_size(self) -> int:
        return self.max_prompt_tokens + self.ctx_margin

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _pid_file() -> Path:
    return RUN_DIR / "server.pid"


def _cfg_file() -> Path:
    return RUN_DIR / "server_config.json"


def build_env(cfg: ServerConfig) -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = cfg.llama_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["VK_ICD_FILENAMES"] = ICD
    return env


def build_cmd(cfg: ServerConfig) -> list[str]:
    # -c は全スロット合計の文脈長。スロットごとに ctx_size を確保する。
    return [
        str(Path(cfg.llama_dir) / "llama-server"),
        "-m", cfg.model,
        "--device", cfg.device,
        "-ngl", str(cfg.ngl),
        "-c", str(cfg.ctx_size * cfg.parallel),
        "-np", str(cfg.parallel),
        "-b", str(cfg.batch),
        "-t", str(cfg.threads),
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--cache-prompt",
    ]


def health(base_url: str = f"http://127.0.0.1:{DEFAULT_PORT}", timeout: float = 3.0) -> bool:
    try:
        r = requests.get(base_url + "/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def read_pid() -> int | None:
    try:
        pid = int(_pid_file().read_text().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def start(cfg: ServerConfig, wait: float = 180.0) -> int:
    """llama-server を起動して /health が ok になるまで待つ。既に ok なら何もしない。pid を返す。"""
    if health(cfg.base_url):
        return read_pid() or -1
    if not Path(ICD).exists():
        raise RuntimeError(f"Vulkan ICD not found: {ICD}（NVIDIA ICD を避けるため必須）")
    if not Path(cfg.model).exists():
        raise FileNotFoundError(cfg.model)
    exe = Path(cfg.llama_dir) / "llama-server"
    if not exe.exists():
        raise FileNotFoundError(exe)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log = open(RUN_DIR / "server.log", "ab")
    proc = subprocess.Popen(
        build_cmd(cfg), env=build_env(cfg), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _pid_file().write_text(str(proc.pid))
    _cfg_file().write_text(json.dumps(asdict(cfg), indent=1))
    t0 = time.time()
    while time.time() - t0 < wait:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}); see {RUN_DIR / 'server.log'}")
        if health(cfg.base_url):
            return proc.pid
        time.sleep(1.0)
    raise TimeoutError("llama-server did not become healthy")


def stop(timeout: float = 20.0) -> bool:
    pid = read_pid()
    if pid is None:
        return False
    os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.3)
    else:
        os.kill(pid, signal.SIGKILL)
    _pid_file().unlink(missing_ok=True)
    return True


def load_config() -> ServerConfig:
    try:
        return ServerConfig(**json.loads(_cfg_file().read_text()))
    except Exception:
        return ServerConfig()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "status", "health", "restart"])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--max-prompt-tokens", type=int, default=256, help="prompt 上限。server ctx = これ + 32")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--device", default="Vulkan0")
    args = ap.parse_args(argv)
    cfg = ServerConfig(model=args.model, port=args.port, max_prompt_tokens=args.max_prompt_tokens,
                       parallel=args.parallel, ngl=args.ngl, device=args.device)
    if args.action in ("stop", "restart"):
        print("stopped" if stop() else "not running (no pid file)")
    if args.action in ("start", "restart"):
        pid = start(cfg)
        print(f"started pid={pid} ctx={cfg.ctx_size} url={cfg.base_url}")
    if args.action == "health":
        ok = health(cfg.base_url)
        print("ok" if ok else "down")
        return 0 if ok else 1
    if args.action == "status":
        print(json.dumps({"pid": read_pid(), "healthy": health(cfg.base_url), "config": asdict(load_config())}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
