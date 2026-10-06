"""Student の KD 学習 CLI。

  python -m tb250distill.student.train --backend cl --device "GT 430" --data data/tok/NAME \
      --run-dir runs/common/gt430 --config common_s --init runs/common/init_common_s.npz \
      --epochs 3 --batch-size 32 --seed 0 [--resume]

再現性の方針:
  - 初期 weight: --init の npz（無ければ seed から決定的に生成して run-dir/init.npz に保存）。全 GPU で同一 npz を使う。
  - sample order・候補 permutation は (seed, epoch) だけから決まる numpy default_rng で生成する
    （デバイス・backend 非依存）。よって「RNG state」= (seed, epoch, epoch 内位置)。
  - np backend は --resume でビット一致で再開できる。cl backend は embedding 勾配の float atomic add の
    加算順が非決定的なので、再開後・再実行で最下位ビット程度の差が出る（学習結果は統計的に同等）。
  - 損失: gold 有り 0.8*KD + 0.2*CE、無し KD のみ（KD = T^2 KL(softmax(t/T)||softmax(s/T)), T=2）。
    --source-loss "jnli:kd=0.2,ce=0.8" で、shard の optional キー `source` が一致する sample の gold 有り重みを置き換える
    （複数は ';' 区切り。gold 無し sample は常に KD のみ。`source` キーが無い shard・指定なしは従来どおり）。
  - AdamW(lr 2e-3, betas 0.9/0.999, eps 1e-8, wd 0.01、biases/emb にも一律適用 = torch.optim.AdamW 既定と同じ)、grad clip 1.0。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from . import model as M
from . import evaluate as E
from . import semfix as SF

CSV_COLS = ["step", "epoch", "train_loss", "kd", "ce", "val_loss", "val_kl", "val_agree", "val_gold_acc",
            "val_brier", "val_ece", "samples_per_s", "tokens_per_s", "step_ms", "vram_mb", "temp_c", "wall_s", "paused_s",
            "power_w", "gpu_busy_pct", "sclk_mhz", "sem_loss"]   # power_w/gpu_busy_pct/sclk_mhz は AMD(sysfs) のみ（NVIDIA では空）。
# sem_loss = Candidate Semantic Distillation の cosine loss（λ 倍する前の値）。--sem-cand-weight 無しでは空
CKPT_FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("tb250.train")


# --------------------------------------------------------------------------------------
# 環境情報
# --------------------------------------------------------------------------------------

def _run(cmd, cwd=None, timeout=20):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:  # noqa: BLE001
        return 127, "", str(e)


def git_info():
    """commit hash（環境変数 TB250_GIT_COMMIT → git → <repo>/GIT_COMMIT の順。coordinator と同じ優先順）と diff。"""
    info = {"commit": None, "source": None, "diff": None, "note": None}
    env = os.environ.get("TB250_GIT_COMMIT")
    if env and env.strip():
        info["commit"], info["source"] = env.strip(), "env TB250_GIT_COMMIT"
    rc, out, _ = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    if rc == 0 and out.strip():
        if info["commit"] is None:
            info["commit"], info["source"] = out.strip(), "git"
        rc2, d, _ = _run(["git", "diff", "HEAD"], cwd=REPO_ROOT)
        if rc2 == 0:
            info["diff"] = d
    else:
        rc3, st, _ = _run(["git", "status", "--short"], cwd=REPO_ROOT)
        if rc3 == 0:
            info["note"] = "git repo に commit が無い（HEAD 未解決）。status:\n" + st
    if info["commit"] is None:
        fp = os.path.join(REPO_ROOT, "GIT_COMMIT")
        if os.path.isfile(fp):
            info["commit"], info["source"] = open(fp).read().strip(), "GIT_COMMIT file"
    return info


class SmiMonitor:
    """GPU の VRAM / 温度（AMD は電力・使用率・sclk も）を --smi-interval 間隔で取得する。

    - NVIDIA: nvidia-smi（PCI bus id 優先、無ければデバイス名で照合）。power/busy/sclk は取らない（None）。
    - AMD(amdgpu): /sys/class/drm/card*/device の sysfs を読む（tb250distill.hw.amdgpu）。対象カードは
      OpenCL の pci_bus_info → 無ければ ID/名前対応表で特定し、特定できなければ値は None のまま（別カードを読まない）。
    電力は時間加重平均（台形則）を積算し、energy/sample の算出に使う。温度ガードの一時停止中は recording=False にして積算しない。
    """

    def __init__(self, be, interval=10.0, drm_root=None):
        self.enabled = getattr(be, "name", "") == "cl"
        self.pci = getattr(be, "pci_bus_id", None)
        self.dev_name = getattr(be, "device_name", "") or ""
        self.interval = interval
        self.last_t = 0.0
        self.vram = None
        self.temp = None
        self.power = None
        self.busy = None
        self.sclk = None
        self.vram_max = None
        self.temp_max = None
        self.power_max = None
        self.err = None
        self.recording = True
        self.power_samples = 0
        self.power_integral_j = 0.0     # ∫ power dt（記録中の区間のみ）
        self.power_time_s = 0.0
        self.busy_sum = 0.0
        self.busy_n = 0
        self._prev = None               # (t, power) 直前の記録サンプル
        self.drm_root = drm_root
        self.kind = None                # "nvidia" | "amdgpu"
        self.ident = None               # amdgpu のカード特定結果（dict）
        self._amd_dir = None
        if self.enabled:
            self.kind = self._detect_kind(be)
            if self.kind == "amdgpu":
                self._identify_amd(be)

    @staticmethod
    def _detect_kind(be):
        dev = getattr(be, "dev", None)
        try:
            vendor = str(getattr(dev, "vendor", "") or "")
        except Exception:  # noqa: BLE001
            vendor = ""
        hay = (vendor + " " + (getattr(be, "device_name", "") or "")).lower()
        if any(k in hay for k in ("nvidia", "geforce", "quadro")):
            return "nvidia"
        if any(k in hay for k in ("amd", "advanced micro", "radeon", "ati ")):
            return "amdgpu"
        return "nvidia"                 # 従来挙動

    def _identify_amd(self, be):
        from tb250distill.hw import amdgpu
        addr, note = (None, "backend has no OpenCL device")
        dev = getattr(be, "dev", None)
        if dev is not None:
            addr, note = amdgpu.opencl_pci_bus_info(dev)
        self.ident = amdgpu.identify_card(self.dev_name, addr, self.drm_root or amdgpu.SYS_DRM, pci_reason=note)
        self._amd_dir = self.ident.get("device_dir")
        if self._amd_dir is None:
            self.err = "amdgpu card を特定できない: %s" % self.ident.get("reason")

    def info(self):
        """config.json / train.log / hardware.json に残す監視設定（特定結果）。"""
        d = {"kind": self.kind, "enabled": self.enabled, "interval_s": self.interval, "opencl_device": self.dev_name}
        if self.kind == "amdgpu":
            d["amdgpu"] = {k: v for k, v in (self.ident or {}).items() if k != "device_dir"}
            d["amdgpu"]["sysfs_device_dir"] = self._amd_dir
            d["power_source"] = "amdgpu sysfs hwmon power1_average（無ければ power1_input）。GPU 単体の値でシステム全体の電力ではない"
        elif self.kind == "nvidia":
            d["source"] = "nvidia-smi (memory.used, temperature.gpu); power/utilization は GeForce+390 では N/A"
        return d

    def poll(self, force=False):
        if not self.enabled:
            return
        now = time.time()
        if not force and now - self.last_t < self.interval:
            return
        self.last_t = now
        if self.kind == "amdgpu":
            self._poll_amd(now)
        else:
            self._poll_nvidia()

    def _poll_amd(self, now):
        if self._amd_dir is None:
            return
        from tb250distill.hw import amdgpu
        v = amdgpu.read_amd(self._amd_dir)
        self.vram, self.temp, self.power, self.busy, self.sclk = (v["vram_mb"], v["temp_c"], v["power_w"],
                                                                  v["busy_pct"], v["sclk_mhz"])
        if self.vram is None and self.temp is None:
            self.err = "sysfs を読めない: %s" % self._amd_dir
        else:
            self.err = None
        if self.vram is not None:
            self.vram_max = max(self.vram_max or 0, self.vram)
        if self.temp is not None:
            self.temp_max = max(self.temp_max or 0, self.temp)
        if self.power is not None:
            self.power_max = max(self.power_max or 0, self.power)
        if not self.recording:
            self._prev = None
            return
        if self.busy is not None:
            self.busy_sum += self.busy
            self.busy_n += 1
        if self.power is not None:
            self.power_samples += 1
            if self._prev is not None:
                dt = now - self._prev[0]
                if dt > 0:
                    self.power_integral_j += (self._prev[1] + self.power) / 2.0 * dt
                    self.power_time_s += dt
            self._prev = (now, self.power)

    def _poll_nvidia(self):
        rc, out, err = _run(["nvidia-smi", "--query-gpu=pci.bus_id,name,memory.used,temperature.gpu",
                             "--format=csv,noheader,nounits"], timeout=15)
        if rc != 0:
            self.err = err.strip()[:200]
            return
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4:
                continue
            bus, name, mem, temp = parts[:4]
            ok = False
            if self.pci:
                ok = bus.lower().endswith(self.pci.lower())
            else:
                short = self.dev_name.replace("GeForce", "").strip().lower()
                ok = bool(short) and short in name.lower()
            if ok:
                try:
                    self.vram = float(mem)
                    self.temp = float(temp)
                except ValueError:
                    continue
                self.vram_max = max(self.vram_max or 0, self.vram)
                self.temp_max = max(self.temp_max or 0, self.temp)
                return

    def power_mean(self):
        """記録区間の時間加重平均電力（W）。区間が無ければ None。"""
        if self.power_time_s > 0:
            return self.power_integral_j / self.power_time_s
        return None

    def restore(self, integral_j, time_s):
        """--resume 時に前プロセスまでの積算を引き継ぐ。"""
        self.power_integral_j += float(integral_j or 0.0)
        self.power_time_s += float(time_s or 0.0)


def _add_to_hardware_json(run_dir, key, value):
    """hardware.json に student 側の追加情報を足す（失敗しても学習は止めない）。"""
    path = os.path.join(run_dir, "hardware.json")
    try:
        with open(path) as f:
            d = json.load(f)
        d[key] = value
        with open(path, "w") as f:
            json.dump(d, f, indent=1, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        log.warning("hardware.json への %s 追記に失敗: %s", key, e)


def energy_summary(gpu, samples, wall_active_s):
    """電力の実測サマリ。energy/sample = 平均電力 x active wall / samples（active = 温度ガード停止を除く wall）。
    電力が取れなければ measured=False（推定値は出さない）。電力は GPU 単体の sysfs 値（システム全体ではない）。"""
    pm = gpu.power_mean() if hasattr(gpu, "power_mean") else None
    out = {"measured": pm is not None,
           "scope": "GPU 単体（amdgpu sysfs hwmon power1_average/power1_input）。システム全体・壁電力ではない",
           "power_mean_w": pm, "power_max_w": getattr(gpu, "power_max", None),
           "power_samples": getattr(gpu, "power_samples", 0),
           "power_window_s": getattr(gpu, "power_time_s", 0.0), "wall_active_s": wall_active_s, "samples": samples,
           "gpu_busy_mean_pct": (gpu.busy_sum / gpu.busy_n) if getattr(gpu, "busy_n", 0) else None,
           "energy_j": None, "energy_per_sample_j": None}
    if pm is not None:
        out["energy_j"] = pm * wall_active_s
        if samples > 0:
            out["energy_per_sample_j"] = pm * wall_active_s / samples
    return out


def collect_environment(run_dir, be, args, cfg):
    """hardware.json / environment.txt。tb250distill.hw.collect があれば collect(run_dir) を呼ぶ。"""
    status = {}
    try:
        from tb250distill.hw import collect as hwc  # type: ignore
        hwc.collect(run_dir)
        status["hardware_collect"] = "ok"
    except Exception as e:  # noqa: BLE001
        status["hardware_collect"] = f"skipped: {type(e).__name__}: {e}"
    hw_path = os.path.join(run_dir, "hardware.json")
    if not os.path.exists(hw_path):
        with open(hw_path, "w") as f:
            json.dump({"note": "tb250distill.hw.collect が利用できないため最小情報のみ",
                       "collect_status": status["hardware_collect"],
                       "platform": platform.platform(), "machine": platform.machine(),
                       "python": sys.version, "backend": args.backend,
                       "device_name": getattr(be, "device_name", None),
                       "pci_bus_id": getattr(be, "pci_bus_id", None),
                       "opencl_version": getattr(be, "version", None)}, f, indent=1)
    env_path = os.path.join(run_dir, "environment.txt")
    lines = []
    lines.append("[student]")
    lines.append(f"python {sys.version.split()[0]} on {platform.platform()}")
    for mod in ("numpy", "pyopencl", "pyclblast", "sentencepiece"):
        try:
            m = __import__(mod)
            lines.append(f"{mod} {getattr(m, '__version__', getattr(m, 'VERSION_TEXT', '?'))}")
        except Exception:
            pass
    lines.append(f"backend {args.backend} device {getattr(be, 'device_name', None)} "
                 f"pci {getattr(be, 'pci_bus_id', None)} ocl {getattr(be, 'version', None)}")
    lines.append("argv " + " ".join(sys.argv))
    mode = "a" if os.path.exists(env_path) else "w"
    with open(env_path, mode) as f:
        f.write("\n".join(lines) + "\n")
    return status


# --------------------------------------------------------------------------------------
# チェックポイント
# --------------------------------------------------------------------------------------

def save_ckpt(path_base, model, meta):
    """path_base.npz + path_base.json（atomic）。npz は evaluate/--init からも読める（__meta__ 付き）。"""
    params = model.get_params()
    m_, v_ = model.get_opt()
    meta = dict(meta)
    meta["format"] = "tb250distill-student-v1"
    tmp = path_base + ".tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, __meta__=np.array(json.dumps(meta)), opt_m=m_, opt_v=v_, **params)
    os.replace(tmp, path_base + ".npz")
    tmpj = path_base + ".tmp.json"
    with open(tmpj, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmpj, path_base + ".json")


def load_ckpt(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    cfg = M.Config.from_dict(meta["config"])
    params = {n: z[n] for n, _ in M.param_specs(cfg)}
    return cfg, params, z["opt_m"], z["opt_v"], meta


# --------------------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Student KD 学習", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--backend", default="np", choices=["np", "cl"])
    ap.add_argument("--device", default=None, help='OpenCL デバイス名の部分一致（例 "GT 430"）')
    ap.add_argument("--data", required=True, help="data/tok/<name>（<split>_L<Lp>.npz を含む）")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="common_s", help="preset (common_s|s430|s710|s730) / JSON ファイル / JSON 文字列")
    ap.add_argument("--init", default=None, help="初期 weight npz（全 GPU 共通）。無ければ seed から生成して保存")
    ap.add_argument("--epochs", type=float, default=1, help="エポック数（小数可。端数は最後のエポックを途中で打ち切る）")
    ap.add_argument("--max-steps", type=int, default=None, help="総 step を制限（スモークテスト用）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true", help="run-dir/ckpt/last.npz から再開")
    ap.add_argument("--lp", type=int, default=None, help="shard の Lp（<split>_L<Lp>.npz）")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--limit-train", type=int, default=None, help="先頭 N 件だけ使う")
    ap.add_argument("--eval-every", type=int, default=0, help="validation 間隔 step（0=epoch 末）")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=200, help="resume 用 last.npz の保存間隔 step（0=無効）")
    ap.add_argument("--smi-interval", type=float, default=20.0,
                    help="GPU 監視（nvidia-smi / amdgpu sysfs: VRAM・温度・電力・使用率・sclk）の取得間隔（秒）。温度ガードの確認間隔も兼ねる")
    ap.add_argument("--temp-pause", type=float, default=88.0, help="GPU 温度がこれ以上なら学習を一時停止（℃、cl backend のみ）")
    ap.add_argument("--temp-resume", type=float, default=78.0, help="一時停止中、この温度以下になったら再開（℃）")
    ap.add_argument("--no-temp-guard", action="store_true", help="温度ガードを無効化（非推奨）")
    ap.add_argument("--source-loss", action="append", default=None, metavar="SRC:kd=A,ce=B",
                    help='source 別の損失重み（gold 有り sample の 0.8*KD+0.2*CE を置き換える）。例 "jnli:kd=0.2,ce=0.8"。'
                         "複数は ';' 区切り（または複数回指定）。shard の `source` キーが必要。")
    ap.add_argument("--sem-cand-weight", type=float, default=0.0, metavar="LAMBDA",
                    help="Candidate Semantic Distillation の重み λ。L_total = L_KD/CE + λ*mean(1-cos(proj(h_cand), t_cand))。"
                         "h_cand = 候補を文脈なし（初期 hidden=0）で同じ GRU に通した最終層最終 hidden。0=無効（追加 forward 無し・従来とビット一致）")
    ap.add_argument("--sem-emb", default=None, metavar="DIR",
                    help="teacher.embed の出力 data/emb/<provider>/<shard>（--sem-cand-weight>0 で必須）")
    ap.add_argument("--sem-dim", type=int, default=None,
                    help="projection head の出力次元。既定 = pca.npz の d。小さくすると PCA の先頭成分だけを使う")
    ap.add_argument("--sem-ctx-weight", type=float, default=0.0,
                    help="予約: Context Semantic Distillation（prefix 最終 hidden -> context embedding の cosine loss）。未実装（0 以外はエラー）")
    SF.add_args(ap)   # --sem-batch / --sem-loss ...（SEMFIX。既定では従来経路）
    ap.add_argument("--np-dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--replay-db", default=None, help="robust variant 情報を引く replay.sqlite（shard に無い場合）")
    return ap.parse_args(argv)


def install_sigterm_as_interrupt():
    """SIGTERM を KeyboardInterrupt として扱う（SIGINT と同じく last checkpoint を保存して終了する）。
    `&` で起動したバックグラウンド run は SIGINT が無視されるので、停止は SIGTERM になる。メインスレッド以外では何もしない。
    戻り値: 元のハンドラへ戻す関数。"""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    def _handler(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    prev = signal.signal(signal.SIGTERM, _handler)
    return lambda: signal.signal(signal.SIGTERM, prev)


def epoch_order(seed, epoch, n, kmax):
    """(seed, epoch) だけから決まる sample 順序と候補 permutation 用の乱数（デバイス非依存）。"""
    order = np.random.default_rng([seed, epoch, 1]).permutation(n)
    keys = np.random.default_rng([seed, epoch, 2]).random((n, kmax))
    return order, keys


def main(argv=None):
    args = parse_args(argv)
    run_dir = args.run_dir
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
    last_path = os.path.join(run_dir, "ckpt", "last")
    resuming = args.resume and os.path.exists(last_path + ".npz")
    if (not args.resume) and os.path.exists(last_path + ".npz"):
        raise SystemExit(f"{run_dir} には既存 checkpoint がある。--resume するか別の --run-dir を使う")

    handlers = [logging.FileHandler(os.path.join(run_dir, "train.log"), mode="a"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers, force=True)

    # --- データ ---
    tr_path = M.find_shard(args.data, args.train_split, args.lp)
    if tr_path is None:
        raise SystemExit(f"train shard not found in {args.data}")
    tr = M.Shard(tr_path)
    if args.sem_ctx_weight:
        raise SystemExit("--sem-ctx-weight は未実装（Context Semantic Distillation は今後。0 にする）")
    sem_w = float(args.sem_cand_weight)
    if not (np.isfinite(sem_w) and sem_w >= 0):
        raise SystemExit("--sem-cand-weight は有限の非負数")
    sem = sem_info = None
    if args.sem_emb:
        try:
            M.assert_not_eval_only_dir(args.sem_emb)   # data/emb_eval（評価専用 Teacher 埋め込み）は重み 0 でも拒否
        except ValueError as e:
            raise SystemExit(f"--sem-emb: {e}")
    if sem_w > 0:
        if not args.sem_emb:
            raise SystemExit("--sem-cand-weight > 0 には --sem-emb（teacher.embed の出力ディレクトリ）が必要")
        try:
            sem, sem_info = M.load_sem_dir(args.sem_emb, tr.item_id, args.sem_dim)
            sem.check_against(tr)
        except (ValueError, OSError) as e:
            raise SystemExit(f"--sem-emb: {e}")
    elif args.sem_emb:
        log.warning("--sem-emb は --sem-cand-weight=0 のため使わない（追加 forward なし）")
    if args.limit_train:
        sel = np.arange(min(args.limit_train, tr.n))
        tr = tr.subset(sel)
        if sem is not None:
            sem = sem.subset(sel)
    va_path = M.find_shard(args.data, args.val_split, tr.lp)
    va = M.Shard(va_path) if va_path else None
    extra_shards = {}
    for sp in ("test", "robust"):
        p = M.find_shard(args.data, sp, tr.lp)
        if p:
            extra_shards[sp] = M.Shard(p)

    # --- モデル設定・初期値 ---
    if resuming:
        cfg, params, opt_m, opt_v, rmeta = load_ckpt(last_path + ".npz")
    elif args.init:
        cfg, params, _ = M.load_params_npz(args.init)
        c2 = M.get_config(args.config)
        if (c2.vocab, c2.emb, c2.hidden, c2.layers) != (cfg.vocab, cfg.emb, cfg.hidden, cfg.layers) \
                and args.config not in ("common_s",):
            log.warning("--config と --init の構造が異なる。--init の構造を使う: %s", cfg.to_dict())
    else:
        cfg = M.get_config(args.config, vocab=args.vocab)
        params = M.init_params(cfg, args.seed)
    want_sem = sem.d if sem is not None else 0
    if resuming:
        if cfg.sem_dim != want_sem:
            raise SystemExit(f"RESUME: checkpoint の sem_dim={cfg.sem_dim} と今回の設定（sem_dim={want_sem}）が違う。"
                             "元の run と同じ --sem-cand-weight/--sem-emb/--sem-dim で再開する")
    else:
        if cfg.sem_dim not in (0, want_sem):
            raise SystemExit(f"--init の sem_dim={cfg.sem_dim} と今回の sem_dim={want_sem} が違う")
        if want_sem and not cfg.sem_dim:
            cfg.sem_dim = want_sem
            params = {**params, **M.init_sem_params(cfg, args.seed)}   # projection head は seed 決定的な別 stream で初期化
        elif cfg.sem_dim and not want_sem:
            log.warning("--init に projection head があるが sem 学習は無効。head を捨てる")
            cfg.sem_dim = 0
            params = {k: v for k, v in params.items() if not k.startswith("sem.")}
    try:
        source_loss = M.parse_source_loss(args.source_loss)
    except ValueError as e:
        raise SystemExit(str(e))
    if source_loss:
        src_arr = tr.extra.get("source")
        if src_arr is None:
            log.warning("--source-loss が指定されたが train shard に `source` キーが無い。source 別重みは無効（従来の損失）")
        else:
            present = {str(x): int((src_arr == x).sum()) for x in set(src_arr.tolist())}
            for nm in source_loss:
                if nm not in present:
                    log.warning("--source-loss の source '%s' は train shard に存在しない（存在: %s）", nm, sorted(present))
            log.info("source 別損失重み: %s（該当 sample 数 %s）", source_loss,
                     {nm: present.get(nm, 0) for nm in source_loss})
    all_shards = [s for s in [tr, va, *extra_shards.values()] if s is not None]
    max_tok = max(s.max_token() for s in all_shards)
    if max_tok >= cfg.vocab:
        raise SystemExit(f"shard の最大 token id {max_tok} >= vocab {cfg.vocab}（--config / --vocab を確認）")
    lp = max(s.lp for s in all_shards)
    lc = max(s.lc for s in all_shards)
    kmax = max(s.kmax for s in all_shards)
    cfg.lp, cfg.lc = lp, lc
    if not resuming and not args.init:
        M.save_params_npz(os.path.join(run_dir, "init.npz"), cfg, params, {"seed": args.seed})

    be = M.make_backend(args.backend, args.device, args.np_dtype)
    model = M.Student(be, cfg, args.batch_size, kmax, train=True, max_lp=lp, max_lc=lc, max_sem=args.sem_batch)
    model.set_params(params)
    model.sem_lambda = sem_w if sem is not None else 0.0
    try:
        model.sem_ext = SF.build_hook(args, tr, sem, sem_w)   # None = 従来経路（判断バッチ内の候補・cos）
    except ValueError as e:
        raise SystemExit(f"--sem-batch/--sem-loss: {e}")
    if sem is not None:
        log.info("Candidate Semantic Distillation: λ=%g d=%d provider=%s unique_strings=%d (pca d=%d) emb=%s", sem_w, sem.d,
                 sem_info["provider"], sem_info["n_strings"], sem_info["d_pca"], args.sem_emb)

    # --- 計画 ---
    n = tr.n
    spe = math.ceil(n / args.batch_size)
    total = int(round(args.epochs * spe))
    if args.max_steps:
        total = min(total, args.max_steps)
    total = max(total, 1)
    ck_steps = {}
    for fr in CKPT_FRACS:
        ck_steps.setdefault(int(round(total * fr)), []).append(int(round(fr * 100)))
    eval_every = args.eval_every or spe

    git = git_info()
    step = 0
    wall_prev = 0.0
    paused = {"s": 0.0, "n": 0}
    samples_done = 0
    power_prev = (0.0, 0.0)
    best = {"val_kl": float("inf"), "step": None}
    if resuming:
        step = int(rmeta["step"])
        wall_prev = float(rmeta.get("wall_s", 0.0))
        paused = {"s": float(rmeta.get("paused_s", 0.0)), "n": int(rmeta.get("pauses", 0))}
        samples_done = int(rmeta.get("samples_done", step * args.batch_size))
        power_prev = (rmeta.get("power_integral_j", 0.0), rmeta.get("power_time_s", 0.0))
        best = rmeta.get("best", best)
        model.set_opt(opt_m, opt_v)
        old_a = rmeta.get("args") or {}
        if float(old_a.get("sem_cand_weight") or 0.0) != sem_w:
            log.warning("RESUME: --sem-cand-weight が元の run と異なる（元 %s / 今回 %s）", old_a.get("sem_cand_weight"), sem_w)
        old_sl = M.parse_source_loss((rmeta.get("args") or {}).get("source_loss"))
        if old_sl != source_loss:
            log.warning("RESUME: --source-loss が元の run と異なる（元 %s / 今回 %s）", old_sl, source_loss)
        log.info("RESUME from step %d (wall %.1fs) best=%s", step, wall_prev, best)

    gpu = SmiMonitor(be, args.smi_interval)
    if resuming and hasattr(gpu, "restore"):
        gpu.restore(*power_prev)
    gpu_info = gpu.info() if hasattr(gpu, "info") else {}
    log.info("gpu monitor: %s", json.dumps(gpu_info, ensure_ascii=False))
    if not resuming:
        status = collect_environment(run_dir, be, args, cfg)
        _add_to_hardware_json(run_dir, "student_gpu_monitor", gpu_info)
        cfgd = {"args": vars(args), "model": cfg.to_dict(), "n_params": M.param_count(cfg),
                "n_params_breakdown": M.param_count(cfg, breakdown=True),
                "backend": args.backend, "device": getattr(be, "device_name", None),
                "pci_bus_id": getattr(be, "pci_bus_id", None),
                "data": {"dir": args.data, "train": tr_path, "val": va_path, "n_train": n,
                         "n_val": va.n if va else 0, "lp": lp, "lc": lc, "kmax": kmax},
                "plan": {"steps_per_epoch": spe, "total_steps": total, "checkpoint_steps": {str(k): v for k, v in ck_steps.items()},
                         "eval_every": eval_every, "best_metric": "val_kl（最小）"},
                "gpu_monitor": gpu_info,
                "temp_guard": {"enabled": gpu.enabled and not args.no_temp_guard, "pause_c": args.temp_pause,
                               "resume_c": args.temp_resume, "check_interval_s": args.smi_interval},
                "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": args.wd,
                              "clip_grad_norm": args.clip},
                "loss": {"T": M.KD_T, "kd_weight_with_gold": M.W_KD_GOLD, "ce_weight_with_gold": M.W_CE_GOLD,
                         "source_loss": {k: {"kd": v[0], "ce": v[1]} for k, v in source_loss.items()},
                         "source_loss_active": bool(source_loss) and tr.extra.get("source") is not None},
                "sem": {"enabled": sem is not None, "cand_weight": sem_w, "ctx_weight": 0.0 if not args.sem_ctx_weight else None,
                        "sem_dim": cfg.sem_dim, "emb_dir": args.sem_emb if sem is not None else None,
                        "info": sem_info, "loss": "mean over valid train candidates of 1-cos(Linear(h_cand), t_cand); "
                                                  "h_cand = candidate-only pass (h0=0, shared GRU), final layer final hidden",
                        "head_params": int(cfg.hidden * cfg.sem_dim + cfg.sem_dim) if cfg.sem_dim else 0,
                        "semfix": model.sem_ext.describe() if model.sem_ext is not None else None},
                "git_commit": git["commit"], "git_commit_source": git["source"], "git_note": git["note"],
                "init": args.init or os.path.join(run_dir, "init.npz"), **status}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfgd, f, indent=1, ensure_ascii=False)
        if git["diff"]:
            with open(os.path.join(run_dir, "git.diff"), "w") as f:
                f.write(git["diff"])
    log.info("backend=%s device=%s params=%d steps=%d (spe=%d) N=%d Lp=%d Lc=%d Kmax=%d git=%s",
             args.backend, getattr(be, "device_name", None), M.param_count(cfg), total, spe, n, lp, lc, kmax,
             (git["commit"] or "none")[:12])

    csv_path = os.path.join(run_dir, "metrics.csv")
    new_csv = not (resuming and os.path.exists(csv_path))
    csv_cols = list(CSV_COLS)
    if not new_csv:   # 旧 run の再開: 既存 header が CSV_COLS の先頭部分なら、その列だけ書く（列数を揃える）
        with open(csv_path) as f0:
            hdr0 = f0.readline().strip().split(",")
        if hdr0 != csv_cols and csv_cols[:len(hdr0)] == hdr0:
            csv_cols = hdr0
    csv_f = open(csv_path, "a" if not new_csv else "w", newline="")
    cw = csv.writer(csv_f)
    if new_csv:
        cw.writerow(csv_cols)

    t_start = time.perf_counter()

    def wall():
        return wall_prev + (time.perf_counter() - t_start)

    def meta_now(pct=None):
        return {"config": cfg.to_dict(), "step": step, "epoch": step / spe, "steps_per_epoch": spe,
                "total_steps": total, "pct": pct, "seed": args.seed, "wall_s": wall(), "paused_s": paused["s"],
                "pauses": paused["n"], "best": best, "samples_done": samples_done,
                "power_integral_j": getattr(gpu, "power_integral_j", 0.0), "power_time_s": getattr(gpu, "power_time_s", 0.0),
                "dataset_position": {"epoch": step // spe, "batch_in_epoch": step % spe,
                                     "order": "default_rng([seed,epoch,1]).permutation; cand keys default_rng([seed,epoch,2])"},
                "rng_state": {"seed": args.seed, "epoch": step // spe, "batch_in_epoch": step % spe},
                "git_commit": git["commit"], "args": vars(args)}

    def validate():
        if va is None:
            return None
        s = E.predict_scores(model, va)
        m, _ = E.core_metrics(s, va)
        return m

    def write_row(trn, val):
        gpu.poll(force=val is not None)
        row = {"step": step, "epoch": round(step / spe, 4)}
        if trn:
            row.update({"train_loss": trn["loss"], "kd": trn["kd"], "ce": trn["ce"],
                        "samples_per_s": trn["sps"], "tokens_per_s": trn["tps"], "step_ms": trn["ms"],
                        "sem_loss": trn.get("sem")})
        if val:
            row.update({"val_loss": val["val_loss"], "val_kl": val["kl"], "val_agree": val["agreement"],
                        "val_gold_acc": val["gold_acc"], "val_brier": val["brier"],
                        "val_ece": val["ece15"] if val["ece15"] is not None else val["ece15_vs_teacher"]})
        row["vram_mb"] = gpu.vram
        row["temp_c"] = gpu.temp
        row["power_w"] = getattr(gpu, "power", None)
        row["gpu_busy_pct"] = getattr(gpu, "busy", None)
        row["sclk_mhz"] = getattr(gpu, "sclk", None)
        row["wall_s"] = round(wall(), 2)
        row["paused_s"] = round(paused["s"], 2)
        cw.writerow(["" if row.get(c) is None else (f"{row[c]:.6g}" if isinstance(row.get(c), float) else row.get(c))
                     for c in csv_cols])
        csv_f.flush()

    warned = {"none": False}

    def temp_guard():
        """GPU 温度が temp_pause 以上なら temp_resume 以下になるまで待機（停止時間は paused_s に積み、wall と区別して記録）。"""
        if not gpu.enabled or args.no_temp_guard:
            return
        gpu.poll()
        if gpu.temp is None:
            if not warned["none"]:
                warned["none"] = True
                log.warning("温度を取得できない（%s: %s）。温度ガード無効で続行", getattr(gpu, "kind", None) or "gpu monitor", gpu.err)
            return
        if gpu.temp < args.temp_pause:
            return
        t0 = time.perf_counter()
        log.warning("TEMP PAUSE at step %d: %.0fC >= %.0fC。%.0fC 以下まで待機", step, gpu.temp, args.temp_pause,
                    args.temp_resume)
        gpu.recording = False           # 停止中の電力は active 区間の平均に混ぜない
        try:
            while True:
                time.sleep(5.0)
                gpu.poll(force=True)
                if gpu.temp is None or gpu.temp <= args.temp_resume:
                    break
        finally:
            gpu.recording = True
        dt = time.perf_counter() - t0
        paused["s"] += dt
        paused["n"] += 1
        log.warning("TEMP RESUME at step %d after %.1fs (now %s C, 累計停止 %.1fs / %d 回)", step, dt, gpu.temp,
                    paused["s"], paused["n"])

    def do_eval(tag=""):
        nonlocal best
        val = validate()
        if val is None:
            return None
        log.info("VAL step=%d%s loss=%.4f kl=%.4f agree=%.4f (rand %.4f) gold=%s brier=%s ece=%s", step, tag,
                 val["val_loss"], val["kl"], val["agreement"], val["random_baseline"],
                 None if val["gold_acc"] is None else round(val["gold_acc"], 4),
                 None if val["brier"] is None else round(val["brier"], 4),
                 None if val["ece15"] is None else round(val["ece15"], 4))
        if val["kl"] < best["val_kl"]:
            best = {"val_kl": val["kl"], "step": step, "val_agree": val["agreement"]}
            save_ckpt(os.path.join(run_dir, "ckpt", "best"), model, meta_now("best") | {"val": {k: v for k, v in val.items() if not isinstance(v, (list, dict))}})
            log.info("  new best val_kl=%.5f -> ckpt/best", val["kl"])
        return val

    def save_pct(pct_list):
        for pct in pct_list:
            save_ckpt(os.path.join(run_dir, "ckpt", f"p{pct:03d}"), model, meta_now(pct))
            log.info("checkpoint p%03d at step %d", pct, step)

    # step 0
    if not resuming:
        gpu.poll(force=True)
        val0 = do_eval(" (init)")
        write_row(None, val0)
        if 0 in ck_steps:
            save_pct(ck_steps[0])
        save_ckpt(last_path, model, meta_now())

    acc = None

    def reset_acc():
        return {"loss": 0.0, "kd": 0.0, "ce": 0.0, "sem": 0.0, "n": 0, "ng": 0, "samples": 0, "tokens": 0, "t": 0.0, "steps": 0}

    acc = reset_acc()
    cur_epoch = -1
    order = keys = None
    interrupted = False
    restore_sigterm = install_sigterm_as_interrupt()
    try:
        while step < total:
            temp_guard()
            ep, pos = divmod(step, spe)
            if ep != cur_epoch:
                order, keys = epoch_order(args.seed, ep, n, tr.kmax)
                cur_epoch = ep
            idx = order[pos * args.batch_size:(pos + 1) * args.batch_size]
            t0 = time.perf_counter()
            bt = M.make_batch(tr, idx, keys=keys[idx], source_loss=source_loss, sem=sem)
            if model.sem_ext is not None:
                model.sem_ext.step = step      # 独立ミニバッチは step だけで決まる（--resume で厳密に再現）
            st = model.train_step(bt, args.lr, step + 1, wd=args.wd, clip=args.clip)
            dt = time.perf_counter() - t0
            step += 1
            if not math.isfinite(st["loss"]):
                raise RuntimeError(f"non-finite loss at step {step}: {st}")
            B = bt.B
            acc["loss"] += st["loss"] * B
            acc["kd"] += st["kd"] * B
            acc["ce"] += st["ce"] * st["n_gold"]
            acc["ng"] += st["n_gold"]
            acc["sem"] += st["sem"]
            acc["n"] += B
            acc["samples"] += B
            samples_done += B
            acc["tokens"] += bt.n_tokens
            acc["t"] += dt
            acc["steps"] += 1
            do_val = (step % eval_every == 0) or step == total
            if step % args.log_every == 0 or do_val or step == total:
                gpu.poll()
                trn = {"loss": acc["loss"] / acc["n"], "kd": acc["kd"] / acc["n"],
                       "ce": acc["ce"] / max(1, acc["ng"]), "sps": acc["samples"] / acc["t"],
                       "tps": acc["tokens"] / acc["t"], "ms": acc["t"] / acc["steps"] * 1e3}
                if sem is not None:
                    trn["sem"] = acc["sem"] / acc["steps"]
                log.info("step %d/%d ep %.2f loss %.4f kd %.4f ce %.4f gnorm %.3f | %.1f samples/s %.0f tok/s %.1f ms/step "
                         "vram %s temp %s power %s busy %s sclk %s", step, total, step / spe, trn["loss"], trn["kd"],
                         trn["ce"], st["gnorm"], trn["sps"], trn["tps"], trn["ms"], gpu.vram, gpu.temp,
                         getattr(gpu, "power", None), getattr(gpu, "busy", None), getattr(gpu, "sclk", None))
                if sem is not None:
                    if model.sem_ext is None:
                        log.info("  sem_loss %.4f (cand cosine, λ=%g; train_loss は λ*sem_loss を含む)", trn["sem"], sem_w)
                    else:
                        pm = model.sem_ext.pop_parts()
                        log.info("  sem_loss %.4f (独立ミニバッチ M=%d %s, λ=%g; 各項(重み前) %s)", trn["sem"], model.sem_ext.sampler.M,
                                 "+".join(model.sem_ext.spec.terms), sem_w, {k: round(v, 4) for k, v in (pm or {}).items()})
                val = do_eval() if do_val else None
                write_row(trn, val)
                acc = reset_acc()
            if step in ck_steps and step != 0:
                save_pct(ck_steps[step])
            if args.ckpt_every and step % args.ckpt_every == 0 and step != total:
                save_ckpt(last_path, model, meta_now())
    except KeyboardInterrupt:
        interrupted = True
        log.info("interrupted at step %d; saving last checkpoint", step)
    finally:
        restore_sigterm()
    save_ckpt(last_path, model, meta_now())
    csv_f.close()
    train_wall = wall()

    summary = {"steps_done": step, "total_steps": total, "interrupted": interrupted, "wall_s": train_wall,
               "paused_s": paused["s"], "pauses": paused["n"], "wall_active_s": train_wall - paused["s"],
               "best": best, "vram_max_mb": gpu.vram_max, "temp_max_c": gpu.temp_max, "smi_error": gpu.err,
               "samples_done": samples_done}
    summary["energy"] = energy_summary(gpu, samples_done, train_wall - paused["s"])
    if summary["energy"]["measured"]:
        log.info("ENERGY (GPU 単体・sysfs 実測): mean %.2f W, active wall %.1fs, %d samples -> %.4f J/sample",
                 summary["energy"]["power_mean_w"], summary["energy"]["wall_active_s"], samples_done,
                 summary["energy"]["energy_per_sample_j"])
    if (not args.no_final_eval) and (not interrupted) and step >= total:
        shards = {"val": va, **extra_shards}
        res = E.evaluate_all(model, shards, replay_db=args.replay_db, latency=True)
        res["final_step"] = step
        res["train_summary"] = summary
        res["n_params"] = M.param_count(cfg)
        res["config"] = cfg.to_dict()
        res["backend"] = args.backend
        res["device"] = getattr(be, "device_name", None)
        with open(os.path.join(run_dir, "eval.json"), "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        for name, m in res["splits"].items():
            log.info("FINAL %s: agree %.4f (random %.4f) kl %.4f gold_acc %s ece %s", name, m["agreement"],
                     m["random_baseline"], m["kl"], m["gold_acc"], m["ece15"])
        km = res.get("key_metrics") or {}
        if km.get("massive_test_agreement") is not None:
            log.info("FINAL key_metrics: massive_test_agreement %.4f (n %s, random %.4f)", km["massive_test_agreement"],
                     km.get("massive_test_n"), km.get("massive_test_random_baseline"))
        for name, bs in res.get("by_source", {}).items():
            for src_name, m in bs.items():
                log.info("FINAL %s/%s: n %s agree %.4f kl %.4f gold_acc %s", name, src_name, m.get("n"),
                         m["agreement"], m["kl"], m.get("gold_acc"))
    log.info("done: %s", json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
