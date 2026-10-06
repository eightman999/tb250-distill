"""ハードウェア情報収集（Phase 0）。

`collect(out_dir)` で hardware.json と environment.txt を書き出し、dict を返す。
CLI: python -m tb250distill.hw.collect --out runs/hw

取得に失敗した項目は例外にせず "error: ..." 文字列で記録する。
ドライバ・カーネルモジュール等は一切変更しない（読み取りのみ）。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

VK_ICD = "/usr/share/vulkan/icd.d/radeon_icd.json"
SYS_PCI = "/sys/bus/pci/devices"


def _err(e) -> str:
    return "error: %s: %s" % (type(e).__name__, e)


def _run(cmd, env=None, timeout=60) -> str:
    """コマンドを実行して stdout を返す。失敗時は 'error: ...' 文字列。"""
    try:
        e = dict(os.environ)
        if env:
            e.update(env)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e)
        if p.returncode != 0:
            return "error: rc=%d %s" % (p.returncode, (p.stderr or p.stdout).strip()[:500])
        return p.stdout.strip()
    except Exception as ex:  # noqa: BLE001
        return _err(ex)


def _read(path: str):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception as ex:  # noqa: BLE001
        return _err(ex)


def _is_err(v) -> bool:
    return isinstance(v, str) and v.startswith("error")


# ---------------------------------------------------------------- CPU / RAM
def collect_cpu() -> dict:
    out: dict = {}
    try:
        info = _read("/proc/cpuinfo")
        names = re.findall(r"^model name\s*:\s*(.+)$", info, re.M)
        out["model"] = names[0] if names else "error: no model name"
        out["logical_cores"] = len(names)
        mhz = re.findall(r"^cpu MHz\s*:\s*(.+)$", info, re.M)
        out["mhz_now"] = [float(x) for x in mhz] if mhz else "error: no cpu MHz"
        flags = re.findall(r"^flags\s*:\s*(.+)$", info, re.M)
        if flags:
            fs = flags[0].split()
            out["simd_flags"] = [f for f in fs if f in ("sse4_2", "avx", "avx2", "fma", "avx512f")]
    except Exception as ex:  # noqa: BLE001
        out["error"] = _err(ex)
    out["lscpu"] = _run(["lscpu"])
    return out


def collect_ram() -> dict:
    out: dict = {}
    try:
        mi = _read("/proc/meminfo")
        for key in ("MemTotal", "MemAvailable", "SwapTotal"):
            m = re.search(r"^%s:\s+(\d+) kB" % key, mi, re.M)
            out[key + "_MiB"] = int(m.group(1)) // 1024 if m else "error: missing"
    except Exception as ex:  # noqa: BLE001
        out["error"] = _err(ex)
    return out


def collect_system() -> dict:
    u = platform.uname()
    return {
        "hostname": u.node,
        "kernel": u.release,
        "kernel_version": u.version,
        "machine": u.machine,
        "os_release": _read("/etc/os-release"),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- GPU (PCI)
def _lspci_name(slot: str) -> str:
    out = _run(["lspci", "-mm", "-s", slot])
    if _is_err(out):
        return out
    # "03:00.0 "VGA compatible controller" "Advanced Micro Devices..." "Navi 24 [Radeon RX 6400]" ..."
    parts = re.findall(r'"([^"]*)"', out)
    return " / ".join(parts[1:3]) if len(parts) >= 3 else out


def _gpu_pci_devices() -> list[dict]:
    gpus = []
    for d in sorted(glob.glob(SYS_PCI + "/*")):
        cls = _read(d + "/class")
        if _is_err(cls) or not cls.startswith("0x03"):
            continue
        slot = os.path.basename(d)
        vendor = _read(d + "/vendor")
        device = _read(d + "/device")
        drv_link = d + "/driver"
        driver = os.path.basename(os.readlink(drv_link)) if os.path.islink(drv_link) else None
        g = {
            "pci_slot": slot,
            "pci_id": "%s:%s" % (vendor.replace("0x", ""), device.replace("0x", "")),
            "class": cls,
            "lspci_name": _lspci_name(slot),
            "kernel_driver": driver,
            "link": {
                "current_width": _read(d + "/current_link_width"),
                "max_width": _read(d + "/max_link_width"),
                "current_speed": _read(d + "/current_link_speed"),
                "max_speed": _read(d + "/max_link_speed"),
            },
            "_sysfs": d,
        }
        gpus.append(g)
    return gpus


# ---------------------------------------------------------------- NVIDIA
_NV_FIELDS = [
    "index", "name", "pci.bus_id", "memory.total", "memory.used", "temperature.gpu",
    "pstate", "clocks.sm", "clocks.mem", "clocks.gr", "power.draw", "utilization.gpu",
    "driver_version", "vbios_version",
]


def collect_nvidia() -> dict:
    out: dict = {"driver_proc": _read("/proc/driver/nvidia/version")}
    out["nvidia_smi_L"] = _run(["nvidia-smi", "-L"])
    out["nvidia_smi_q_clock_perf_temp"] = _run(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,TEMPERATURE"])
    # 一部フィールドが非対応でも全体が落ちないよう個別に問い合わせる。
    base = _run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
    gpus: dict = {}
    if _is_err(base):
        out["error"] = base
        out["gpus"] = gpus
        return out
    for line in base.splitlines():
        idx, bus = [x.strip() for x in line.split(",", 1)]
        gpus[idx] = {"pci_bus_id": bus}
    for fld in _NV_FIELDS:
        if fld in ("index", "pci.bus_id"):
            continue
        r = _run(["nvidia-smi", "--query-gpu=index,%s" % fld, "--format=csv,noheader"])
        if _is_err(r):
            for idx in gpus:
                gpus[idx][fld] = r
            continue
        for line in r.splitlines():
            idx, val = [x.strip() for x in line.split(",", 1)]
            if idx in gpus:
                gpus[idx][fld] = val
    out["gpus"] = gpus
    return out


# ---------------------------------------------------------------- AMD
def _amd_hwmon_temps(sysfs: str) -> dict:
    temps = {}
    for p in sorted(glob.glob(sysfs + "/hwmon/hwmon*/temp*_input")):
        v = _read(p)
        if _is_err(v):
            temps[p] = v
            continue
        try:
            temps[os.path.relpath(p, sysfs)] = int(v) / 1000.0
        except ValueError:
            temps[os.path.relpath(p, sysfs)] = "error: bad value %r" % v
    return temps or "error: no hwmon temp"


def collect_amd(sysfs: str) -> dict:
    out: dict = {}
    for key in ("mem_info_vram_total", "mem_info_vram_used", "mem_info_vis_vram_total", "mem_info_gtt_total"):
        v = _read(sysfs + "/" + key)
        out[key + "_MiB"] = int(v) // (1024 * 1024) if re.fullmatch(r"\d+", v or "") else v
    out["power_dpm_force_performance_level"] = _read(sysfs + "/power_dpm_force_performance_level")
    out["power_dpm_state"] = _read(sysfs + "/power_dpm_state")
    for key in ("pp_dpm_sclk", "pp_dpm_mclk", "pp_dpm_fclk", "pp_dpm_socclk", "pp_dpm_pcie"):
        if os.path.exists(sysfs + "/" + key):
            out[key] = _read(sysfs + "/" + key)
    out["temperatures_C"] = _amd_hwmon_temps(sysfs)
    # drm card 名（/sys/class/drm/card*/device/hwmon でも辿れるが slot からの対応を残す）
    cards = [os.path.basename(os.path.dirname(c)) for c in glob.glob("/sys/class/drm/card[0-9]*/device")
             if os.path.realpath(c) == os.path.realpath(sysfs)]
    out["drm_cards"] = cards
    return out


# ---------------------------------------------------------------- Vulkan / OpenCL
def collect_vulkan() -> dict:
    env = {"VK_ICD_FILENAMES": VK_ICD}
    out = {
        "VK_ICD_FILENAMES": VK_ICD,
        "summary": _run(["vulkaninfo", "--summary"], env=env, timeout=60),
    }
    s = out["summary"]
    if not _is_err(s):
        out["devices"] = re.findall(r"deviceName\s*=\s*(.+)", s)
        out["driver_names"] = re.findall(r"driverName\s*=\s*(.+)", s)
    return out


def collect_opencl() -> dict:
    try:
        import pyopencl as cl
    except Exception as ex:  # noqa: BLE001
        return {"error": _err(ex)}
    res = []
    try:
        for pi, p in enumerate(cl.get_platforms()):
            pd = {"platform_index": pi, "name": p.name, "version": p.version, "vendor": p.vendor, "devices": []}
            for di, d in enumerate(p.get_devices()):
                dd: dict = {"device_index": di}
                for key, attr in (
                    ("name", "name"), ("version", "version"), ("opencl_c_version", "opencl_c_version"),
                    ("driver_version", "driver_version"), ("global_mem_bytes", "global_mem_size"),
                    ("max_mem_alloc_bytes", "max_mem_alloc_size"), ("local_mem_bytes", "local_mem_size"),
                    ("max_compute_units", "max_compute_units"), ("max_clock_mhz", "max_clock_frequency"),
                    ("max_work_group_size", "max_work_group_size"), ("type", "type"),
                ):
                    try:
                        v = getattr(d, attr)
                        dd[key] = str(v) if key == "type" else v
                    except Exception as ex:  # noqa: BLE001
                        dd[key] = _err(ex)
                try:
                    dd["has_fp64"] = "cl_khr_fp64" in d.extensions
                except Exception as ex:  # noqa: BLE001
                    dd["has_fp64"] = _err(ex)
                try:  # NVIDIA は PCI bus id を取れる
                    dd["pci_bus_id_nv"] = d.get_info(cl.device_info.PCI_BUS_ID_NV)
                except Exception:  # noqa: BLE001
                    pass
                pd["devices"].append(dd)
            res.append(pd)
    except Exception as ex:  # noqa: BLE001
        return {"error": _err(ex), "platforms": res}
    return {"platforms": res}


# ---------------------------------------------------------------- 本体
def collect_hardware() -> dict:
    hw: dict = {}
    steps = {
        "system": collect_system,
        "cpu": collect_cpu,
        "ram": collect_ram,
        "vulkan": collect_vulkan,
        "opencl": collect_opencl,
        "nvidia": collect_nvidia,
    }
    for k, fn in steps.items():
        try:
            hw[k] = fn()
        except Exception as ex:  # noqa: BLE001
            hw[k] = _err(ex)

    nv = hw.get("nvidia", {})
    nv_by_bus = {}
    if isinstance(nv, dict) and isinstance(nv.get("gpus"), dict):
        for g in nv["gpus"].values():
            bus = g.get("pci_bus_id", "")
            nv_by_bus[bus.lower()[-12:]] = g  # "00000000:04:00.0" -> "0000:04:00.0"

    gpus = []
    try:
        pci = _gpu_pci_devices()
    except Exception as ex:  # noqa: BLE001
        pci = []
        hw["gpu_error"] = _err(ex)
    for g in pci:
        sysfs = g.pop("_sysfs")
        drv = g["kernel_driver"]
        entry = dict(g)
        try:
            if drv == "nvidia":
                n = nv_by_bus.get(g["pci_slot"].lower()[-12:], None)
                if n is None:
                    entry["nvidia"] = "error: no nvidia-smi entry for %s" % g["pci_slot"]
                else:
                    entry["nvidia"] = n
                    entry["vram_MiB"] = n.get("memory.total")
                    entry["temperature_C"] = n.get("temperature.gpu")
                    entry["pstate"] = n.get("pstate")
                    entry["clocks"] = {
                        "sm": n.get("clocks.sm"), "mem": n.get("clocks.mem"), "gr": n.get("clocks.gr"),
                    }
                    entry["driver"] = {
                        "name": "nvidia",
                        "version": n.get("driver_version"),
                        "proc_version": nv.get("driver_proc"),
                    }
            elif drv == "amdgpu":
                a = collect_amd(sysfs)
                entry["amdgpu"] = a
                entry["vram_MiB"] = a.get("mem_info_vram_total_MiB")
                entry["temperature_C"] = a.get("temperatures_C")
                entry["power_state"] = {
                    "power_dpm_force_performance_level": a.get("power_dpm_force_performance_level"),
                    "pp_dpm_sclk": a.get("pp_dpm_sclk"),
                    "pp_dpm_mclk": a.get("pp_dpm_mclk"),
                }
                entry["clocks"] = {"pp_dpm_sclk": a.get("pp_dpm_sclk"), "pp_dpm_mclk": a.get("pp_dpm_mclk")}
                entry["driver"] = {
                    "name": "amdgpu",
                    "version": "kernel " + platform.release(),
                    "module_version": _read("/sys/module/amdgpu/version") if os.path.exists("/sys/module/amdgpu/version") else None,
                }
            else:
                entry["driver"] = {"name": drv, "version": "kernel " + platform.release()}
                entry["note"] = "未使用/未収集（%s）" % drv
        except Exception as ex:  # noqa: BLE001
            entry["error"] = _err(ex)
        gpus.append(entry)
    hw["gpus"] = gpus
    return hw


def collect_environment() -> str:
    parts = []
    parts.append("# python\n%s\nexecutable: %s" % (sys.version, sys.executable))
    parts.append("# uname -a\n" + _run(["uname", "-a"]))
    here = Path(__file__).resolve().parent
    commit = _run(["git", "-C", str(here), "rev-parse", "HEAD"])
    if _is_err(commit) or not commit:
        commit = "none"
    parts.append("# git commit\n" + commit)
    pf = _run([sys.executable, "-m", "pip", "freeze"], timeout=120)
    if _is_err(pf):
        try:
            from importlib import metadata
            pf = "\n".join(sorted("%s==%s" % (d.metadata["Name"], d.version) for d in metadata.distributions()))
        except Exception as ex:  # noqa: BLE001
            pf = pf + "\n" + _err(ex)
    parts.append("# pip freeze\n" + pf)
    return "\n\n".join(parts) + "\n"


def collect(out_dir) -> dict:
    """hardware.json と environment.txt を out_dir に書き出し、hardware dict を返す。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        hw = collect_hardware()
    except Exception as ex:  # noqa: BLE001
        hw = {"error": _err(ex)}
    (out / "hardware.json").write_text(json.dumps(hw, indent=2, ensure_ascii=False, default=str) + "\n")
    try:
        env = collect_environment()
    except Exception as ex:  # noqa: BLE001
        env = _err(ex) + "\n"
    (out / "environment.txt").write_text(env)
    return hw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Collect hardware.json / environment.txt")
    ap.add_argument("--out", default="runs/hw")
    a = ap.parse_args(argv)
    hw = collect(a.out)
    for g in hw.get("gpus", []):
        print("%s %s [%s] drv=%s vram=%s link=%s/%s %s temp=%s" % (
            g["pci_slot"], g.get("lspci_name"), g["pci_id"], g.get("kernel_driver"), g.get("vram_MiB"),
            g["link"]["current_width"], g["link"]["max_width"], g["link"]["current_speed"], g.get("temperature_C")))
    print("wrote", Path(a.out) / "hardware.json", Path(a.out) / "environment.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
