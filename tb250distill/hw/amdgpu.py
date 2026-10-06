"""amdgpu（AMD GPU）の VRAM・温度・電力・使用率・sclk を sysfs から読む。GPU は使わない（ファイル読み取りのみ）。

  - 対象カードの特定（推測で別カードを読まない）:
      1. OpenCL の cl_khr_pci_bus_info（pyopencl に属性があれば使い、無ければ ctypes で clGetDeviceInfo(0x410F) を直接呼ぶ。
         どちらも OpenCL コンテキストは作らない）で得た PCI アドレスを、sysfs の PCI アドレス
         （/sys/class/drm/cardN/device の realpath 末尾 0000:BB:DD.F）と照合。
      2. 取れなければ sysfs の vendor/device ID と OpenCL デバイス名の対応表（NAME_TABLE）で照合。
         一致するカードがちょうど 1 枚のときだけ採用。0 枚・複数枚なら None（読まない）。
  - 読む値（欠損・不正値はその項目だけ None）:
      mem_info_vram_used（bytes -> MiB）、hwmon/hwmon*/temp1_input（m°C -> °C）、
      hwmon/hwmon*/power1_average（無ければ power1_input、µW -> W）、gpu_busy_percent、pp_dpm_sclk の `*` 行（MHz）。
    電力は GPU 単体（ASIC / ボード）の sysfs 値で、システム全体（壁）の電力ではない。
"""
from __future__ import annotations

import glob
import os
import re
import struct

SYS_DRM = "/sys/class/drm"
AMD_VENDOR = "0x1002"
CL_DEVICE_PCI_BUS_INFO_KHR = 0x410F

# device ID（小文字 4 桁 hex、0x なし）-> OpenCL デバイス名に含まれる（小文字化・空白正規化後の）部分文字列
NAME_TABLE: dict[str, tuple[str, ...]] = {
    "6995": ("wx 2100", "wx2100"),                  # Polaris12 Radeon Pro WX 2100
    "743f": ("rx 6400", "rx 6500", "navi24", "navi 24"),   # Navi24 RX 6400
}


def _read(path: str):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:  # noqa: BLE001
        return None


def _num(path: str):
    """整数ファイルを読む。無い・読めない・数値でなければ None。"""
    v = _read(path)
    if v is None:
        return None
    try:
        return int(v.split()[0])
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------------------
# カード列挙・特定
# --------------------------------------------------------------------------------------

def list_cards(drm_root: str = SYS_DRM) -> list[dict]:
    """/sys/class/drm/card<N>（connector の card0-DP-1 等は除く）のうち vendor が AMD のもの。"""
    cards = []
    for p in sorted(glob.glob(os.path.join(drm_root, "card[0-9]*"))):
        name = os.path.basename(p)
        if not re.fullmatch(r"card\d+", name):
            continue
        dev = os.path.join(p, "device")
        vendor = (_read(os.path.join(dev, "vendor")) or "").lower()
        if vendor != AMD_VENDOR:
            continue
        device = (_read(os.path.join(dev, "device")) or "").lower().replace("0x", "")
        cards.append({"card": name, "device_dir": dev,
                      "pci_addr": os.path.basename(os.path.realpath(dev)).lower(),
                      "vendor": vendor, "device_id": device})
    return cards


def parse_pci_bus_info(v) -> str | None:
    """cl_device_pci_bus_info_khr 相当の値を "dddd:bb:dd.f" にする。解釈できなければ None。
    受け付ける形: 属性 pci_domain/pci_bus/pci_device/pci_function を持つ object、dict、4 要素の tuple/list、
    16 バイト（uint32 x4, little endian）の bytes、"dddd:bb:dd.f" 文字列。"""
    try:
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip().lower() if re.fullmatch(r"([0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", v.strip()) else None
        if isinstance(v, (bytes, bytearray, memoryview)):
            b = bytes(v)
            if len(b) < 16:
                return None
            dom, bus, dev, fn = struct.unpack("<4I", b[:16])
        elif isinstance(v, dict):
            dom, bus, dev, fn = (v["pci_domain"], v["pci_bus"], v["pci_device"], v["pci_function"])
        elif hasattr(v, "pci_bus"):
            dom, bus, dev, fn = (getattr(v, "pci_domain", 0), v.pci_bus, v.pci_device, v.pci_function)
        else:
            dom, bus, dev, fn = tuple(v)[:4]
        dom, bus, dev, fn = int(dom), int(bus), int(dev), int(fn)
        if not (0 <= bus < 256 and 0 <= dev < 32 and 0 <= fn < 8 and 0 <= dom < 65536):
            return None
        return "%04x:%02x:%02x.%d" % (dom, bus, dev, fn)
    except Exception:  # noqa: BLE001
        return None


def opencl_pci_bus_info(dev) -> tuple[str | None, str]:
    """pyopencl の Device から PCI アドレスを得る（コンテキストは作らない）。(addr or None, 経路/理由)。"""
    try:
        import pyopencl as cl  # type: ignore
    except Exception as e:  # noqa: BLE001
        return None, f"pyopencl unavailable: {type(e).__name__}"
    attr = getattr(cl.device_info, "PCI_BUS_INFO_KHR", None)
    if attr is not None:
        try:
            addr = parse_pci_bus_info(dev.get_info(attr))
            if addr:
                return addr, "pyopencl PCI_BUS_INFO_KHR"
        except Exception as e:  # noqa: BLE001
            return None, f"get_info(PCI_BUS_INFO_KHR) failed: {type(e).__name__}: {e}"[:200]
    # pyopencl に属性が無い版: ICD ローダ経由で clGetDeviceInfo を直接呼ぶ（cl_khr_pci_bus_info）
    try:
        import ctypes
        import ctypes.util
        lib = None
        for nm in (ctypes.util.find_library("OpenCL"), "libOpenCL.so.1", "libOpenCL.so"):
            if nm:
                try:
                    lib = ctypes.CDLL(nm)
                    break
                except OSError:
                    continue
        if lib is None:
            return None, "libOpenCL not loadable"
        buf = ctypes.create_string_buffer(16)
        fn = lib.clGetDeviceInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        rc = fn(ctypes.c_void_p(dev.int_ptr), CL_DEVICE_PCI_BUS_INFO_KHR, 16, buf, None)
        if rc != 0:
            return None, f"clGetDeviceInfo(PCI_BUS_INFO_KHR) rc={rc}（未対応）"
        addr = parse_pci_bus_info(buf.raw)
        return (addr, "ctypes clGetDeviceInfo(0x410F)") if addr else (None, "pci_bus_info 値が解釈不能")
    except Exception as e:  # noqa: BLE001
        return None, f"ctypes pci_bus_info failed: {type(e).__name__}: {e}"[:200]


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower())


def identify_card(device_name: str = "", pci_addr: str | None = None, drm_root: str = SYS_DRM,
                  pci_reason: str | None = None) -> dict:
    """OpenCL デバイス（名前 / PCI アドレス）に対応する sysfs カードを特定する。
    戻り値: {"card","device_dir","pci_addr","device_id","method","reason"}。特定できなければ card=None, device_dir=None。"""
    cards = list_cards(drm_root)
    res = {"card": None, "device_dir": None, "pci_addr": None, "device_id": None, "method": None,
           "reason": None, "candidates": [c["card"] for c in cards], "opencl_name": device_name,
           "opencl_pci_bus_info": pci_addr, "pci_bus_info_note": pci_reason}

    def pick(c, method):
        res.update(card=c["card"], device_dir=c["device_dir"], pci_addr=c["pci_addr"], device_id=c["device_id"],
                   method=method, reason=None)
        return res

    if not cards:
        res["reason"] = "AMD(0x1002) の drm card が sysfs に無い"
        return res
    if pci_addr:
        want = pci_addr.lower()
        hit = [c for c in cards if c["pci_addr"] == want or (want.count(":") == 1 and c["pci_addr"].endswith(want))]
        if len(hit) == 1:
            return pick(hit[0], "pci_bus_info")
        res["reason"] = f"pci_bus_info {pci_addr} に一致する amdgpu card が {len(hit)} 枚（推測しない）"
        return res
    name = _norm_name(device_name)
    hit = [c for c in cards if any(k in name for k in NAME_TABLE.get(c["device_id"], ()))]
    if len(hit) == 1:
        return pick(hit[0], "name_table")
    if not hit:
        res["reason"] = f"デバイス名 {device_name!r} が対応表（{sorted(NAME_TABLE)}）のどの AMD card とも一致しない"
    else:
        res["reason"] = f"デバイス名 {device_name!r} に一致する card が {len(hit)} 枚（{[c['card'] for c in hit]}）で一意でない"
    return res


# --------------------------------------------------------------------------------------
# 読み取り
# --------------------------------------------------------------------------------------

def _hwmon_value(device_dir: str, fname: str):
    """hwmon*/<fname> のうち最初に数値で読めたもの（raw 整数）。"""
    for p in sorted(glob.glob(os.path.join(device_dir, "hwmon", "hwmon*", fname))):
        v = _num(p)
        if v is not None:
            return v
    return None


def parse_sclk(text: str | None) -> float | None:
    """pp_dpm_sclk の `*` が付いた行の MHz。"""
    if not text:
        return None
    for line in text.splitlines():
        if "*" in line:
            m = re.search(r"(\d+(?:\.\d+)?)\s*m?hz", line, re.I)
            if m:
                return float(m.group(1))
    return None


def read_amd(device_dir: str | None) -> dict:
    """1 回分の読み取り。キー: vram_mb, temp_c, power_w, busy_pct, sclk_mhz（読めない項目は None）。"""
    out = {"vram_mb": None, "temp_c": None, "power_w": None, "busy_pct": None, "sclk_mhz": None}
    if not device_dir:
        return out
    used = _num(os.path.join(device_dir, "mem_info_vram_used"))
    if used is not None:
        out["vram_mb"] = used / (1024 * 1024)
    t = _hwmon_value(device_dir, "temp1_input")
    if t is not None:
        out["temp_c"] = t / 1000.0
    p = _hwmon_value(device_dir, "power1_average")
    if p is None:
        p = _hwmon_value(device_dir, "power1_input")
    if p is not None:
        out["power_w"] = p / 1e6
    b = _num(os.path.join(device_dir, "gpu_busy_percent"))
    if b is not None:
        out["busy_pct"] = float(b)
    out["sclk_mhz"] = parse_sclk(_read(os.path.join(device_dir, "pp_dpm_sclk")))
    return out


def read_by_name(device_name: str, drm_root: str = SYS_DRM) -> dict:
    """名前ベース特定 + 読み取り（OpenCL コンテキスト不要。確認用スクリプト向け）。"""
    ident = identify_card(device_name, None, drm_root)
    return {"identification": {k: v for k, v in ident.items() if k != "device_dir"}, "values": read_amd(ident["device_dir"])}
