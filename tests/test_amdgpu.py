"""amdgpu sysfs 監視（fake sysfs ディレクトリ）: パース・カード特定・欠損時 None・電力積算・train 統合。GPU 不要。"""
import json
import os
import struct
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb250distill.hw import amdgpu as A  # noqa: E402
from tb250distill.student import train as T  # noqa: E402


def w(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_card(root, card, pci, vendor, device, *, vram=None, temp=None, pavg=None, pin=None, busy=None, sclk=None,
              hwmon="hwmon2"):
    """root/devices/<pci> に実体、root/drm/<card>/device をそこへのシンボリックリンクにする。"""
    d = root / "devices" / pci
    w(d / "vendor", vendor + "\n")
    w(d / "device", device + "\n")
    if vram is not None:
        w(d / "mem_info_vram_used", f"{vram}\n")
    if temp is not None:
        w(d / "hwmon" / hwmon / "temp1_input", f"{temp}\n")
    if pavg is not None:
        w(d / "hwmon" / hwmon / "power1_average", f"{pavg}\n")
    if pin is not None:
        w(d / "hwmon" / hwmon / "power1_input", f"{pin}\n")
    if busy is not None:
        w(d / "gpu_busy_percent", f"{busy}\n")
    if sclk is not None:
        w(d / "pp_dpm_sclk", sclk)
    c = root / "drm" / card
    c.mkdir(parents=True, exist_ok=True)
    os.symlink(d, c / "device")
    (root / "drm" / f"{card}-DP-1").mkdir(exist_ok=True)   # connector（無視されること）
    return d


SCLK_RX = "0: 500Mhz \n1: 1870Mhz *\n2: 2320Mhz \n"
SCLK_WX = "0: 214Mhz \n1: 734Mhz \n7: 1219Mhz *\n"


@pytest.fixture
def sysfs(tmp_path):
    make_card(tmp_path, "card0", "0000:00:02.0", "0x8086", "0x5902")                       # Intel（対象外）
    make_card(tmp_path, "card1", "0000:03:00.0", "0x1002", "0x743f", vram=1278533632, temp=63000, pavg=16000000,
              busy=87, sclk=SCLK_RX, hwmon="hwmon2")
    make_card(tmp_path, "card2", "0000:06:00.0", "0x1002", "0x6995", vram=408870912, temp=58000, pin=9075000,
              busy=0, sclk=SCLK_WX, hwmon="hwmon3")                                       # power1_average 無し
    return str(tmp_path / "drm")


# ---------------------------------------------------------------- 読み取り
def test_read_rx6400_uses_power_average(sysfs):
    v = A.read_amd(os.path.join(sysfs, "card1", "device"))
    assert v["vram_mb"] == pytest.approx(1278533632 / 1048576)
    assert v["temp_c"] == 63.0 and v["power_w"] == 16.0 and v["busy_pct"] == 87.0 and v["sclk_mhz"] == 1870.0


def test_read_wx2100_falls_back_to_power_input(sysfs):
    v = A.read_amd(os.path.join(sysfs, "card2", "device"))
    assert v["power_w"] == pytest.approx(9.075) and v["temp_c"] == 58.0 and v["sclk_mhz"] == 1219.0
    assert v["busy_pct"] == 0.0


def test_missing_files_give_none(tmp_path):
    d = make_card(tmp_path, "card1", "0000:03:00.0", "0x1002", "0x743f", vram=1048576)   # vram だけ
    v = A.read_amd(str(d))
    assert v["vram_mb"] == 1.0
    assert v["temp_c"] is None and v["power_w"] is None and v["busy_pct"] is None and v["sclk_mhz"] is None
    assert A.read_amd(None)["vram_mb"] is None
    assert A.read_amd(str(tmp_path / "nonexistent"))["temp_c"] is None


def test_garbage_values_give_none(tmp_path):
    d = make_card(tmp_path, "card1", "0000:03:00.0", "0x1002", "0x743f", vram="abc", temp="N/A", pavg="", busy="x",
                  sclk="0: 500Mhz\n1: 900Mhz\n")                                        # `*` 行なし
    v = A.read_amd(str(d))
    assert all(x is None for x in v.values())


def test_power_average_unreadable_falls_back(tmp_path):
    d = make_card(tmp_path, "card1", "0000:03:00.0", "0x1002", "0x743f", pavg="garbage", pin=5000000)
    assert A.read_amd(str(d))["power_w"] == 5.0


def test_parse_sclk():
    assert A.parse_sclk(SCLK_RX) == 1870.0
    assert A.parse_sclk(None) is None and A.parse_sclk("") is None
    assert A.parse_sclk("0: 300MHz *") == 300.0


# ---------------------------------------------------------------- カード特定
def test_parse_pci_bus_info_forms():
    assert A.parse_pci_bus_info(struct.pack("<4I", 0, 3, 0, 0)) == "0000:03:00.0"
    assert A.parse_pci_bus_info((0, 6, 0, 0)) == "0000:06:00.0"
    assert A.parse_pci_bus_info({"pci_domain": 0, "pci_bus": 0x0a, "pci_device": 1, "pci_function": 2}) == "0000:0a:01.2"
    o = types.SimpleNamespace(pci_domain=0, pci_bus=3, pci_device=0, pci_function=0)
    assert A.parse_pci_bus_info(o) == "0000:03:00.0"
    assert A.parse_pci_bus_info("0000:03:00.0") == "0000:03:00.0"
    assert A.parse_pci_bus_info("03:00.0") == "03:00.0"
    assert A.parse_pci_bus_info(None) is None and A.parse_pci_bus_info(b"\x00") is None
    assert A.parse_pci_bus_info((0, 999, 0, 0)) is None and A.parse_pci_bus_info("junk") is None


def test_list_cards_ignores_connectors_and_non_amd(sysfs):
    cards = A.list_cards(sysfs)
    assert [c["card"] for c in cards] == ["card1", "card2"]
    assert cards[0]["pci_addr"] == "0000:03:00.0" and cards[1]["device_id"] == "6995"


def test_identify_by_pci_bus_info(sysfs):
    r = A.identify_card("whatever name", "0000:06:00.0", sysfs)
    assert r["card"] == "card2" and r["method"] == "pci_bus_info"
    r = A.identify_card("", "03:00.0", sysfs)      # domain 無しも末尾一致
    assert r["card"] == "card1"


def test_pci_bus_info_takes_priority_over_name(sysfs):
    # 名前は RX 6400 を示すが PCI は WX 2100 -> PCI を信じる
    r = A.identify_card("AMD Radeon RX 6400", "0000:06:00.0", sysfs)
    assert r["card"] == "card2"


def test_pci_bus_info_without_match_is_none_not_guess(sysfs):
    r = A.identify_card("AMD Radeon RX 6400", "0000:09:00.0", sysfs)
    assert r["card"] is None and r["device_dir"] is None and "推測しない" in r["reason"]


def test_identify_by_name_table(sysfs):
    r = A.identify_card("AMD Radeon Pro WX 2100", None, sysfs)
    assert r["card"] == "card2" and r["method"] == "name_table"
    r = A.identify_card("AMD Radeon RX 6400 (radeonsi, navi24, LLVM 19.1.7, DRM 3.61)", None, sysfs)
    assert r["card"] == "card1" and r["method"] == "name_table"


def test_identify_unknown_name_is_none(sysfs):
    r = A.identify_card("AMD Radeon RX 7900 XTX", None, sysfs)
    assert r["card"] is None and r["reason"]
    assert A.identify_card("", None, sysfs)["card"] is None


def test_identify_ambiguous_same_model_is_none(tmp_path):
    make_card(tmp_path, "card1", "0000:03:00.0", "0x1002", "0x6995")
    make_card(tmp_path, "card2", "0000:06:00.0", "0x1002", "0x6995")
    r = A.identify_card("AMD Radeon Pro WX 2100", None, str(tmp_path / "drm"))
    assert r["card"] is None and "一意でない" in r["reason"]
    # PCI で一意に決まる
    assert A.identify_card("AMD Radeon Pro WX 2100", "0000:06:00.0", str(tmp_path / "drm"))["card"] == "card2"


def test_identify_no_amd_cards(tmp_path):
    make_card(tmp_path, "card0", "0000:00:02.0", "0x8086", "0x5902")
    assert A.identify_card("AMD Radeon RX 6400", None, str(tmp_path / "drm"))["card"] is None
    assert A.identify_card("x", None, str(tmp_path / "nodrm"))["card"] is None


def test_read_by_name(sysfs):
    r = A.read_by_name("AMD Radeon Pro WX 2100", sysfs)
    assert r["identification"]["card"] == "card2" and r["values"]["power_w"] == pytest.approx(9.075)


# ---------------------------------------------------------------- SmiMonitor
class FakeDev:
    def __init__(self, vendor, name):
        self.vendor, self.name = vendor, name


class FakeBE:
    name = "cl"

    def __init__(self, vendor, name, pci=None):
        self.dev = FakeDev(vendor, name)
        self.device_name = name
        self.pci_bus_id = pci


@pytest.fixture
def no_ocl_pci(monkeypatch):
    monkeypatch.setattr(A, "opencl_pci_bus_info", lambda dev: (None, "test: unavailable"))


def test_monitor_amd_by_name(sysfs, no_ocl_pci):
    m = T.SmiMonitor(FakeBE("Advanced Micro Devices, Inc.", "AMD Radeon Pro WX 2100"), 20.0, drm_root=sysfs)
    assert m.kind == "amdgpu" and m.err is None and m.ident["card"] == "card2" and m.ident["method"] == "name_table"
    m.poll(force=True)
    assert m.temp == 58.0 and m.power == pytest.approx(9.075) and m.vram == pytest.approx(408870912 / 1048576)
    assert m.sclk == 1219.0 and m.busy == 0.0 and m.temp_max == 58.0
    info = m.info()
    assert info["kind"] == "amdgpu" and info["amdgpu"]["card"] == "card2" and "GPU 単体" in info["power_source"]
    json.dumps(info)


def test_monitor_amd_unidentified_stays_none(sysfs, no_ocl_pci):
    m = T.SmiMonitor(FakeBE("Advanced Micro Devices, Inc.", "AMD Radeon RX 7900 XTX"), 20.0, drm_root=sysfs)
    m.poll(force=True)
    assert m.temp is None and m.vram is None and m.power is None and m.err and "特定できない" in m.err


def test_monitor_amd_uses_pci_bus_info(sysfs, monkeypatch):
    monkeypatch.setattr(A, "opencl_pci_bus_info", lambda dev: ("0000:03:00.0", "test"))
    m = T.SmiMonitor(FakeBE("AMD", "unknown name"), 20.0, drm_root=sysfs)
    assert m.ident["card"] == "card1" and m.ident["method"] == "pci_bus_info"
    m.poll(force=True)
    assert m.temp == 63.0 and m.power == 16.0


def test_monitor_detects_nvidia_and_np(sysfs):
    m = T.SmiMonitor(FakeBE("NVIDIA Corporation", "NVIDIA GeForce GT 430"), 20.0, drm_root=sysfs)
    assert m.kind == "nvidia" and m.ident is None
    np_be = types.SimpleNamespace(name="np")
    m = T.SmiMonitor(np_be, 20.0)
    assert not m.enabled and m.kind is None
    m.poll(force=True)
    assert m.temp is None


def test_monitor_respects_interval(sysfs, no_ocl_pci, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(T.time, "time", lambda: clock["t"])
    m = T.SmiMonitor(FakeBE("AMD", "AMD Radeon RX 6400"), 20.0, drm_root=sysfs)
    m.poll()
    assert m.temp == 63.0
    p = os.path.realpath(os.path.join(sysfs, "card1", "device")) + "/hwmon/hwmon2/temp1_input"
    open(p, "w").write("70000\n")
    clock["t"] += 5
    m.poll()
    assert m.temp == 63.0                  # interval 内は再読込しない
    clock["t"] += 20
    m.poll()
    assert m.temp == 70.0 and m.temp_max == 70.0


def test_power_integration_and_pause_exclusion(sysfs, no_ocl_pci, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(T.time, "time", lambda: clock["t"])
    m = T.SmiMonitor(FakeBE("AMD", "AMD Radeon RX 6400"), 1.0, drm_root=sysfs)   # power1_average = 16 W
    pf = os.path.realpath(os.path.join(sysfs, "card1", "device")) + "/hwmon/hwmon2/power1_average"
    m.poll(force=True)                       # t=0, 16W
    clock["t"] = 10.0
    open(pf, "w").write("26000000\n")
    m.poll(force=True)                       # t=10, 26W -> 区間 (16+26)/2*10 = 210 J
    assert m.power_mean() == pytest.approx(21.0)
    m.recording = False                      # 温度ガード停止中
    clock["t"] = 100.0
    open(pf, "w").write("1000000\n")
    m.poll(force=True)
    m.recording = True
    clock["t"] = 110.0
    open(pf, "w").write("21000000\n")
    m.poll(force=True)                       # 直前の記録サンプルが無いので区間を作らない
    assert m.power_time_s == pytest.approx(10.0) and m.power_mean() == pytest.approx(21.0)
    assert m.power_max == 26.0 and m.power_samples == 3
    e = T.energy_summary(m, samples=500, wall_active_s=100.0)
    assert e["measured"] and e["power_mean_w"] == pytest.approx(21.0)
    assert e["energy_j"] == pytest.approx(2100.0) and e["energy_per_sample_j"] == pytest.approx(4.2)
    assert "GPU 単体" in e["scope"]


def test_energy_summary_without_power_is_not_estimated():
    e = T.energy_summary(types.SimpleNamespace(power_mean=lambda: None), 100, 50.0)
    assert e["measured"] is False and e["energy_j"] is None and e["energy_per_sample_j"] is None
    e = T.energy_summary(types.SimpleNamespace(), 100, 50.0)    # 旧 FakeSmi 相当
    assert e["measured"] is False


def test_restore_carries_over_resume(sysfs, no_ocl_pci):
    m = T.SmiMonitor(FakeBE("AMD", "AMD Radeon RX 6400"), 1.0, drm_root=sysfs)
    m.restore(1000.0, 50.0)
    assert m.power_mean() == pytest.approx(20.0)
