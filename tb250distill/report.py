"""最終レポート集計（指示書 §23）。runs/common/* と runs/optimized/* を読み、表と JSON を出す。

  python -m tb250distill.report [--runs-dir runs] [--out-md docs/REPORT_TABLE.md] [--out-json runs/report.json]

読むもの（run dir ごと）: config.json, metrics.csv, eval.json, hardware.json（無い項目は "n/a"）。
  - samples/s   : metrics.csv の samples_per_s の中央値（step 0 の validation 行は除く）
  - Agreement/KL/ECE/gold acc/Brier: eval.json の test split（無ければ val。どちらを使ったかを列に明記）
  - ECE は gold 基準の ece15（gold が無ければ teacher 一致基準の ece15_vs_teacher）
  - VRAM        : eval.json の train_summary.vram_max_mb（無ければ metrics.csv の vram_mb 最大）。nvidia-smi 由来
  - wall-clock / paused_s / max temp: train_summary（無ければ metrics.csv）
  - robust      : eval.json の robust.variants（agreement / KL / 元 item との差）
  - latency     : eval.json の inference.latency_batch1_ms.mean（batch=1、end-to-end）
  - cascade     : runs/cascade/*.json があれば追記
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_INFO = {"gt430": ("GT 430", "Fermi (GF108)"), "gt710": ("GT 710", "Kepler (GK208B)"),
            "gt730": ("GT 730", "Kepler (GK208B)")}
PLAN_PREFIX = {"common": "C", "optimized": "S"}
PLAN_TITLE = {"common": "Common-S（3 GPU で同一モデル・同一条件）", "optimized": "GPU 別最適化 Student"}
NA = "n/a"


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def read_csv(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:  # noqa: BLE001
        return []


def fnum(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def col(rows, name):
    return [v for v in (fnum(r.get(name)) for r in rows) if v is not None]


def get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def hw_summary(hw, gkey):
    """hardware.json の gpus から該当 GPU の driver / PCIe link を拾う（構造が違えば None）。"""
    num = "".join(ch for ch in GPU_INFO.get(gkey, ("", ""))[0] if ch.isdigit())
    if not num or not isinstance(hw, dict):
        return None
    for g in hw.get("gpus", []) if isinstance(hw.get("gpus"), list) else []:
        if isinstance(g, dict) and num in str(g.get("lspci_name", "")):
            link = g.get("link") or {}
            return {"lspci_name": g.get("lspci_name"), "driver": (g.get("driver") or {}).get("version"),
                    "pcie": f"x{link.get('current_width')} {link.get('current_speed')}" if link else None,
                    "vram_MiB": g.get("vram_MiB")}
    return None


def summarize_run(plan, run_dir):
    key = os.path.basename(run_dir)
    cfg = read_json(os.path.join(run_dir, "config.json"), {}) or {}
    ev = read_json(os.path.join(run_dir, "eval.json"), {}) or {}
    hw = read_json(os.path.join(run_dir, "hardware.json"), {}) or {}
    rows = read_csv(os.path.join(run_dir, "metrics.csv"))
    tr = ev.get("train_summary") or {}
    name, arch = GPU_INFO.get(key, (key, NA))
    label = PLAN_PREFIX[plan] + "".join(ch for ch in key if ch.isdigit())
    split = "test" if "test" in (ev.get("splits") or {}) else ("val" if "val" in (ev.get("splits") or {}) else None)
    m = (ev.get("splits") or {}).get(split) or {}
    sps = col(rows, "samples_per_s")
    vram = tr.get("vram_max_mb")
    if vram is None:
        vv = col(rows, "vram_mb")
        vram = max(vv) if vv else None
    tmax = tr.get("temp_max_c")
    if tmax is None:
        tv = col(rows, "temp_c")
        tmax = max(tv) if tv else None
    wall = tr.get("wall_s")
    if wall is None:
        w = col(rows, "wall_s")
        wall = w[-1] if w else None
    paused = tr.get("paused_s")
    if paused is None:
        p = col(rows, "paused_s")
        paused = p[-1] if p else None
    mdl = cfg.get("model") or ev.get("config") or {}
    ece = m.get("ece15")
    if ece is None:
        ece = m.get("ece15_vs_teacher")
    robust = {}
    for v, d in ((ev.get("robust") or {}).get("variants") or {}).items():
        robust[v] = {k: d.get(k) for k in ("n", "agreement", "kl", "delta_agreement", "delta_kl",
                                           "pred_consistency_same_candidate_set")}
    lat = get(ev, "inference", "latency_batch1_ms") or {}
    thr = get(ev, "inference", "throughput") or {}
    kl_rows = [(fnum(r.get("step")), fnum(r.get("val_kl"))) for r in rows if fnum(r.get("val_kl")) is not None]
    return {
        "plan": plan, "run": label, "run_dir": run_dir, "gpu_key": key, "gpu": name, "architecture": arch,
        "n_params": cfg.get("n_params") or ev.get("n_params"),
        "model": mdl or None,
        "backend": cfg.get("backend") or ev.get("backend"),
        "device": cfg.get("device") or ev.get("device"),
        "samples_per_s": statistics.median(sps) if sps else None,
        "tokens_per_s_median": statistics.median(col(rows, "tokens_per_s")) if col(rows, "tokens_per_s") else None,
        "eval_split": split, "eval_n": m.get("n"),
        "agreement": m.get("agreement"), "random_baseline": m.get("random_baseline"),
        "kl": m.get("kl"), "ece": ece, "gold_acc": m.get("gold_acc"), "brier": m.get("brier"),
        "vram_mb": vram, "wall_s": wall, "paused_s": paused, "temp_max_c": tmax,
        "wall_active_s": tr.get("wall_active_s"),
        "latency_batch1_ms_mean": lat.get("mean"), "latency_batch1_ms_p95": lat.get("p95"),
        "infer_items_per_s": thr.get("items_per_s"),
        "robust": robust or None,
        "permutation_max_abs_prob_diff": get(ev, "permutation_test", "max_abs_prob_diff"),
        "steps_done": tr.get("steps_done") or (int(fnum(rows[-1].get("step"))) if rows and fnum(rows[-1].get("step")) is not None else None),
        "total_steps": get(cfg, "plan", "total_steps"),
        "val_kl_first": kl_rows[0][1] if kl_rows else None, "val_kl_last": kl_rows[-1][1] if kl_rows else None,
        "eval_ckpt": get(ev, "meta", "ckpt"),
        "git_commit": cfg.get("git_commit"),
        "hardware": hw_summary(hw, key),
        "has": {"config": bool(cfg), "metrics": bool(rows), "eval": bool(ev), "hardware": bool(hw)},
    }


def collect_runs(runs_dir):
    out = []
    for plan in ("common", "optimized"):
        for d in sorted(glob.glob(os.path.join(runs_dir, plan, "*"))):
            if os.path.isdir(d) and any(os.path.exists(os.path.join(d, f)) for f in ("config.json", "metrics.csv", "eval.json")):
                out.append(summarize_run(plan, d))
    return out


def load_cascades(runs_dir):
    out = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "cascade", "*.json"))):
        d = read_json(p)
        if d and "splits" in d:
            d["_path"] = p
            out.append(d)
    return out


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------

def f(x, fmt=".4f"):
    if x is None:
        return NA
    try:
        return format(x, fmt)
    except (TypeError, ValueError):
        return str(x)


def params_s(n):
    if n is None:
        return NA
    return f"{n / 1e6:.2f}M" if n >= 1e5 else str(n)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def vram_s(v):
    return NA if v is None else f"{v:.0f} MiB"


def render_md(runs, cascades):
    L = ["# TB250 蒸留 最終表", "", f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}（`python -m tb250distill.report`）。値が取れていない項目は n/a。",
         "Agreement / KL / ECE は eval.json の test split（無い場合は val、列 Split に明記）。KL = KL(teacher‖student)、"
         "ECE は gold 基準 15 bin。samples/s は学習中央値、VRAM は nvidia-smi 由来（np backend では n/a）。", ""]
    plans = [p for p in ("common", "optimized") if any(r["plan"] == p for r in runs)]
    if not runs:
        L.append("run が見つからない（runs/common, runs/optimized）。")
    for plan in plans:
        rs = [r for r in runs if r["plan"] == plan]
        L += [f"## §23 最終表: {PLAN_TITLE[plan]}", ""]
        L.append(table(["Run", "GPU", "Architecture", "Student params", "Backend", "samples/s", "Agreement", "KL", "ECE", "VRAM", "Split"],
                       [[r["run"], r["gpu"], r["architecture"], params_s(r["n_params"]), r["backend"] or NA,
                         f(r["samples_per_s"], ".1f"), f(r["agreement"]), f(r["kl"]), f(r["ece"]),
                         vram_s(r["vram_mb"]), r["eval_split"] or NA] for r in rs]))
        L += ["", f"### 補足指標: {PLAN_TITLE[plan]}", ""]
        L.append(table(["Run", "wall-clock", "paused_s", "max temp", "gold acc", "Brier", "random baseline", "latency b=1 (ms)", "p95 (ms)",
                        "steps", "val KL 初→終"],
                       [[r["run"], f(r["wall_s"], ".0f") + (" s" if r["wall_s"] is not None else ""), f(r["paused_s"], ".0f"),
                         f(r["temp_max_c"], ".0f") + (" C" if r["temp_max_c"] is not None else ""), f(r["gold_acc"]),
                         f(r["brier"]), f(r["random_baseline"]), f(r["latency_batch1_ms_mean"], ".2f"),
                         f(r["latency_batch1_ms_p95"], ".2f"),
                         f"{r['steps_done']}/{r['total_steps']}" if r["steps_done"] is not None else NA,
                         f"{f(r['val_kl_first'], '.3f')} -> {f(r['val_kl_last'], '.3f')}"] for r in rs]))
        L += ["", f"### robustness: {PLAN_TITLE[plan]}", ""]
        rrows = []
        for r in rs:
            if not r["robust"]:
                rrows.append([r["run"], NA, NA, NA, NA, NA, NA])
                continue
            for v, d in sorted(r["robust"].items()):
                rrows.append([r["run"], v, d.get("n", NA), f(d.get("agreement")), f(d.get("kl")),
                              f(d.get("delta_agreement"), "+.4f"), f(d.get("delta_kl"), "+.4f")])
        L.append(table(["Run", "variant", "n", "agreement", "KL", "Δagreement vs 元 item", "ΔKL vs 元 item"], rrows))
        L.append("")
    for c in cascades:
        sec = c["splits"].get(c.get("eval_split", "test")) or {}
        cas, single = sec.get("cascade"), sec.get("single") or {}
        if not cas:
            continue
        L += [f"## Cascade ({os.path.basename(c['_path'])}, split={c.get('eval_split')})", "",
              f"段: {' -> '.join(c['stages'])} -> Teacher。閾値（val で選択）: {c['thresholds']}。"
              f"選択基準: {get(c, 'search', 'selection', 'constraint')}; {get(c, 'search', 'selection', 'objective')}。", ""]
        rows = [["cascade", f(cas["accuracy_gold"]), f(cas["agreement"]), f(cas["mean_latency_ms"], ".2f"),
                 f(cas["teacher_escalation_rate"], ".3f"), f(cas["energy_per_decision_j_estimated_tdp"], ".4f")]]
        for nm, s in single.items():
            rows.append([f"{nm} のみ", f(s["accuracy_gold"]), f(s["agreement"]), f(s["mean_latency_ms"], ".2f"),
                         f(s["teacher_escalation_rate"], ".3f"), f(s["energy_per_decision_j_estimated_tdp"], ".4f")])
        L.append(table(["方式", "gold acc", "teacher agreement", "平均 latency (ms)", "Teacher 送り率", "energy/decision 推定 (J)"], rows))
        L += ["", "段ごとの escalation rate（到達 item のうち次段へ送った割合）:", ""]
        L.append(table(["段", "τ", "reach", "accept", "escalation"],
                       [[s["name"], f(s["tau"], ".2f"), f(s["reach_rate"], ".3f"), f(s["accept_rate"], ".3f"),
                         f(s["escalation_rate"], ".3f") + (" (未使用)" if s.get("skipped") else "")] for s in cas["stages"]]))
        L += ["", f"energy per decision（実測）は null（NVIDIA 390 GeForce で電力取得不可）。推定列は {get(c, 'assumptions', 'energy_note', default=NA)}", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="最終レポート集計")
    ap.add_argument("--runs-dir", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--out-md", default=os.path.join(REPO_ROOT, "docs", "REPORT_TABLE.md"))
    ap.add_argument("--out-json", default=None, help="既定 <runs-dir>/report.json")
    a = ap.parse_args(argv)
    runs_dir = os.path.abspath(a.runs_dir)
    runs = collect_runs(runs_dir)
    cascades = load_cascades(runs_dir)
    md = render_md(runs, cascades)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_md)), exist_ok=True)
    with open(a.out_md, "w") as fh:
        fh.write(md)
    out_json = a.out_json or os.path.join(runs_dir, "report.json")
    with open(out_json, "w") as fh:
        json.dump({"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": runs,
                   "cascade": [{k: v for k, v in c.items()} for c in cascades]}, fh, indent=1, ensure_ascii=False)
    print(f"runs: {len(runs)}, cascade: {len(cascades)} -> {a.out_md}, {out_json}")
    return 0


if __name__ == "__main__":
    main()
