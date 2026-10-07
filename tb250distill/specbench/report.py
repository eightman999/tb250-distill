"""specbench の集計。run が出した results.jsonl / status.jsonl / telemetry.csv / plan.json から
summary.json と summary.md を作る。

  python -m tb250distill.specbench.report runs/specbench/<名前> [--reference <config名>]

指標:
  - gen tok/s      : llama-server の timings.predicted_per_second の中央値（生成だけの速度）。
  - wall tok/s     : tokens_predicted / 壁時計（HTTP 往復込み）の中央値。
  - speedup        : reference 比（gen tok/s の中央値同士の比）。wall 基準も出す。
  - 受理率         : Σdraft_n_accepted / Σdraft_n。
  - 受理長（推定） : 受理率 × n_max。1 回の検証あたりの受理トークン数の推定（検証回数は取れないので、
                     ドラフト長が常に n_max だと仮定。p_min で打ち切ると過大/過小になりうる）。
  - J/token        : Σenergy_j / Σtokens_predicted（prompt 処理と待ちも含む。2 枚合計と RX/WX 内訳）。
  - match_ref      : content の sha256 が reference の最小 rep と一致した割合。greedy でもデバイス/量子化経路の
                     浮動小数差で分岐しうるので、失敗ではなく指標として扱う。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

CARDS = ("rx", "wx")


# --------------------------------------------------------------------------------------
# 読み込み
# --------------------------------------------------------------------------------------

def load_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue            # 書き込み途中で中断された最終行は捨てる
    except FileNotFoundError:
        pass
    return rows


def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def _f(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def load_telemetry_peaks(path) -> dict[str, dict[str, dict]]:
    """telemetry.csv -> {config: {card: {"vram_peak_mb", "gtt_peak_mb", "temp_max_c", "power_max_w"}}}。
    gtt_mb 列の無い古い CSV では gtt_peak_mb は None。"""
    out: dict[str, dict[str, dict]] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = out.setdefault(row.get("config", ""), {}).setdefault(
                    row.get("card", ""), {"vram_peak_mb": None, "gtt_peak_mb": None, "temp_max_c": None, "power_max_w": None})
                for src, dst in (("vram_mb", "vram_peak_mb"), ("gtt_mb", "gtt_peak_mb"), ("temp_c", "temp_max_c"),
                                 ("power_w", "power_max_w")):
                    v = _f(row.get(src))
                    if v is not None and (d[dst] is None or v > d[dst]):
                        d[dst] = v
    except FileNotFoundError:
        pass
    return out


# --------------------------------------------------------------------------------------
# 集計
# --------------------------------------------------------------------------------------

def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _ratio(a, b):
    return None if a is None or b is None or b == 0 else a / b


def _timing(r: dict, key: str):
    return _f((r.get("timings") or {}).get(key))


def _sum_over(rows: list[dict], energy_key: str):
    """energy が取れた request だけで Σenergy / Σtokens（J/token）。"""
    e = t = 0.0
    n = 0
    for r in rows:
        ev, tok = _f(r.get(energy_key)), _f(r.get("tokens_predicted"))
        if ev is None or not tok:
            continue
        e += ev
        t += tok
        n += 1
    return (e / t) if n and t > 0 else None


def reference_hashes(rows: list[dict], reference: str) -> dict[str, tuple[int, str]]:
    """reference config の（prompt ごと）最小 rep の ok record の (rep, sha256)。"""
    best: dict[str, tuple[int, str]] = {}
    for r in rows:
        if r.get("config") != reference or not r.get("ok") or not r.get("sha256"):
            continue
        pid, rep = r["prompt_id"], int(r.get("rep", 0))
        if pid not in best or rep < best[pid][0]:
            best[pid] = (rep, r["sha256"])
    return best


def match_stats(rows: list[dict], reference: str, ref_hash: dict[str, tuple[int, str]]):
    """(一致数, 比較数)。reference 自身の基準 record（最小 rep）は比較対象から除く。"""
    ok = tot = 0
    for r in rows:
        pid = r.get("prompt_id")
        if pid not in ref_hash or not r.get("sha256"):
            continue
        ref_rep, ref_sha = ref_hash[pid]
        if r.get("config") == reference and int(r.get("rep", 0)) == ref_rep:
            continue
        tot += 1
        ok += r["sha256"] == ref_sha
    return ok, tot


def summarize_config(name: str, rows: list[dict], status: dict | None, cfg: dict | None, peaks: dict,
                     reference: str, ref_hash: dict) -> dict:
    ok_rows = [r for r in rows if r.get("ok")]
    n_fail = len(rows) - len(ok_rows)
    gen = [_timing(r, "predicted_per_second") for r in ok_rows]
    wall = [(_f(r.get("tokens_predicted")) / _f(r["wall_s"])) if _f(r.get("wall_s")) and r.get("tokens_predicted") else None
            for r in ok_rows]
    draft_n = sum(int(_timing(r, "draft_n") or 0) for r in ok_rows)
    draft_acc = sum(int(_timing(r, "draft_n_accepted") or 0) for r in ok_rows)
    accept_rate = _ratio(draft_acc, draft_n)
    n_max = (cfg or {}).get("n_max")
    if n_max is None and (cfg or {}).get("spec_type") == "draft-simple":
        n_max = 3
    mok, mtot = match_stats(ok_rows, reference, ref_hash)
    if status is not None:
        st = status.get("status")
    else:
        st = "ok" if ok_rows else "not_run"
    p = peaks.get(name, {})
    return {
        "name": name,
        "note": (cfg or {}).get("note", ""),
        "spec_type": (cfg or {}).get("spec_type"),
        "status": st,
        "load_s": (status or {}).get("load_s"),
        "n_ok": len(ok_rows),
        "n_fail": n_fail,
        "gen_tps": median(gen),
        "wall_tps": median(wall),
        "speedup": None,
        "speedup_wall": None,
        "draft_n": draft_n,
        "draft_n_accepted": draft_acc,
        "accept_rate": accept_rate,
        "accept_len_est": (accept_rate * n_max) if accept_rate is not None and n_max else None,
        "prompt_tps": median([_timing(r, "prompt_per_second") for r in ok_rows]),
        "j_per_token": _sum_over(ok_rows, "energy_j"),
        "j_per_token_rx": _sum_over(ok_rows, "energy_j_rx"),
        "j_per_token_wx": _sum_over(ok_rows, "energy_j_wx"),
        "vram_peak_mb_rx": (p.get("rx") or {}).get("vram_peak_mb"),
        "vram_peak_mb_wx": (p.get("wx") or {}).get("vram_peak_mb"),
        "gtt_peak_mb_rx": (p.get("rx") or {}).get("gtt_peak_mb"),
        "gtt_peak_mb_wx": (p.get("wx") or {}).get("gtt_peak_mb"),
        "temp_max_c_rx": (p.get("rx") or {}).get("temp_max_c"),
        "temp_max_c_wx": (p.get("wx") or {}).get("temp_max_c"),
        "match_ref_rate": _ratio(mok, mtot),
        "match_ref_n": mok,
        "match_ref_total": mtot,
        "temp_wait_s": sum(_f(r.get("temp_wait_s")) or 0.0 for r in rows),
        "log_tail": (status or {}).get("log_tail"),
        "error": (status or {}).get("error"),
    }


def summarize(out_dir, reference: str | None = None) -> dict:
    out = Path(out_dir)
    plan = load_json(out / "plan.json", {}) or {}
    rows = load_jsonl(out / "results.jsonl")
    status_rows = load_jsonl(out / "status.jsonl")
    peaks = load_telemetry_peaks(out / "telemetry.csv")
    reference = reference or plan.get("reference")

    cfg_by_name = {c["name"]: c for c in plan.get("configs", [])}
    order = [c["name"] for c in plan.get("configs", [])]
    for r in rows + status_rows:
        if r.get("config") and r["config"] not in order:
            order.append(r["config"])
    status_by_name: dict[str, dict] = {}
    for s in status_rows:
        status_by_name[s["config"]] = s       # 同じ config が複数あれば最後（再開後）が有効

    ref_hash = reference_hashes(rows, reference) if reference else {}
    configs = []
    for name in order:
        crows = [r for r in rows if r.get("config") == name]
        configs.append(summarize_config(name, crows, status_by_name.get(name), cfg_by_name.get(name), peaks,
                                        reference or "", ref_hash))
    ref = next((c for c in configs if c["name"] == reference), None)
    for c in configs:
        if ref is not None:
            c["speedup"] = _ratio(c["gen_tps"], ref["gen_tps"])
            c["speedup_wall"] = _ratio(c["wall_tps"], ref["wall_tps"])

    # カテゴリ別 gen tok/s の中央値
    categories = sorted({r["category"] for r in rows if r.get("ok") and r.get("category")})
    cat_table: dict[str, dict[str, float | None]] = {}
    for name in order:
        cat_table[name] = {
            cat: median([_timing(r, "predicted_per_second") for r in rows
                         if r.get("config") == name and r.get("ok") and r.get("category") == cat])
            for cat in categories}
    return {
        "plan": plan.get("plan"),
        "reference": reference,
        "args": plan.get("args"),
        "n_requests": len(rows),
        "configs": configs,
        "categories": categories,
        "category_gen_tps": cat_table,
    }


# --------------------------------------------------------------------------------------
# 出力
# --------------------------------------------------------------------------------------

def fmt(x, nd: int = 2, suffix: str = "") -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and nd == 0:
        return f"{x:.0f}{suffix}"
    return f"{x:.{nd}f}{suffix}" if isinstance(x, (int, float)) else str(x)


def _pair(a, b, nd: int = 2) -> str:
    return f"{fmt(a, nd)} / {fmt(b, nd)}"


def render_md(s: dict) -> str:
    L: list[str] = []
    L.append(f"# specbench summary（plan: {s.get('plan') or '?'}、reference: {s.get('reference') or '?'}）")
    L.append("")
    L.append("gen tok/s = timings.predicted_per_second の中央値、speedup は reference 比。"
             "受理長は受理率 × n_max の推定。J/token・VRAM・GTT・温度は RX 6400 / WX 2100 の順。"
             "GTT は VRAM からシステムメモリへはみ出した分の目安（他プロセスの分も含む）。")
    L.append("")
    L.append("| config | status | gen tok/s | wall tok/s | speedup | 受理率 | 受理長(推定) | prompt tok/s | J/token 合計 (RX / WX) "
             "| VRAM peak MiB (RX / WX) | GTT peak MiB (RX / WX) | 最高温度 ℃ (RX / WX) | match_ref |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|")
    for c in s["configs"]:
        mr = "-" if c["match_ref_rate"] is None else f"{c['match_ref_rate']:.2f} ({c['match_ref_n']}/{c['match_ref_total']})"
        jt = "-" if c["j_per_token"] is None and c["j_per_token_rx"] is None else \
            f"{fmt(c['j_per_token'], 3)} ({_pair(c['j_per_token_rx'], c['j_per_token_wx'], 3)})"
        L.append("| {name} | {status} | {gen} | {wall} | {sp} | {acc} | {al} | {pp} | {jt} | {vr} | {gt} | {tm} | {mr} |".format(
            name=c["name"], status=c["status"], gen=fmt(c["gen_tps"]), wall=fmt(c["wall_tps"]),
            sp=fmt(c["speedup"], 2, "x"),
            acc="-" if c["accept_rate"] is None else f"{c['accept_rate']:.2f}",
            al=fmt(c["accept_len_est"]), pp=fmt(c["prompt_tps"], 1), jt=jt,
            vr=_pair(c["vram_peak_mb_rx"], c["vram_peak_mb_wx"], 0),
            gt=_pair(c.get("gtt_peak_mb_rx"), c.get("gtt_peak_mb_wx"), 0), tm=_pair(c["temp_max_c_rx"], c["temp_max_c_wx"], 0),
            mr=mr))
    L.append("")
    if s["categories"]:
        L.append("## カテゴリ別 gen tok/s（中央値）")
        L.append("")
        L.append("| config | " + " | ".join(s["categories"]) + " |")
        L.append("|---|" + "---:|" * len(s["categories"]))
        for c in s["configs"]:
            row = s["category_gen_tps"].get(c["name"], {})
            L.append(f"| {c['name']} | " + " | ".join(fmt(row.get(cat)) for cat in s["categories"]) + " |")
        L.append("")
    bad = [c for c in s["configs"] if c["status"] not in ("ok",)]
    if bad:
        L.append("## 実行できなかった / 異常終了した config")
        L.append("")
        for c in bad:
            L.append(f"### {c['name']}（{c['status']}）")
            if c.get("error"):
                L.append(f"error: {c['error']}")
            tail = c.get("log_tail") or []
            if tail:
                L.append("")
                L.append("```")
                L.extend(tail[-12:])
                L.append("```")
            L.append("")
    fails = [c for c in s["configs"] if c["n_fail"]]
    if fails:
        L.append("## request エラーがあった config")
        L.append("")
        for c in fails:
            L.append(f"- {c['name']}: 失敗 {c['n_fail']} / 成功 {c['n_ok']}")
        L.append("")
    waits = [c for c in s["configs"] if c["temp_wait_s"]]
    if waits:
        L.append("温度ガードで待った時間: " + ", ".join(f"{c['name']} {c['temp_wait_s']:.0f}s" for c in waits))
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def write_summary(out_dir, reference: str | None = None) -> dict:
    s = summarize(out_dir, reference)
    out = Path(out_dir)
    (out / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "summary.md").write_text(render_md(s), encoding="utf-8")
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="specbench の結果を集計して summary.json / summary.md を出す")
    ap.add_argument("out", help="run の出力ディレクトリ")
    ap.add_argument("--reference", default=None, help="reference config 名（既定は plan.json）")
    args = ap.parse_args(argv)
    if not Path(args.out).is_dir():
        print(f"出力ディレクトリが無い: {args.out}", file=sys.stderr)
        return 2
    s = write_summary(args.out, args.reference)
    print(render_md(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
