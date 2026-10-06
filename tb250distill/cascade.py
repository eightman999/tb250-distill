"""Phase 9: cascade 判定（GT430(S) -> GT710(M) -> GT730(L) -> Teacher）のオフライン合成。

GPU は 1 プロセス 1 枚なので、各 run の `coordinator eval`（= student.evaluate --dump-preds）が出す
<run_dir>/preds.npz（val/test の per-item student 確率・teacher 確率・gold・batch=1 推論 latency）を読み、
段階判定をここでオフラインに再現する。GPU も replay DB への書き込みも行わない。

  python -m tb250distill.cascade --plan optimized [--runs-dir runs] [--replay-db data/replay.sqlite]
  python -m tb250distill.cascade --stage S=runs/common/gt430 --stage M=runs/common/gt710 --stage L=runs/common/gt730 ...

判定規則: 段 i の max prob >= τ_i なら採用、未満なら次段へ。全段が不採用なら Teacher（replay DB の teacher 出力 = 正解扱い）。

τ の選択基準（val で決める。test は一切使わない）:
  1. 制約: cascade の teacher top-1 一致率 >= 全段 Teacher 送り（= 一致率 1.0）-- max_agree_drop（既定 0.01 = -1pt）。
  2. その中で Teacher への escalation 率が最小。
  3. 同率なら val 平均 latency が小さい、さらに同率なら τ の合計が大きい（保守的）。
  τ の探索は各段 grid（既定 0.50:0.99:0.01）+ inf（その段を使わない = 実行せず全件を次段へ。latency/energy も無し）の直積を全探索する。
  inf を含むので「全段 Teacher 送り」は常に制約を満たし、解は必ず存在する。
  注意: grid が細かく val が小さいと val への過適合があり得る（test で確認する）。

latency: 段 i まで到達した item は 段 1..i の latency を累積（各段の per-item batch=1 latency）、Teacher 送りは
  さらに teacher.latency_ms（replay DB）を加える。DB に無い item は DB 内の平均、DB が無ければ --teacher-latency-ms、
  それも無ければ teacher latency は不明（平均 latency は null、student 段のみの合計を mean_student_latency_ms に出す）。

energy per decision: NVIDIA 390 の GeForce は電力を取得できないので `energy_per_decision_j` は null。
  別キー energy_per_decision_j_estimated_tdp に「TDP 仮定値 × latency」の【推定値】を出す（実測ではない）。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sqlite3
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STAGES = [("S", "gt430"), ("M", "gt710"), ("L", "gt730")]
# 仮定 TDP（W）。GT730 は版により 23〜38W（既定は中間の 30W）。RX6400 は 53W。
DEFAULT_TDP = {"S": 49.0, "M": 19.0, "L": 30.0, "teacher": 53.0}
TDP_NOTE = ("TDP 仮定値 x latency の推定。実測ではない（NVIDIA 390 GeForce は電力取得不可）。"
            "GT430 49W / GT710 19W / GT730 23-38W（既定30W）/ RX6400 53W、idle・PCIe・CPU 消費は含まない")


# --------------------------------------------------------------------------------------
# 入力
# --------------------------------------------------------------------------------------

def load_preds(path):
    """preds.npz（またはその置かれた run dir）-> ({split: dict}, meta)。"""
    if os.path.isdir(path):
        path = os.path.join(path, "preds.npz")
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"])) if "__meta__" in z.files else {}
    splits = {}
    for key in z.files:
        if key.endswith("_item_id"):
            sp = key[:-len("_item_id")]
            splits[sp] = {"item_id": z[key], "k": z[f"{sp}_k"], "probs": z[f"{sp}_probs"],
                          "t_probs": z[f"{sp}_t_probs"], "gold": z[f"{sp}_gold"],
                          "latency_ms": z[f"{sp}_latency_ms"]}
    meta["path"] = path
    return splits, meta


def teacher_latency_from_db(db_path, item_ids):
    """item_id 配列 -> teacher.latency_ms 配列（無い item は NaN）。DB が開けなければ None。"""
    if not db_path or not os.path.isfile(db_path):
        return None
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    try:
        out = np.full(len(item_ids), np.nan)
        pos = {int(i): j for j, i in enumerate(item_ids)}
        ids = [int(i) for i in item_ids]
        for s in range(0, len(ids), 500):
            chunk = ids[s:s + 500]
            q = ",".join("?" * len(chunk))
            for iid, lat in con.execute(f"SELECT item_id, latency_ms FROM teacher WHERE item_id IN ({q})", chunk):
                if lat is not None:
                    out[pos[int(iid)]] = float(lat)
        return out
    finally:
        con.close()


class SplitData:
    """1 split・全段を item_id で揃えた配列一式。"""

    def __init__(self, names, stage_splits, split, teacher_lat):
        parts = [s[split] for s in stage_splits]
        ids0 = np.sort(parts[0]["item_id"])
        for nm, p in zip(names, parts):
            if not np.array_equal(np.sort(p["item_id"]), ids0):
                raise SystemExit(f"split {split}: 段 {nm} の item_id 集合が他段と一致しない（同じ shard で eval したか確認）")
        self.item_id = ids0
        self.n = len(ids0)
        self.names = list(names)
        self.conf, self.pred, self.agree, self.gok, self.lat = [], [], [], [], []
        ref = None
        for nm, p in zip(names, parts):
            o = np.argsort(p["item_id"], kind="stable")
            k = p["k"][o]
            if ref is None:
                ref = {"k": k, "t": p["t_probs"][o], "gold": p["gold"][o]}
            elif not np.array_equal(ref["k"], k):
                raise SystemExit(f"split {split}: 段 {nm} の k（候補数）が他段と一致しない")
            probs = p["probs"][o]
            self.conf.append(probs.max(axis=1))
            pr = probs.argmax(axis=1)
            self.pred.append(pr)
            lat = np.asarray(p["latency_ms"][o], dtype=np.float64)
            if np.isnan(lat).all():
                raise SystemExit(f"split {split}: 段 {nm} の latency が全て NaN")
            self.lat.append(np.where(np.isnan(lat), np.nanmean(lat), lat))   # 一部だけ測った場合は平均で補う
        self.k = ref["k"]
        self.gold = ref["gold"]
        self.t_pred = ref["t"].argmax(axis=1)
        self.t_probs = ref["t"]
        hg = self.gold >= 0
        self.has_gold = hg
        for pr in self.pred:
            self.agree.append(pr == self.t_pred)
            self.gok.append(np.where(hg, pr == self.gold, False))
        self.t_gok = np.where(hg, self.t_pred == self.gold, False)
        if teacher_lat is None:
            self.t_lat, self.t_lat_known = np.zeros(self.n), False
        else:
            tl = np.asarray(teacher_lat, dtype=np.float64)
            # teacher_lat は parts[0] ではなく ids0（ソート済み）順で与えられる
            if np.isnan(tl).all():
                self.t_lat, self.t_lat_known = np.zeros(self.n), False
            else:
                self.t_lat, self.t_lat_known = np.where(np.isnan(tl), np.nanmean(tl), tl), True


# --------------------------------------------------------------------------------------
# cascade 評価・探索
# --------------------------------------------------------------------------------------

def _tdp_vec(names, tdp):
    return [tdp.get(n) for n in names], tdp.get("teacher")


def evaluate_cascade(d, taus, tdp):
    """taus: 各段の閾値（inf 可）。per-item の最終判定から全指標を計算する。"""
    n_st = len(d.names)
    rem = np.ones(d.n, bool)
    final_agree = np.zeros(d.n, bool)
    final_gok = np.zeros(d.n, bool)
    lat = np.zeros(d.n)
    en = np.zeros(d.n)
    en_ok = True
    tdps, t_tdp = _tdp_vec(d.names, tdp)
    stages = []
    for i, tau in enumerate(taus):
        if np.isinf(tau):   # τ=inf: その段は使わない（実行もしない = latency/energy 無し）
            stages.append({"name": d.names[i], "tau": None, "skipped": True, "reach_rate": 0.0, "accept_rate": 0.0,
                           "escalation_rate": None, "n_reach": 0, "n_accept": 0, "agreement_among_accepted": None})
            continue
        reach = rem.copy()
        lat[reach] += d.lat[i][reach]
        if tdps[i] is None:
            en_ok = False
        else:
            en[reach] += tdps[i] * d.lat[i][reach] / 1e3
        acc = reach & (d.conf[i] >= tau)
        final_agree[acc] = d.agree[i][acc]
        final_gok[acc] = d.gok[i][acc]
        stages.append({"name": d.names[i], "tau": None if np.isinf(tau) else float(tau),
                       "reach_rate": float(reach.mean()), "accept_rate": float(acc.mean()),
                       "escalation_rate": float(1 - acc.sum() / max(1, reach.sum())),
                       "n_reach": int(reach.sum()), "n_accept": int(acc.sum()),
                       "agreement_among_accepted": float(d.agree[i][acc].mean()) if acc.any() else None})
        rem &= ~acc
    lat[rem] += d.t_lat[rem]
    if t_tdp is None:
        en_ok = False
    else:
        en[rem] += t_tdp * d.t_lat[rem] / 1e3
    final_agree[rem] = True                    # Teacher 自身の判定 = teacher top-1
    final_gok[rem] = d.t_gok[rem]
    student_lat = lat - np.where(rem, d.t_lat, 0.0)
    hg = d.has_gold
    out = {"n": int(d.n), "n_gold": int(hg.sum()),
           "accuracy_gold": float(final_gok[hg].mean()) if hg.any() else None,
           "agreement": float(final_agree.mean()),
           "mean_latency_ms": float(lat.mean()) if d.t_lat_known else None,
           "mean_student_latency_ms": float(student_lat.mean()),
           "teacher_latency_known": bool(d.t_lat_known),
           "teacher_escalation_rate": float(rem.mean()),
           "stages": stages,
           "energy_per_decision_j": None,
           "energy_per_decision_j_estimated_tdp": float(en.mean()) if (en_ok and d.t_lat_known) else None}
    return out


def single_baselines(d, tdp):
    """各段単独 / Teacher 単独の指標（cascade と同じ項目）。"""
    out = {}
    hg = d.has_gold
    for i, nm in enumerate(d.names):
        e = tdp.get(nm)
        out[nm] = {"accuracy_gold": float(d.gok[i][hg].mean()) if hg.any() else None,
                   "agreement": float(d.agree[i].mean()),
                   "mean_latency_ms": float(d.lat[i].mean()),
                   "teacher_escalation_rate": 0.0,
                   "energy_per_decision_j": None,
                   "energy_per_decision_j_estimated_tdp": None if e is None else float((e * d.lat[i] / 1e3).mean())}
    e = tdp.get("teacher")
    out["Teacher"] = {"accuracy_gold": float(d.t_gok[hg].mean()) if hg.any() else None, "agreement": 1.0,
                      "mean_latency_ms": float(d.t_lat.mean()) if d.t_lat_known else None,
                      "teacher_escalation_rate": 1.0, "energy_per_decision_j": None,
                      "energy_per_decision_j_estimated_tdp":
                          float((e * d.t_lat / 1e3).mean()) if (e is not None and d.t_lat_known) else None}
    return out


def parse_grid(s):
    a, b, c = (float(x) for x in s.split(":"))
    n = int(round((b - a) / c)) + 1
    return [round(a + i * c, 10) for i in range(n)]


def search_taus(d, grid, max_drop, max_combos=3_000_000):
    """val 上で全探索して τ を選ぶ（モジュール docstring の選択基準）。"""
    grid = list(grid) + [float("inf")]
    n_st = len(d.names)
    combos = len(grid) ** n_st
    if combos > max_combos:
        raise SystemExit(f"探索 {combos} 通りは多すぎる。--tau-grid を粗くする（例 0.5:0.99:0.03）")
    acc_mask = [np.stack([d.conf[i] >= g for g in grid]) for i in range(n_st)]   # (G, N)
    t_lat_sum_vec = d.t_lat
    N = d.n
    min_agree = 1.0 - max_drop
    best = {"key": None}
    stats = {"n_combos": combos, "n_feasible": 0}

    def rec(i, rem, agree_cnt, lat_sum, taus):
        if i == n_st:
            n_rem = int(rem.sum())
            agree = (agree_cnt + n_rem) / N
            if agree + 1e-12 < min_agree:
                return
            stats["n_feasible"] += 1
            lat = lat_sum + float(t_lat_sum_vec[rem].sum())
            key = (n_rem / N, lat / N, -sum(min(t, 1e6) for t in taus))
            if best["key"] is None or key < best["key"]:
                best.update({"key": key, "taus": list(taus), "val_agreement": agree})
            return
        lat_here = lat_sum + float(d.lat[i][rem].sum())
        for gi, g in enumerate(grid):
            if np.isinf(g):   # 段を使わない: そのまま次段へ（latency も加算しない）
                rec(i + 1, rem, agree_cnt, lat_sum, taus + [g])
                continue
            a = rem & acc_mask[i][gi]
            ac = agree_cnt + int((a & d.agree[i]).sum())
            rec(i + 1, rem & ~a, ac, lat_here, taus + [g])

    rec(0, np.ones(N, bool), 0, 0.0, [])
    stats["selection"] = {"constraint": f"cascade agreement >= {min_agree:.4f} (= all-teacher 1.0 - {max_drop})",
                          "objective": "min teacher escalation rate, tie: min mean latency, tie: max sum(tau)"}
    return best["taus"], stats


# --------------------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------------------

def parse_kv(items, cast=str):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = cast(v)
    return out


def run(a):
    stage_paths = parse_kv(a.stage)
    if not stage_paths:
        if not a.plan:
            raise SystemExit("--plan か --stage NAME=PATH が必要")
        stage_paths = {nm: os.path.join(a.runs_dir, a.plan, key) for nm, key in DEFAULT_STAGES}
    names = list(stage_paths)
    loaded = []
    metas = {}
    for nm in names:
        sp, meta = load_preds(stage_paths[nm])
        loaded.append(sp)
        metas[nm] = {"preds": meta.get("path"), "device": meta.get("device"), "backend": meta.get("backend"),
                     "n_params": meta.get("n_params"), "ckpt": meta.get("ckpt")}
    tdp = dict(DEFAULT_TDP)
    tdp.update(parse_kv(a.tdp, float))
    t_lat_default = a.teacher_latency_ms

    data = {}
    for sp in (a.select_split, a.eval_split):
        if any(sp not in s for s in loaded):
            raise SystemExit(f"preds に split {sp!r} が無い（coordinator eval --split で作ったか確認）")
        ids = np.sort(loaded[0][sp]["item_id"])
        tl = teacher_latency_from_db(a.replay_db, ids)
        if tl is None and t_lat_default is not None:
            tl = np.full(len(ids), float(t_lat_default))
        elif tl is not None and np.isnan(tl).all() and t_lat_default is not None:
            tl = np.full(len(ids), float(t_lat_default))
        data[sp] = SplitData(names, loaded, sp, tl)

    grid = parse_grid(a.tau_grid)
    dv, dt = data[a.select_split], data[a.eval_split]
    t0 = time.time()
    taus, sstats = search_taus(dv, grid, a.max_agree_drop)
    sstats["search_seconds"] = round(time.time() - t0, 2)
    res = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "plan": a.plan, "stages": names,
           "stage_sources": metas, "select_split": a.select_split, "eval_split": a.eval_split,
           "thresholds": {nm: (None if np.isinf(t) else t) for nm, t in zip(names, taus)},
           "threshold_note": "None = その段を使わない（全件を次段へ）。全て None なら全件 Teacher",
           "search": {"grid": [grid[0], grid[-1], len(grid)], "max_agree_drop": a.max_agree_drop, **sstats},
           "assumptions": {"tdp_w": tdp, "energy_note": TDP_NOTE,
                           "energy_per_decision_j": "null（NVIDIA 390 GeForce は電力取得不可）",
                           "teacher_latency_source": ("replay DB teacher.latency_ms" if a.replay_db and os.path.isfile(a.replay_db)
                                                      else ("--teacher-latency-ms" if t_lat_default is not None else "不明")),
                           "latency_definition": "各段 batch=1 の end-to-end（梱包+upload+forward+download）を到達段まで累積 + teacher.latency_ms"},
           "splits": {a.select_split: {"cascade": evaluate_cascade(dv, taus, tdp), "single": single_baselines(dv, tdp)},
                      a.eval_split: {"cascade": evaluate_cascade(dt, taus, tdp), "single": single_baselines(dt, tdp)}}}
    out = a.out or os.path.join(a.runs_dir, "cascade", f"cascade_{a.plan or 'custom'}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print_summary(res, a.eval_split)
    print("wrote", out)
    return res


def _fmt(x, f=".4f"):
    return "n/a" if x is None else format(x, f)


def print_summary(res, split):
    print(f"thresholds (selected on {res['select_split']}): {res['thresholds']}  "
          f"[{res['search']['n_feasible']}/{res['search']['n_combos']} feasible]")
    sec = res["splits"][split]
    rows = [("cascade", sec["cascade"])] + list(sec["single"].items())
    print(f"{split}: {'':10}{'gold_acc':>9} {'agree':>7} {'lat_ms':>8} {'T-esc':>6} {'E_est_J':>8}")
    for nm, r in rows:
        lat = r.get("mean_latency_ms")
        print(f"{'':6}{nm:10}{_fmt(r['accuracy_gold']):>9} {_fmt(r['agreement']):>7} {_fmt(lat, '.2f'):>8} "
              f"{_fmt(r['teacher_escalation_rate'], '.3f'):>6} {_fmt(r['energy_per_decision_j_estimated_tdp'], '.4f'):>8}")
    for s in sec["cascade"]["stages"]:
        print(f"   stage {s['name']}: tau={s['tau']}{' (skipped)' if s.get('skipped') else ''} reach={s['reach_rate']:.3f} accept={s['accept_rate']:.3f} "
              f"escalation={_fmt(s['escalation_rate'], '.3f')}")


def build_parser():
    ap = argparse.ArgumentParser(description="cascade 判定（オフライン合成）", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--plan", choices=["common", "optimized"], default=None, help="runs/<plan>/{gt430,gt710,gt730}/preds.npz を S,M,L として読む")
    ap.add_argument("--runs-dir", default=os.path.join(REPO_ROOT, "runs"))
    ap.add_argument("--stage", action="append", help="NAME=preds.npz|run_dir（段の順に繰り返す。--plan より優先）")
    ap.add_argument("--replay-db", default=os.path.join(REPO_ROOT, "data", "replay.sqlite"))
    ap.add_argument("--teacher-latency-ms", type=float, default=None, help="DB に latency が無い場合の仮定値")
    ap.add_argument("--select-split", default="val", help="τ を決める split")
    ap.add_argument("--eval-split", default="test", help="報告する split")
    ap.add_argument("--tau-grid", default="0.50:0.99:0.01", help="start:stop:step（+ inf を自動追加）")
    ap.add_argument("--max-agree-drop", type=float, default=0.01, help="全段 Teacher 送り比の一致率低下の許容（0.01 = -1pt）")
    ap.add_argument("--tdp", action="append", help="NAME=W（例 S=49 M=19 L=30 teacher=53）。energy の【推定】用")
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.runs_dir = os.path.abspath(a.runs_dir)
    return run(a)


if __name__ == "__main__":
    main()
