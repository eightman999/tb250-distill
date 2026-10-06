"""Wikipedia 事前学習 run と Baseline A の評価比較表（docs/WIKI_PRETRAIN.md 用）。

  python -m tb250distill.student.wiki_compare --wiki runs/wikiA/common/wx2100/eval.json \
      --baseline runs/baseA/common/wx2100/eval.json --baseline-all 'runs/baseA/common/*/eval.json'

各列は eval.json（best val-KL ckpt の評価）から取る。JCQA gold は test split の by_source.jcqa.gold_acc（n=500）。
二項分布の標準誤差（SE = sqrt(p(1-p)/n)）も併記する: 差が ~2 SE 未満なら有意とは言えない。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics


def load(p):
    with open(p) as f:
        return json.load(f)


def _g(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def metrics(e):
    """eval.json -> 比較に使う指標 dict。"""
    t = _g(e, "splits", "test") or {}
    v = _g(e, "splits", "val") or {}
    bs = e.get("by_source") or {}
    var = _g(e, "robust", "variants") or {}
    m = {
        "val_kl": v.get("kl"),
        "test_agree": t.get("agreement"), "test_kl": t.get("kl"), "ece": t.get("ece15"), "test_gold": t.get("gold_acc"),
        "massive_agree": _g(e, "key_metrics", "massive_test_agreement"),
        "jcqa_gold": _g(bs, "test", "jcqa", "gold_acc"), "jcqa_gold_val": _g(bs, "val", "jcqa", "gold_acc"),
        "jcqa_agree": _g(bs, "test", "jcqa", "agreement"),
        "massive_gold": _g(bs, "test", "massive", "gold_acc"), "w2c_gold": _g(bs, "test", "when2call", "gold_acc"),
        "synth_gold": _g(bs, "test", "synth", "gold_acc"),
        "cand_para": _g(var, "cand_paraphrase", "agreement"), "unseen_cand": _g(var, "unseen_cand", "agreement"),
        "perm_consistency": _g(var, "perm", "pred_consistency_same_candidate_set"),
        "jcqa_n": _g(bs, "test", "jcqa", "n"),
    }
    return m


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


COLS = [("val_kl", "best val KL"), ("test_agree", "test agree"), ("test_kl", "test KL"), ("ece", "ECE"),
        ("massive_agree", "MASSIVE agree"), ("jcqa_gold", "JCQA gold (test)"), ("jcqa_gold_val", "JCQA gold (val)"),
        ("massive_gold", "MASSIVE gold"), ("w2c_gold", "W2C gold"), ("synth_gold", "synth gold"),
        ("cand_para", "cand_para agree"), ("unseen_cand", "unseen_cand agree")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--baseline", required=True, help="同一 GPU（WX 2100）の Baseline A eval.json")
    ap.add_argument("--baseline-all", default=None, help="Baseline A 全 GPU の eval.json の glob（平均・sd 用）")
    a = ap.parse_args(argv)
    rows = [("Baseline A (WX 2100)", metrics(load(a.baseline)))]
    allm = []
    if a.baseline_all:
        allm = [metrics(load(p)) for p in sorted(glob.glob(a.baseline_all))]
        mean, sd = {}, {}
        for k, _ in COLS:
            xs = [m[k] for m in allm if m.get(k) is not None]
            mean[k] = statistics.mean(xs) if xs else None
            sd[k] = statistics.stdev(xs) if len(xs) > 1 else None
        rows.append((f"Baseline A 平均 (n={len(allm)})", mean))
    w = metrics(load(a.wiki))
    rows.append(("Wikipedia 事前学習 (WX 2100)", w))
    print("| run | " + " | ".join(h for _, h in COLS) + " |")
    print("|---|" + "---:|" * len(COLS))
    for name, m in rows:
        print(f"| {name} | " + " | ".join(fmt(m.get(k)) for k, _ in COLS) + " |")
    if allm:
        print("\nBaseline A run 間 sd: " + ", ".join(f"{h} {fmt(sd[k], 3)}" for k, h in COLS if sd.get(k) is not None))
    n = w.get("jcqa_n") or 500
    p = w.get("jcqa_gold")
    if p is not None:
        se = math.sqrt(p * (1 - p) / n)
        print(f"\nJCQA gold (test, n={n}) の二項 SE = {se:.3f}（95% 区間 ±{1.96 * se:.3f}）。random = 0.200。")
    b = rows[0][1]
    d = {k: (w[k] - b[k]) for k, _ in COLS if w.get(k) is not None and b.get(k) is not None}
    print("\n差（Wikipedia - Baseline A WX2100）: " + ", ".join(f"{h} {d[k]:+.3f}" for k, h in COLS if k in d))


if __name__ == "__main__":
    main()
