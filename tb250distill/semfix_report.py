"""SEMFIX の表を作る（診断 JSON と各 run の eval.json から Markdown）。

  python -m tb250distill.semfix_report screen --diag runs/semfix/screen/diag [--ckpt best]
  python -m tb250distill.semfix_report prod --root runs/semfix/prod
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _f(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def _head(r):
    sp = r["spaces"]
    return sp.get("head") or sp["probe"], ("head" if "head" in sp else "probe")


def load_diag(path):
    d = json.loads(Path(path).read_text())
    prov = next(iter(d["providers"].values()))
    return d, prov


def screen_rows(diag_dir, ckpt="best"):
    rows = []
    for p in sorted(Path(diag_dir).glob("*.json")):
        m = re.match(r"^(?P<cond>.+?)_(?P<gpu>(?:gt|wx)\d+(?:s\d+)?)_(?P<ck>best|p\d+)$", p.stem)
        if m:
            cond, gpu, ck = m["cond"], m["gpu"], m["ck"]
        elif p.stem.startswith("ref_init"):
            cond, gpu, ck = "ref_init(p000)", "-", "p000"
        else:
            continue
        if ck != ckpt and not cond.startswith("ref_init"):
            continue
        d, prov = load_diag(p)
        h, sp = _head(prov)
        tc = h["teacher_cos"]
        nb = h["neighbors"]["recall"]
        rows.append({"cond": cond, "gpu": gpu, "ck": ck, "space": sp, "split": prov.get("unseen_split") or d.get("unseen_split"),
                     "acc": h["paraphrase"]["intent_acc"], "acc_l": h["paraphrase"]["intent_acc_samelang"], "auc": h["paraphrase"]["auc_samelang"],
                     "spec_u": tc["unseen"]["specificity"], "cos_u": tc["unseen"]["cos"], "r1": nb["@1"], "r5": nb["@5"], "r10": nb["@10"],
                     "seen_acc": h["seen_paraphrase"]["intent_acc"], "spec_s": tc["seen_massive"]["specificity"], "cos_s": tc["seen_massive"]["cos"],
                     "t_acc": prov["teacher_ref"]["paraphrase"]["intent_acc"],
                     "close": prov["verdict_primary"]["close_enough"]})
    return rows


def render_screen(diag_dir, ckpt="best"):
    rows = screen_rows(diag_dir, ckpt)
    L = [f"| 条件 | GPU | ckpt | unseen intent acc | (同言語) | AUC | specificity unseen | cos unseen | R@1 | R@5 | R@10 | seen intent acc | specificity seen | cos seen |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (-(r["acc"] or 0))):
        L.append(f"| {r['cond']} | {r['gpu']} | {r['ck']} | **{_f(r['acc'])}** | {_f(r['acc_l'])} | {_f(r['auc'])} | {_f(r['spec_u'])} | {_f(r['cos_u'])} | "
                 f"{_f(r['r1'])} | {_f(r['r5'])} | {_f(r['r10'])} | {_f(r['seen_acc'])} | {_f(r['spec_s'])} | {_f(r['cos_s'])} |")
    if rows:
        L.append(f"\nTeacher 空間の unseen intent acc = {_f(rows[0]['t_acc'])}（上限の目安）。空間 = head（z。学習時に損失を取った空間）。unseen = MASSIVE {rows[0]['split']} の候補文字列。")
    return "\n".join(L) + "\n"


def prod_rows(root):
    out = []
    for ev in sorted(Path(root).glob("*/*/eval.json")):
        cond, gpu = ev.parent.parent.name, ev.parent.name
        e = json.loads(ev.read_text())
        km = e.get("key_metrics") or {}
        bs = e.get("by_source", {}).get("test", {})
        sp = e.get("splits", {})
        rb = e.get("robust") or {}
        row = {"cond": cond, "gpu": gpu, "massive_agree": km.get("massive_test_agreement"),
               "test_agree": sp.get("test", {}).get("agreement"), "test_kl": sp.get("test", {}).get("kl"),
               "val_kl": sp.get("val", {}).get("kl"),
               "synth_gold": (bs.get("synth") or {}).get("gold_acc"), "w2c_gold": (bs.get("when2call") or {}).get("gold_acc"),
               "jcqa_gold": (bs.get("jcqa") or {}).get("gold_acc"), "massive_gold": (bs.get("massive") or {}).get("gold_acc"),
               "robust": {k: v for k, v in rb.items()} if isinstance(rb, dict) else None,
               "train_summary": (e.get("train_summary") or {})}
        d = ev.parent / "diag_test.json"
        if d.exists():
            _, prov = load_diag(d)
            h, _sp = _head(prov)
            row["diag"] = {"acc": h["paraphrase"]["intent_acc"], "spec_u": h["teacher_cos"]["unseen"]["specificity"],
                           "cos_u": h["teacher_cos"]["unseen"]["cos"], "r5": h["neighbors"]["recall"]["@5"],
                           "seen_acc": h["seen_paraphrase"]["intent_acc"]}
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--diag", default="runs/semfix/screen/diag")
    s.add_argument("--ckpt", default="best")
    p = sub.add_parser("prod")
    p.add_argument("--root", default="runs/semfix/prod")
    a = ap.parse_args(argv)
    if a.cmd == "screen":
        print(render_screen(a.diag, a.ckpt))
    else:
        print(json.dumps(prod_rows(a.root), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
