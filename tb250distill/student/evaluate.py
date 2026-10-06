"""Student の評価（DESIGN.md「評価指標」一式）。eval.json を出力する。

指標（scores は raw logit、確率は softmax(scores) = T=1）:
  agreement（teacher top-1 一致）, gold accuracy, KL(teacher||student), NLL(gold), Brier（多クラス、gold 有りのみ）,
  ECE（15 bin、信頼度=student max prob。正解=gold 一致 / teacher top-1 一致の 2 種）,
  random baseline（= mean(1/k)）、entropy 比較・「teacher uncertain / student confident」抽出、
  candidate permutation test、robust split の variant 別 agreement/KL と元 item との差、推論 latency/throughput。

CLI: python -m tb250distill.student.evaluate --backend np|cl [--device "GT 430"] --data data/tok/NAME \
        --ckpt runs/.../ckpt/last.npz --out eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

import numpy as np

from . import model as M

ECE_BINS = 15


# --------------------------------------------------------------------------------------
# 基本関数（float64, host）
# --------------------------------------------------------------------------------------

def _mask(k, K):
    return np.arange(K)[None, :] < k[:, None]


def softmax_masked(s, k, T=1.0):
    mask = _mask(k, s.shape[1])
    x = np.where(mask, s / T, -np.inf)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def entropy(p):
    q = np.where(p > 0, p, 1.0)
    return -(p * np.log(q)).sum(axis=1)


def kl_div(pt, ps):
    """KL(pt||ps)。ps は softmax 出力なので 0 になりうる場合は極小値で下駄を履かせる。"""
    eps = 1e-300
    return np.where(pt > 0, pt * (np.log(np.maximum(pt, eps)) - np.log(np.maximum(ps, eps))), 0.0).sum(axis=1)


def composite_loss(scores, t_logits, k, gold, T=M.KD_T, w_kd=M.W_KD_GOLD, w_ce=M.W_CE_GOLD):
    """train.py と同じ損失（item ごと）: gold 有り w_kd*KD + w_ce*CE、無し KD。KD = T^2 KL(softmax(t/T)||softmax(s/T))"""
    pt = softmax_masked(t_logits, k, T)
    ps = softmax_masked(scores, k, T)
    kd = T * T * kl_div(pt, ps)
    p1 = softmax_masked(scores, k)
    has = gold >= 0
    g = np.where(has, gold, 0)
    ce = np.where(has, -np.log(np.maximum(p1[np.arange(len(g)), g], 1e-300)), 0.0)
    return np.where(has, w_kd * kd + w_ce * ce, kd), kd, ce


def ece_score(conf, correct, bins=ECE_BINS):
    n = len(conf)
    if n == 0:
        return float("nan")
    b = np.minimum((conf * bins).astype(int), bins - 1)
    e = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability(conf, correct, bins=ECE_BINS):
    b = np.minimum((conf * bins).astype(int), bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        out.append({"bin": i, "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.any() else None,
                    "acc": float(correct[m].mean()) if m.any() else None})
    return out


# --------------------------------------------------------------------------------------
# 推論
# --------------------------------------------------------------------------------------

def predict_scores(model, sh, keys=None):
    """shard 全体の scores (N,Kmax)。keys (N,Kmax) を与えると候補を並べ替えて推論し、元の順序へ戻して返す。"""
    N, K = sh.n, sh.kmax
    out = np.zeros((N, K))
    bs = model.Bm
    for i in range(0, N, bs):
        idx = np.arange(i, min(N, i + bs))
        bt = M.make_batch(sh, idx, keys=None if keys is None else keys[idx])
        s = model.predict(bt)
        orig = np.zeros_like(s)
        np.put_along_axis(orig, bt.perm, s, axis=1)
        out[idx, :bt.K] = orig
    return out


def core_metrics(scores, sh, uncertain_thr=0.6, confident_thr=0.85, n_examples=20):
    """元の候補順序での指標一式（dict）と per-item 配列（robust 比較用）を返す。"""
    k = sh.k
    N = sh.n
    K = scores.shape[1]
    pt = softmax_masked(sh.t_logits.astype(np.float64), k)
    ps = softmax_masked(scores, k)
    pred = ps.argmax(axis=1)
    tpred = pt.argmax(axis=1)
    agree = (pred == tpred)
    kl = kl_div(pt, ps)
    gold = sh.gold
    hg = gold >= 0
    g = np.where(hg, gold, 0)
    ar = np.arange(N)
    loss, kd, ce = composite_loss(scores, sh.t_logits.astype(np.float64), k, gold)
    rb = float(np.mean(1.0 / k))
    m = {"n": int(N), "n_gold": int(hg.sum()), "mean_k": float(k.mean()),
         "random_baseline": rb,
         "random_baseline_gold": float(np.mean(1.0 / k[hg])) if hg.any() else None,
         "agreement": float(agree.mean()),
         "agreement_minus_random": float(agree.mean() - rb),
         "kl": float(kl.mean()),
         "val_loss": float(loss.mean()),
         "kd_loss": float(kd.mean()),
         "ce_loss": float(ce[hg].mean()) if hg.any() else None,
         "brier_vs_teacher": float(((ps - pt) ** 2).sum(axis=1).mean())}
    if hg.any():
        onehot = (np.arange(K)[None, :] == g[:, None]).astype(np.float64)
        correct = (pred == g)
        m.update({
            "gold_acc": float(correct[hg].mean()),
            "teacher_gold_acc": float((tpred == g)[hg].mean()),
            "nll_gold": float(-np.log(np.maximum(ps[ar, g], 1e-300))[hg].mean()),
            "teacher_nll_gold": float(-np.log(np.maximum(pt[ar, g], 1e-300))[hg].mean()),
            "brier": float((((ps - onehot) ** 2).sum(axis=1))[hg].mean()),
            "teacher_brier": float((((pt - onehot) ** 2).sum(axis=1))[hg].mean()),
            "ece15": ece_score(ps.max(axis=1)[hg], correct[hg].astype(float)),
        })
        m["reliability_gold"] = reliability(ps.max(axis=1)[hg], correct[hg].astype(float))
    else:
        m.update({"gold_acc": None, "teacher_gold_acc": None, "nll_gold": None, "teacher_nll_gold": None,
                  "brier": None, "teacher_brier": None, "ece15": None})
    m["ece15_vs_teacher"] = ece_score(ps.max(axis=1), agree.astype(float))
    # entropy 比較・teacher uncertain / student confident
    hs, ht = entropy(ps), entropy(pt)
    multi = k >= 2
    lk = np.log(np.maximum(k, 2))
    nht, nhs = ht / lk, hs / lk
    smax = ps.max(axis=1)
    tmax = pt.max(axis=1)
    unc_conf = multi & (nht >= uncertain_thr) & (smax >= confident_thr)
    conf_unc = multi & (nht < 1 - uncertain_thr) & (nhs >= uncertain_thr)
    ent = {"student_entropy_mean": float(hs.mean()), "teacher_entropy_mean": float(ht.mean()),
           "student_norm_entropy_mean": float(nhs[multi].mean()),
           "teacher_norm_entropy_mean": float(nht[multi].mean()),
           "student_maxprob_mean": float(smax.mean()), "teacher_maxprob_mean": float(tmax.mean()),
           "entropy_pearson": float(np.corrcoef(hs[multi], ht[multi])[0, 1]) if multi.sum() > 2 else None,
           "thresholds": {"teacher_uncertain_norm_entropy_ge": uncertain_thr,
                          "student_confident_maxprob_ge": confident_thr},
           "teacher_uncertain_n": int((multi & (nht >= uncertain_thr)).sum()),
           "teacher_uncertain_student_confident_n": int(unc_conf.sum()),
           "teacher_uncertain_student_confident_frac_of_uncertain":
               float(unc_conf.sum() / max(1, (multi & (nht >= uncertain_thr)).sum())),
           "teacher_confident_student_uncertain_n": int(conf_unc.sum())}
    order = np.argsort(-(smax - tmax) * unc_conf)[:min(n_examples, int(unc_conf.sum()))]
    ent["teacher_uncertain_student_confident_examples"] = [
        {"item_id": int(sh.item_id[i]), "k": int(k[i]), "teacher_probs": ps_round(pt[i, :k[i]]),
         "student_probs": ps_round(ps[i, :k[i]]), "teacher_norm_entropy": float(nht[i]),
         "student_maxprob": float(smax[i]), "agree": bool(agree[i]),
         "gold": int(gold[i]) if hg[i] else None} for i in order]
    m["entropy"] = ent
    # 位置バイアス診断: 予測位置の分布（student vs teacher vs gold）
    hist = lambda x: (np.bincount(x, minlength=K)[:K] / max(1, len(x))).tolist()
    m["position_hist"] = {"student": hist(pred), "teacher": hist(tpred),
                          "gold": hist(gold[hg]) if hg.any() else None,
                          "tv_student_vs_teacher": float(0.5 * np.abs(np.array(hist(pred)) - np.array(hist(tpred))).sum())}
    per_item = {"agree": agree, "kl": kl, "pred": pred, "gold_correct": np.where(hg, pred == g, -1),
                "maxprob": smax}
    return m, per_item


def ps_round(x):
    return [round(float(v), 4) for v in x]


# --------------------------------------------------------------------------------------
# permutation test / robust / latency
# --------------------------------------------------------------------------------------

def permutation_test(model, sh, base_scores, n_perm=4, seed=12345):
    """候補順序を n_perm 通りランダムに並べ替えて推論し、元の順序へ戻して比較。
    モデルは構造的に permutation 等変なので、差は浮動小数点誤差のみのはず（ラベル/位置を学習していないことの確認）。"""
    k = sh.k
    mask = _mask(k, sh.kmax)
    pb = softmax_masked(base_scores, k)
    rows = []
    for r in range(n_perm):
        keys = np.random.default_rng([seed, r]).random((sh.n, sh.kmax))
        s = predict_scores(model, sh, keys=keys)
        p = softmax_masked(s, k)
        tpred = softmax_masked(sh.t_logits.astype(np.float64), k).argmax(axis=1)
        multi = k >= 2
        # argmax の flip: 上位 2 候補が同点に近い item は除外せず、そのまま数える
        flip = (p.argmax(axis=1) != pb.argmax(axis=1))
        rows.append({
            "perm": r,
            "max_abs_score_diff": float(np.abs(np.where(mask, s - base_scores, 0)).max()),
            "max_abs_prob_diff": float(np.abs(p - pb).max()),
            "mean_kl_vs_identity_order": float(kl_div(pb, p).mean()),
            "argmax_flip_rate": float(flip.mean()),
            "agreement_with_teacher": float((p.argmax(axis=1) == tpred).mean()),
            "agreement_with_teacher_multi": float((p.argmax(axis=1) == tpred)[multi].mean()),
        })
    return {"n_perm": n_perm, "seed": seed, "per_perm": rows,
            "max_abs_prob_diff": max(r["max_abs_prob_diff"] for r in rows),
            "max_argmax_flip_rate": max(r["argmax_flip_rate"] for r in rows),
            "agreement_with_teacher_range": [min(r["agreement_with_teacher"] for r in rows),
                                             max(r["agreement_with_teacher"] for r in rows)]}


def load_variant_meta(rsh, replay_db=None, meta_json=None):
    """robust shard の item_id -> (variant, variant_of)。shard の余分キー → replay DB → JSON の順に探す。"""
    ids = rsh.item_id
    if "variant" in rsh.extra and "variant_of" in rsh.extra:
        return {int(i): (str(v), int(o)) for i, v, o in zip(ids, rsh.extra["variant"], rsh.extra["variant_of"])}
    if replay_db and os.path.isfile(replay_db):
        con = sqlite3.connect(f"file:{replay_db}?mode=ro", uri=True)
        try:
            out = {}
            for i in range(0, len(ids), 500):
                chunk = [int(x) for x in ids[i:i + 500]]
                q = ",".join("?" * len(chunk))
                for iid, v, o in con.execute(f"SELECT item_id, variant, variant_of FROM items WHERE item_id IN ({q})", chunk):
                    if v is not None and o is not None:
                        out[int(iid)] = (str(v), int(o))
            return out
        finally:
            con.close()
    if meta_json and os.path.isfile(meta_json):
        with open(meta_json) as f:
            d = json.load(f)
        return {int(i): (v["variant"], int(v["variant_of"])) for i, v in d.items()}
    return {}


def by_source_metrics(scores, sh, min_n=1):
    """shard に optional キー `source`（文字列配列）があり 2 種類以上なら、source ごとの指標（スカラーのみ）を返す。
    無ければ（従来の shard）空 dict で、既存の出力は変わらない。"""
    src = sh.extra.get("source") if hasattr(sh, "extra") else None
    if src is None or getattr(src, "shape", None) != (sh.n,):
        return {}
    names = sorted(set(str(x) for x in src.tolist()))
    if len(names) < 2:
        return {}
    out = {}
    for nm in names:
        idx = np.where(src == nm)[0]
        if len(idx) < min_n:
            continue
        m, _ = core_metrics(scores[idx], sh.subset(idx))
        out[nm] = {k: v for k, v in m.items() if v is None or isinstance(v, (int, float))}
    return out


def key_metrics(out):
    """eval.json のトップレベル `key_metrics`。massive_test_agreement = test split の source=massive の Teacher top-1 一致率
    （学習時と言い回しの違う候補への汎化の指標。source キーが無い shard / test が無い / massive が無ければ None）。"""
    m = (out.get("by_source", {}).get("test") or {}).get("massive")
    return {"massive_test_agreement": None if m is None else m["agreement"],
            "massive_test_n": None if m is None else m.get("n"),
            "massive_test_random_baseline": None if m is None else m.get("random_baseline")}


def _pred_key(sh, i, j):
    return sh.cand[i, j, :sh.cand_len[i, j]].tobytes()


def robust_report(results, rsh, meta):
    """results: split 名 -> (shard, per_item)。robust split の variant 別 agreement/KL と元 item との差。"""
    if not meta:
        return {"available": False, "reason": "variant メタ情報なし（shard の variant/variant_of キー・replay DB・JSON のいずれも無い）"}
    # item_id -> (shard, row, per_item)
    index = {}
    for name, (sh, pi) in results.items():
        for r, iid in enumerate(sh.item_id):
            index.setdefault(int(iid), (sh, r, pi))
    rsh_, rpi = results["robust"]
    by_var = {}
    for r, iid in enumerate(rsh_.item_id):
        if int(iid) in meta:
            by_var.setdefault(meta[int(iid)][0], []).append(r)
    out = {"available": True, "variants": {}}
    for v, rows in sorted(by_var.items()):
        rows = np.array(rows)
        d = {"n": int(len(rows)), "agreement": float(rpi["agree"][rows].mean()), "kl": float(rpi["kl"][rows].mean())}
        gc = rpi["gold_correct"][rows]
        d["gold_acc"] = float((gc[gc >= 0]).mean()) if (gc >= 0).any() else None
        o_agree, o_kl, o_gc, cons = [], [], [], []
        for r in rows:
            of = meta[int(rsh_.item_id[r])][1]
            if of in index:
                osh, orow, opi = index[of]
                o_agree.append(opi["agree"][orow])
                o_kl.append(opi["kl"][orow])
                o_gc.append(opi["gold_correct"][orow])
                ks = {_pred_key(osh, orow, j) for j in range(osh.k[orow])}
                kv = {_pred_key(rsh_, r, j) for j in range(rsh_.k[r])}
                if ks == kv:
                    cons.append(_pred_key(osh, orow, int(opi["pred"][orow])) == _pred_key(rsh_, r, int(rpi["pred"][r])))
        if o_agree:
            n = len(o_agree)
            sel = [i for i, r in enumerate(rows) if meta[int(rsh_.item_id[r])][1] in index]
            va = rpi["agree"][rows][sel].mean()
            vk = rpi["kl"][rows][sel].mean()
            d.update({"n_paired": n, "paired_variant_agreement": float(va), "paired_original_agreement": float(np.mean(o_agree)),
                      "delta_agreement": float(va - np.mean(o_agree)),
                      "paired_variant_kl": float(vk), "paired_original_kl": float(np.mean(o_kl)),
                      "delta_kl": float(vk - np.mean(o_kl)),
                      "pred_consistency_same_candidate_set": float(np.mean(cons)) if cons else None,
                      "n_consistency": len(cons)})
        out["variants"][v] = d
    return out


def bench_inference(model, sh, n_latency=40, warmup=5, tp_batch=None, reps=5):
    """推論 latency（batch=1）と throughput。時間は 梱包(host) + upload + forward + download の end-to-end。"""
    N = sh.n
    pack, tot = [], []
    for i in range(n_latency + warmup):
        t0 = time.perf_counter()
        bt = M.make_batch(sh, [i % N])
        t1 = time.perf_counter()
        model.predict(bt)
        t2 = time.perf_counter()
        if i >= warmup:
            pack.append(t1 - t0)
            tot.append(t2 - t0)
    tot = np.array(tot) * 1e3
    bs = min(tp_batch or model.Bm, model.Bm, N)
    items = tokens = 0
    t_pred = 0.0
    for r in range(reps + 1):
        idx = (np.arange(bs) + r * bs) % N
        bt = M.make_batch(sh, idx)
        t0 = time.perf_counter()
        model.predict(bt)
        dt = time.perf_counter() - t0
        if r >= 1:
            items += bs
            tokens += bt.n_tokens
            t_pred += dt
    return {"latency_batch1_ms": {"mean": float(tot.mean()), "p50": float(np.percentile(tot, 50)),
                                  "p95": float(np.percentile(tot, 95)), "pack_host_ms_mean": float(np.mean(pack) * 1e3),
                                  "n": int(len(tot))},
            "throughput": {"batch": int(bs), "items_per_s": float(items / t_pred),
                           "tokens_per_s": float(tokens / t_pred), "ms_per_batch": float(t_pred / reps * 1e3)}}


def per_item_latency_ms(model, sh, n=0, warmup=3):
    """item ごとの推論 latency（batch=1、梱包 + upload + forward + download の end-to-end、ms）。
    n>0 なら先頭 n 件だけ測り、残りは NaN。"""
    N = sh.n if not n else min(int(n), sh.n)
    out = np.full(sh.n, np.nan)
    for i in range(min(warmup, N)):
        model.predict(M.make_batch(sh, [i]))
    for i in range(N):
        t0 = time.perf_counter()
        model.predict(M.make_batch(sh, [i]))
        out[i] = (time.perf_counter() - t0) * 1e3
    return out


def dump_preds(model, shards, path, splits=("val", "test"), meta=None, latency_n=0):
    """cascade 用: split ごとの per-item student 確率（元の候補順序・softmax T=1）、teacher 確率、gold、
    batch=1 推論 latency(ms) を npz に保存する。キー: <split>_{item_id,k,probs,t_probs,gold,latency_ms}。"""
    arrs = {}
    for sp in splits:
        sh = shards.get(sp)
        if sh is None or sh.n == 0:
            continue
        s = predict_scores(model, sh)
        arrs[f"{sp}_item_id"] = sh.item_id
        arrs[f"{sp}_k"] = sh.k
        arrs[f"{sp}_probs"] = softmax_masked(s, sh.k)
        arrs[f"{sp}_t_probs"] = softmax_masked(sh.t_logits.astype(np.float64), sh.k)
        arrs[f"{sp}_gold"] = sh.gold
        arrs[f"{sp}_latency_ms"] = per_item_latency_ms(model, sh, n=latency_n)
    arrs["__meta__"] = np.array(json.dumps(meta or {}, ensure_ascii=False, default=str))
    tmp = path + ".tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, path)
    return sorted(k for k in arrs if k != "__meta__")


# --------------------------------------------------------------------------------------
# 統合
# --------------------------------------------------------------------------------------

def evaluate_all(model, shards, n_perm=4, replay_db=None, meta_json=None, latency=True, quick=False):
    """shards: {"val": Shard, "test": Shard, "robust": Shard, ...}。robust は variant 比較、val は permutation test に使う。"""
    out = {"splits": {}}
    results = {}
    scores_by = {}
    for name, sh in shards.items():
        if sh is None or sh.n == 0:
            continue
        s = predict_scores(model, sh)
        scores_by[name] = s
        mtr, pi = core_metrics(s, sh)
        out["splits"][name] = mtr
        results[name] = (sh, pi)
        bs_ = by_source_metrics(s, sh)
        if bs_:
            out.setdefault("by_source", {})[name] = bs_
    out["key_metrics"] = key_metrics(out)
    if not quick:
        pname = "test" if "test" in shards and shards["test"] is not None else "val"
        if pname in scores_by:
            out["permutation_test"] = {"split": pname,
                                       **permutation_test(model, shards[pname], scores_by[pname], n_perm=n_perm)}
        if "robust" in results:
            meta = load_variant_meta(shards["robust"], replay_db, meta_json)
            out["robust"] = robust_report(results, shards["robust"], meta)
        if latency:
            lname = "val" if "val" in shards and shards["val"] is not None else next(iter(results))
            out["inference"] = bench_inference(model, shards[lname])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Student 評価 -> eval.json")
    ap.add_argument("--backend", default="np", choices=["np", "cl"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--lp", type=int, default=None)
    ap.add_argument("--ckpt", required=True, help="checkpoint / init npz")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "robust"])
    ap.add_argument("--replay-db", default=None)
    ap.add_argument("--variant-meta", default=None, help="JSON {item_id: {variant, variant_of}}")
    ap.add_argument("--n-perm", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--out", default="eval.json")
    ap.add_argument("--dump-preds", default=None,
                    help="cascade 用に val/test の per-item student 確率と batch=1 latency を npz へ保存")
    ap.add_argument("--dump-splits", nargs="+", default=["val", "test"])
    ap.add_argument("--dump-latency-n", type=int, default=0, help="latency を測る先頭 item 数（0=全件、残りは NaN）")
    a = ap.parse_args(argv)

    cfg, params, meta = M.load_params_npz(a.ckpt)
    shards = {}
    for sp in a.splits:
        p = M.find_shard(a.data, sp, a.lp)
        if p:
            shards[sp] = M.Shard(p)
    if not shards:
        raise SystemExit(f"no shard found in {a.data} for {a.splits}")
    kmax = max(s.kmax for s in shards.values())
    lp = max(s.lp for s in shards.values())
    lc = max(s.lc for s in shards.values())
    be = M.make_backend(a.backend, a.device)
    model = M.Student(be, cfg, a.batch_size, kmax, train=False, max_lp=lp, max_lc=lc)
    model.set_params(params)
    res = evaluate_all(model, shards, n_perm=a.n_perm, replay_db=a.replay_db, meta_json=a.variant_meta,
                       latency=not a.no_latency)
    res["meta"] = {"ckpt": a.ckpt, "ckpt_meta": {k: v for k, v in meta.items() if k != "config"},
                   "backend": a.backend, "device": getattr(be, "device_name", None), "config": cfg.to_dict(),
                   "n_params": M.param_count(cfg), "data": a.data,
                   "shards": {k: v.path for k, v in shards.items()}}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    if a.dump_preds:
        keys = dump_preds(model, shards, a.dump_preds, splits=a.dump_splits,
                          meta={"ckpt": a.ckpt, "backend": a.backend, "device": getattr(be, "device_name", None),
                                "config": cfg.to_dict(), "n_params": M.param_count(cfg), "data": a.data},
                          latency_n=a.dump_latency_n)
        print("wrote", a.dump_preds, keys)
    for name, m in res["splits"].items():
        print(f"{name}: n={m['n']} agree={m['agreement']:.4f} (random {m['random_baseline']:.4f}) "
              f"kl={m['kl']:.4f} gold_acc={m['gold_acc']} ece={m['ece15']}")
    km = res.get("key_metrics") or {}
    if km.get("massive_test_agreement") is not None:
        print(f"key_metrics: massive_test_agreement={km['massive_test_agreement']:.4f} "
              f"(n={km['massive_test_n']}, random {km['massive_test_random_baseline']:.4f})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
