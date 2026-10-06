"""Candidate Semantic Distillation の診断: Student の「候補意味表現」は Teacher の埋め込みに近いか。MASSIVE test の
未見の言い回し（unseen）で、train に出た文字列（seen）と同じ質の表現が得られているか。

  # 1) 評価専用 Teacher 埋め込み（data/emb_eval/<provider>/pubA/。学習コードは読めない: model.load_sem_dir / train.py が拒否）
  #    先に `python -m tb250distill.teacher.embed serve start --provider <p>` で embedding サーバ（RX 6400）を起動しておく
  python -m tb250distill.diag_sem prepare-eval --db data/replay.sqlite --shard data/tok/pubA \
      --providers qwen3_emb_0p6b qwen3_1p7b [--other-per-source 600]
  # 2) run ごとの診断（np backend の CPU 計算のみ。GPU は使わない）
  python -m tb250distill.diag_sem run --run-dir runs/sem06/common/gt710 [--run-dir ...] [--ckpt PATH --name NAME]
  # 3) 表
  python -m tb250distill.diag_sem summary [--docs docs/DIAG_SEM.md]

文字列集合（すべて train 候補由来か MASSIVE 評価候補）:
  seen_massive  MASSIVE train 候補の文字列（60 intent × 2 言語 × 2 言い回し × 5 ラッパー = 1200）
  seen_<source> train の他 source（synth/jcqa/when2call）から source ごとに seed 固定で抽出した文字列
  unseen        MASSIVE test 候補の文字列（train に出ない。val と同じ集合。240）
intent の対応は public.py の言い回し表（MASSIVE_INTENT_PHRASES × MASSIVE_WRAP）から文字列を再生成して引く（推測しない）。
DB の MASSIVE 全 item で「候補文字列 -> intent」が extra の gold intent / intents_shown と一致することを実行時に検証する。

Student 側の表現（文脈なし = 全層の初期 hidden 0 で候補を GRU に通した最終層の最終 hidden h_c）:
  head   z_c = Linear(h_c)（projection head がある sem run のみ。学習時の loss が見ている空間）
  probe  ridge 回帰 h_c -> Teacher PCA（seen の文字列だけで学習。seen の値は phrase 単位 group K-fold の cross-fit 予測、
         unseen は全 seen で fit したプローブ）。head の無い Baseline A と sem run を同じ尺度で比べるため全 run に適用
  h      h_c そのまま（seen 平均で中心化した cosine。Teacher との cosine は定義できないので近傍・paraphrase 指標だけ）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EVAL_DIRNAME = "emb_eval"
UNK_ID = 1
KS = (1, 5, 10)
CHANCE_FACTOR = 5.0        # 「chance を大きく上回る」= intent retrieval accuracy >= 5 x chance
TEACHER_RATIO = 0.70       # unseen の intent retrieval accuracy が Teacher 空間の 70% 以上
SEEN_RATIO = 0.80          # unseen の cosine が seen の 80% 以上


def _log(msg):
    print(msg, flush=True)


def _sha1_json(obj):
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _r(x, n=4):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


# --------------------------------------------------------------------------------------
# MASSIVE: 文字列 -> intent（public.py の言い回し表から再生成）
# --------------------------------------------------------------------------------------

def massive_string_table():
    """候補文字列 -> [{intent, lang, tier, phrase, wrapper}, ...]。tier 0 = 学習用、1 = 評価用。
    public.conv_massive の `wrapper.format(d=phrase)` と同じ生成規則（表の全組合せ）。"""
    from tb250distill.data import public as P
    tbl = {}
    for intent, d in P.MASSIVE_INTENT_PHRASES.items():
        for lang in ("ja", "en"):
            for tier in (0, 1):
                for pi, ph in enumerate(d[lang][tier]):
                    for wi, w in enumerate(P.MASSIVE_WRAP[lang][tier]):
                        tbl.setdefault(w.format(d=ph), []).append(
                            {"intent": intent, "lang": lang, "tier": tier, "phrase": pi, "wrapper": wi})
    return tbl


def verify_massive_table(conn, tbl):
    """DB の MASSIVE item 全件で、候補文字列 -> intent の対応が extra（gold intent / intents_shown）・lang・tier と一致することを確認。
    戻り値 dict(items, candidates, ambiguous_candidates, mismatches)。mismatches が空でなければ呼び出し側が失敗させる。"""
    from tb250distill.data import public as P
    n_items = n_cand = n_amb = 0
    bad = []
    for r in conn.execute("SELECT item_id, split, lang, candidates, gold, extra FROM items WHERE source='massive'"):
        ex = json.loads(r["extra"]) if r["extra"] else {}
        cands = json.loads(r["candidates"])
        n_items += 1
        intents = []
        for j, c in enumerate(cands):
            n_cand += 1
            ent = tbl.get(c)
            if ent is None:
                bad.append((int(r["item_id"]), j, "string not in table"))
                continue
            if len(ent) != 1:
                n_amb += 1
                intents.append(None)
                continue
            e = ent[0]
            if e["lang"] != r["lang"] or e["tier"] != P.tier(r["split"]):
                bad.append((int(r["item_id"]), j, f"lang/tier mismatch {e['lang']}/{e['tier']} vs {r['lang']}/{P.tier(r['split'])}"))
            intents.append(e["intent"])
        if r["gold"] is not None and intents[int(r["gold"])] is not None and ex.get("intent") not in (None, intents[int(r["gold"])]):
            bad.append((int(r["item_id"]), int(r["gold"]), f"gold intent {intents[int(r['gold'])]} != extra {ex.get('intent')}"))
        if ex.get("intents_shown") and None not in intents and sorted(intents) != sorted(ex["intents_shown"]):
            bad.append((int(r["item_id"]), -1, "candidate intents != extra.intents_shown"))
    return {"items": n_items, "candidates": n_cand, "ambiguous_candidates": n_amb, "mismatches": bad[:20], "n_mismatches": len(bad)}


# --------------------------------------------------------------------------------------
# 評価専用文字列の収集と Teacher 埋め込み
# --------------------------------------------------------------------------------------

def collect_eval_strings(conn, train_strings, other_per_source=600, seed=0, other_sources=None):
    """戻り値: (strings, roles)。roles[i] = {sets, source, in_train, intent, lang, tier, phrase, wrapper}。
    sets: massive_train / massive_val / massive_test / other_<source>（seed 固定の抽出）。"""
    tbl = massive_string_table()
    train_set = set(train_strings)
    sets = {}                      # string -> set of names
    src_of = {}
    other_pool = {}                # source -> ordered unique strings（train split）
    for r in conn.execute("SELECT item_id, split, source, candidates FROM items WHERE split IN ('train','val','test') "
                          "ORDER BY item_id"):
        cands = json.loads(r["candidates"])
        if r["source"] == "massive":
            for c in cands:
                sets.setdefault(c, set()).add(f"massive_{r['split']}")
                src_of[c] = "massive"
        elif r["split"] == "train":
            pool = other_pool.setdefault(r["source"], {})
            for c in cands:
                pool.setdefault(c, None)
    rng = np.random.default_rng(seed)
    for sname in sorted(other_pool):
        if other_sources is not None and sname not in other_sources:
            continue
        cand = [c for c in other_pool[sname] if c not in sets and c in train_set]
        if not cand:
            continue
        pick = rng.choice(len(cand), size=min(other_per_source, len(cand)), replace=False)
        for i in sorted(pick.tolist()):
            sets.setdefault(cand[i], set()).add(f"other_{sname}")
            src_of[cand[i]] = sname

    def order(s):
        names = sorted(sets[s])
        return (0 if "massive_train" in names else 1 if "massive_test" in names else 2 if "massive_val" in names else 3, src_of[s], s)

    strings = sorted(sets, key=order)
    roles = []
    for s in strings:
        ent = tbl.get(s) if src_of[s] == "massive" else None
        e = ent[0] if ent and len(ent) == 1 else {}
        roles.append({"sets": sorted(sets[s]), "source": src_of[s], "in_train": s in train_set,
                      "intent": e.get("intent"), "lang": e.get("lang"), "tier": e.get("tier"),
                      "phrase": e.get("phrase"), "wrapper": e.get("wrapper")})
    return strings, roles


def _atomic_json(path, obj):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _atomic_npy(path, arr):
    tmp = str(path) + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def prepare_eval(conn, shard, provider_names, providers, emb_root="data/emb", eval_root="data/emb_eval", other_per_source=600,
                 seed=0, log=_log):
    """評価専用 Teacher 埋め込みを <eval_root>/<provider>/<shard_name>/ に保存する。
    providers: {provider_name: EmbeddingProvider}。学習用 PCA（<emb_root>/<provider>/<shard_name>/pca.npz）で同じ次元へ射影する。
    既存の学習用ディレクトリには何も書かない。出力先のパスに emb_eval を含まなければ拒否（学習コードのガードと対応）。"""
    from tb250distill.teacher import embed as EM
    shard_name = os.path.basename(os.path.normpath(str(shard)))
    if EVAL_DIRNAME not in Path(os.path.abspath(eval_root)).parts:
        raise SystemExit(f"--eval-root のパスに {EVAL_DIRNAME} を含める（学習コードの読み込みガードの対象にするため）: {eval_root}")
    tbl = massive_string_table()
    ver = verify_massive_table(conn, tbl)
    if ver["n_mismatches"]:
        raise SystemExit(f"MASSIVE 文字列表が DB と不整合: {ver}")
    log(f"massive table verified: {ver['items']} items, {ver['candidates']} candidates, ambiguous {ver['ambiguous_candidates']}")
    out_all = {}
    strings = roles = None
    for pname in provider_names:
        base = Path(emb_root) / pname / shard_name
        train_strings = json.loads((base / "strings.json").read_text())
        if strings is None:
            strings, roles = collect_eval_strings(conn, train_strings, other_per_source, seed)
            log(f"eval strings: {len(strings)} "
                + ", ".join(f"{k}={sum(k in r['sets'] for r in roles)}" for k in sorted({s for r in roles for s in r['sets']})))
        pca = np.load(base / "pca.npz")
        out = Path(eval_root) / pname / shard_name
        out.mkdir(parents=True, exist_ok=True)
        sj = out / "strings.json"
        if sj.exists() and json.loads(sj.read_text()) != strings:
            raise SystemExit(f"{sj} が今回の文字列集合と違う（上書きしない。別名の eval-root を使う）")
        if (out / "emb_pca.npy").exists() and (out / "meta.json").exists():
            log(f"[{pname}] 既存の評価専用埋め込みがある。再抽出しない: {out}")
            out_all[pname] = str(out)
            continue
        t0 = time.perf_counter()
        raw = providers[pname].embed(strings).astype(np.float32)
        emb = EM.apply_pca(raw, pca)
        # 取得経路の一貫性: 学習用に既に埋め込んだ文字列（seen）を再計算した値が学習時の値と一致するか
        tr_index = {s: i for i, s in enumerate(train_strings)}
        sel = [i for i, s in enumerate(strings) if s in tr_index]
        old = np.load(base / "emb_raw.npy", mmap_mode="r")
        rows = np.stack([np.asarray(old[tr_index[strings[i]]]) for i in sel]).astype(np.float64)
        new = raw[sel].astype(np.float64)
        cons = (rows * new).sum(1) / (np.linalg.norm(rows, axis=1) * np.linalg.norm(new, axis=1) + 1e-12)
        _atomic_json(out / "strings.json", strings)
        _atomic_json(out / "roles.json", roles)
        _atomic_npy(out / "emb_raw.npy", raw)
        _atomic_npy(out / "emb_pca.npy", emb)
        open(out / "EVAL_ONLY", "w").write("評価専用 Teacher 埋め込み。学習コードは読まない（model.load_sem_dir / train.py が拒否）\n")
        meta = {"eval_only": True, "provider": pname, "shard": shard_name, **providers[pname].describe(),
                "n_strings": len(strings), "D": int(raw.shape[1]), "d": int(emb.shape[1]),
                "pca_from": str(base / "pca.npz"), "pca_note": "学習用 PCA（train unique 集合で fit）をそのまま適用。評価文字列は PCA に使っていない",
                "strings_sha1": _sha1_json(strings), "other_per_source": other_per_source, "seed": seed,
                "massive_table_check": ver,
                "consistency_with_train_embedding": {"n": len(sel), "cos_mean": float(cons.mean()) if len(sel) else None,
                                                     "cos_min": float(cons.min()) if len(sel) else None},
                "seconds": time.perf_counter() - t0, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        _atomic_json(out / "meta.json", meta)
        log(f"[{pname}] saved {out}: {len(strings)} strings, D={raw.shape[1]} d={emb.shape[1]} "
            f"consistency cos mean={meta['consistency_with_train_embedding']['cos_mean']} min={meta['consistency_with_train_embedding']['cos_min']}")
        out_all[pname] = str(out)
    return out_all


def load_eval_dir(path):
    """診断用に評価専用ディレクトリを読む。戻り値: (strings, roles, T[n,d] float32, meta)。"""
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text())
    if not meta.get("eval_only"):
        raise SystemExit(f"{p} は評価専用埋め込みではない（meta.eval_only が無い）")
    strings = json.loads((p / "strings.json").read_text())
    roles = json.loads((p / "roles.json").read_text())
    T = np.load(p / "emb_pca.npy")
    if not (len(strings) == len(roles) == T.shape[0]):
        raise SystemExit(f"{p}: strings/roles/emb の行数が不一致")
    return strings, roles, T, meta


# --------------------------------------------------------------------------------------
# Student: 文脈なしの候補 hidden（train の _sem_pass と同じ計算を numpy で独立に実装）
# --------------------------------------------------------------------------------------

def tokenize_strings(strings, spm_path, lc):
    """tokenize_data.build_shard と同じ規則（`sp.encode(text) or [UNK]` を lc で切る）。戻り値 ids[S,lc] int32, lens[S], full_len[S]。"""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    ids = np.zeros((len(strings), lc), np.int32)
    lens = np.zeros(len(strings), np.int32)
    full = np.zeros(len(strings), np.int32)
    for i, t in enumerate(strings):
        e = sp.encode(t) or [UNK_ID]
        full[i] = len(e)
        e = e[:lc]
        ids[i, :len(e)] = e
        lens[i] = len(e)
    return ids, lens, full


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def candidate_hidden(params, cfg, ids, lens, batch=1024):
    """候補を全層の初期 hidden = 0 で GRU に通した最終層の最終 hidden [S, H] float32（PyTorch 互換 GRU、長さマスクで carry）。"""
    H, L = cfg.hidden, cfg.layers
    S = ids.shape[0]
    out = np.zeros((S, H), np.float32)
    emb = params["emb"].astype(np.float32)
    for s0 in range(0, S, batch):
        sl = slice(s0, min(S, s0 + batch))
        tok, ln = ids[sl], lens[sl]
        B, T = tok.shape
        x = emb[tok]                                   # (B,T,E)
        for l in range(L):
            wi, wh = params[f"l{l}.wi"].astype(np.float32), params[f"l{l}.wh"].astype(np.float32)
            bi, bh = params[f"l{l}.bi"].astype(np.float32), params[f"l{l}.bh"].astype(np.float32)
            gi = x @ wi + bi                           # (B,T,3H)
            h = np.zeros((B, H), np.float32)
            hs = np.zeros((B, T, H), np.float32)
            for t in range(T):
                g = h @ wh + bh
                a = gi[:, t]
                r = _sig(a[:, :H] + g[:, :H])
                z = _sig(a[:, H:2 * H] + g[:, H:2 * H])
                n = np.tanh(a[:, 2 * H:] + r * g[:, 2 * H:])
                hn = (1.0 - z) * n + z * h
                h = np.where((t < ln)[:, None], hn, h)
                hs[:, t] = h
            x = hs
        out[sl] = h
    return out


def head_project(params, h):
    return (h @ params["sem.wp"].astype(np.float32) + params["sem.bp"].astype(np.float32)).astype(np.float32)


# --------------------------------------------------------------------------------------
# 数値部品
# --------------------------------------------------------------------------------------

def l2n(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cos_rows(a, b):
    return (l2n(a) * l2n(b)).sum(1)


def cos_mat(a, b):
    return l2n(a) @ l2n(b).T


def ridge_fit(X, Y, alpha_rel):
    """切片付き ridge。alpha = alpha_rel * mean(diag(Xc^T Xc))。戻り値 (W[d_in,d_out], xm, ym)。"""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    G = Xc.T @ Xc
    a = alpha_rel * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + a * np.eye(G.shape[0]), Xc.T @ Yc)
    return W, xm, ym


def ridge_predict(model, X):
    W, xm, ym = model
    return (np.asarray(X, np.float64) - xm) @ W + ym


def group_folds(groups, k, seed=0):
    """group ごとに fold を割り当てる（同じ group は同じ fold）。戻り値 fold index[n]。"""
    ug = sorted(set(groups), key=lambda g: str(g))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ug))
    gf = {g: int(perm[i] % k) for i, g in enumerate(ug)}
    return np.array([gf[g] for g in groups])


def ridge_crossfit(X, Y, groups, folds=5, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), seed=0):
    """seen のプローブ。alpha を group K-fold の平均 cosine で選び、その alpha の cross-fit 予測と全データ fit のモデルを返す。
    戻り値 (pred_cv[n,d], model_full, alpha, {alpha: cv_cos})。"""
    fold = group_folds(groups, folds, seed)
    best = None
    scores = {}
    for a in alphas:
        pred = np.zeros_like(np.asarray(Y, np.float64))
        for f in range(folds):
            tr, te = fold != f, fold == f
            if not te.any() or not tr.any():
                continue
            pred[te] = ridge_predict(ridge_fit(X[tr], Y[tr], a), X[te])
        sc = float(cos_rows(pred, Y).mean())
        scores[a] = sc
        if best is None or sc > best[0]:
            best = (sc, a, pred)
    _, alpha, pred = best
    return pred, ridge_fit(X, Y, alpha), alpha, scores


def _average_ranks(a):
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), np.float64)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auc_pos_neg(pos, neg):
    """P(pos の値 > neg の値)（同値は 0.5）。Mann-Whitney U。"""
    pos, neg = np.asarray(pos, np.float64).ravel(), np.asarray(neg, np.float64).ravel()
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = _average_ranks(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def topk_idx(sim, k):
    """各行の上位 k の列 index（sim は -inf を含んでよい）。同値の境界は argpartition 任せ。"""
    k = min(k, sim.shape[1])
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]


def recall_at_k(sim_t, sim_s, ks=KS):
    """行ごとに Teacher 空間の上位 k 近傍集合と Student 空間の上位 k 近傍集合の重なり |A∩B|/k の平均。
    sim_* [U, P]（自分自身の列は呼び出し側が -inf にしてある）。戻り値 {k: recall}。"""
    out = {}
    for k in ks:
        a, b = topk_idx(sim_t, k), topk_idx(sim_s, k)
        out[k] = float(np.mean([len(set(x) & set(y)) / min(k, a.shape[1]) for x, y in zip(a, b)]))
    return out


# --------------------------------------------------------------------------------------
# 空間ごとの指標
# --------------------------------------------------------------------------------------

class Sets:
    """roles から作る index 集合。seen_massive / unseen（massive_test で train に出ない）/ other（source ごと）。"""

    def __init__(self, roles, strings=None, full_len=None, lc=16, unseen_split="test"):
        # unseen_split: "test"（既定）| "val"。MASSIVE の val と test は候補文字列集合がほぼ同一（tier=1 の言い回し。239/240 共通）
        S = len(roles)
        usp = f"massive_{unseen_split}"
        self.n = S
        self.seen_massive = np.array([i for i, r in enumerate(roles) if "massive_train" in r["sets"] and r["intent"] is not None], int)
        self.unseen = np.array([i for i, r in enumerate(roles) if usp in r["sets"] and not r["in_train"]
                                and r["intent"] is not None], int)
        self.other = {}
        for i, r in enumerate(roles):
            for s in r["sets"]:
                if s.startswith("other_"):
                    self.other.setdefault(s[len("other_"):], []).append(i)
        self.other = {k: np.array(v, int) for k, v in sorted(self.other.items())}
        self.seen_all = np.concatenate([self.seen_massive] + list(self.other.values())) if self.other else self.seen_massive
        self.intent = np.array([r["intent"] if r["intent"] is not None else "" for r in roles])
        self.lang = np.array([r["lang"] if r["lang"] is not None else "" for r in roles])
        self.group = [(r["intent"], r["lang"], r["tier"], r["phrase"]) if r["source"] == "massive" else ("s", i)
                      for i, r in enumerate(roles)]
        self.truncated = None if full_len is None else (np.asarray(full_len) > lc)
        self.n_unseen_in_train = int(sum((usp in r["sets"]) and r["in_train"] for r in roles))


def paraphrase_metrics(X, sets, label=""):
    """X [S, d]（行は roles と同順）で、unseen × seen_massive の言い換え指標。
    intent_acc: unseen ごとに最近傍の seen_massive の intent が一致する割合（chance = 同 intent の seen の割合の平均）。
    margin: 同 intent ペアの cosine 平均 - 異 intent ペアの平均（全ペア / 同言語だけ）。auc: 同言語で同 intent ペアの cosine が異 intent より高い確率。"""
    u, s = sets.unseen, sets.seen_massive
    sim = cos_mat(X[u], X[s])
    iu, is_ = sets.intent[u], sets.intent[s]
    lu, ls = sets.lang[u], sets.lang[s]
    same = iu[:, None] == is_[None, :]
    samel = lu[:, None] == ls[None, :]
    nn = sim.argmax(1)
    acc = float((is_[nn] == iu).mean())
    chance = float(same.mean(1).mean())
    sim_l = np.where(samel, sim, -np.inf)
    nnl = sim_l.argmax(1)
    acc_l = float((is_[nnl] == iu).mean())
    same_l = same & samel
    chance_l = float((same_l.sum(1) / np.maximum(samel.sum(1), 1)).mean())
    diff_l = (~same) & samel
    return {
        "intent_acc": _r(acc), "intent_chance": _r(chance), "intent_acc_samelang": _r(acc_l), "intent_chance_samelang": _r(chance_l),
        "cos_same_intent": _r(sim[same].mean()), "cos_diff_intent": _r(sim[~same].mean()), "margin": _r(sim[same].mean() - sim[~same].mean()),
        "cos_same_intent_samelang": _r(sim[same_l].mean()), "cos_diff_intent_samelang": _r(sim[diff_l].mean()),
        "margin_samelang": _r(sim[same_l].mean() - sim[diff_l].mean()),
        "auc_samelang": _r(auc_pos_neg(sim[same_l], sim[diff_l])),
        "n_unseen": int(len(u)), "n_seen_massive": int(len(s)),
    }


def seen_paraphrase_metrics(X, sets):
    """seen_massive 同士の intent 検索（学習した文字列の中で intent が整理されているか）。各 seen 文字列について、
    同じ言い回し（intent・言語・tier・phrase が同じ = ラッパー違いの兄弟）を除いた seen_massive の最近傍の intent が一致する割合。"""
    s = sets.seen_massive
    sim = cos_mat(X[s], X[s])
    keys = [sets.group[i] for i in s]
    sib = np.array([[a == b for b in keys] for a in keys])
    sim = np.where(sib, -np.inf, sim)
    inte = sets.intent[s]
    same = (inte[:, None] == inte[None, :]) & ~sib
    nn = sim.argmax(1)
    chance = float((same.sum(1) / np.maximum((~sib).sum(1), 1)).mean())
    return {"intent_acc": _r(float((inte[nn] == inte).mean())), "intent_chance": _r(chance), "n": int(len(s))}


def neighbor_recall(X, T, sets):
    """pool = seen_massive ∪ unseen。unseen ごとに pool から自分自身を除いた Teacher/Student 上位 k 近傍の重なり。"""
    pool = np.concatenate([sets.seen_massive, sets.unseen])
    u = sets.unseen
    st, ss = cos_mat(T[u], T[pool]), cos_mat(X[u], X[pool])
    selfcol = len(sets.seen_massive) + np.arange(len(u))
    st[np.arange(len(u)), selfcol] = -np.inf
    ss[np.arange(len(u)), selfcol] = -np.inf
    rec = recall_at_k(st, ss)
    P = len(pool)
    return {"recall": {f"@{k}": _r(v) for k, v in rec.items()}, "chance": {f"@{k}": _r(k / (P - 1)) for k in KS}, "pool": int(P)}


def teacher_cosine_metrics(X, T, sets, rng_seed=0):
    """X と Teacher（同次元）の行ごと cosine 平均。seen_massive / 他 source / seen 全体 / unseen、と chance（同集合内の別文字列との平均 cosine）。"""
    def one(idx, massive=False):
        c = cos_rows(X[idx], T[idx])
        m = cos_mat(X[idx], T[idx])
        n = len(idx)
        chance = float((m.sum() - np.trace(m)) / max(1, n * (n - 1)))
        out = {"cos": _r(c.mean()), "chance": _r(chance), "n": int(n)}
        if massive:
            # 言語が同じで intent が違う別文字列の Teacher 埋め込みとの cosine 平均（共通成分・言語成分だけで説明できる cosine の水準）。
            # specificity = cos - これ。PCA 空間は共通成分が大きく、chance（言語混在の全ペア）より現実的な下限になる
            oth = (sets.lang[idx][:, None] == sets.lang[idx][None, :]) & (sets.intent[idx][:, None] != sets.intent[idx][None, :])
            co = float(np.mean((m * oth).sum(1) / np.maximum(oth.sum(1), 1)))
            out["cos_other_intent_samelang"] = _r(co)
            out["specificity"] = _r(c.mean() - co)
        return out
    out = {"seen_massive": one(sets.seen_massive, True), "unseen": one(sets.unseen, True)}
    for sname, idx in sets.other.items():
        out[f"seen_{sname}"] = one(idx)
        if sets.truncated is not None:
            tr = idx[sets.truncated[idx]]
            nt = idx[~sets.truncated[idx]]
            if len(tr) > 1:
                out[f"seen_{sname}_truncated"] = one(tr)
            if len(nt) > 1:
                out[f"seen_{sname}_nontruncated"] = one(nt)
    if sets.other:
        c = cos_rows(X[sets.seen_all], T[sets.seen_all])
        out["seen_all_mean_of_strings"] = {"cos": _r(c.mean()), "n": int(len(sets.seen_all))}
    return out


def space_metrics(X, T, sets, comparable):
    """comparable=True（X と T が同次元で Teacher との cosine が意味を持つ空間: head / probe）のとき Teacher cosine も出す。"""
    out = {"neighbors": neighbor_recall(X, T, sets), "paraphrase": paraphrase_metrics(X, sets),
           "seen_paraphrase": seen_paraphrase_metrics(X, sets)}
    if comparable:
        out["teacher_cos"] = teacher_cosine_metrics(X, T, sets)
    return out


def verdict(space, teacher_ref):
    """判定の目安（DIAG の定義）。space = space_metrics の出力（teacher_cos あり）、teacher_ref = Teacher 空間の paraphrase 指標。"""
    p, tc = space["paraphrase"], space["teacher_cos"]
    acc, chance, tacc = p["intent_acc"], p["intent_chance"], teacher_ref["intent_acc"]
    cu, cs = tc["unseen"]["cos"], tc["seen_massive"]["cos"]
    c1 = acc >= TEACHER_RATIO * tacc
    c2 = acc >= CHANCE_FACTOR * chance
    c3 = cu >= SEEN_RATIO * cs
    return {"intent_acc_unseen": acc, "intent_acc_teacher": tacc, "ratio_to_teacher": _r(acc / tacc if tacc else None),
            "intent_chance": chance, "x_chance": _r(acc / chance if chance else None),
            "cos_unseen": cu, "cos_seen_massive": cs, "ratio_unseen_to_seen": _r(cu / cs if cs else None),
            "specificity_unseen": tc["unseen"].get("specificity"), "specificity_seen_massive": tc["seen_massive"].get("specificity"),
            "cond_intent_ge_70pct_teacher": bool(c1), "cond_intent_ge_5x_chance": bool(c2), "cond_cos_unseen_ge_80pct_seen": bool(c3),
            "close_enough": bool(c1 and c2 and c3)}


# --------------------------------------------------------------------------------------
# 判断経路との関係
# --------------------------------------------------------------------------------------

def _spearman(a, b):
    ra, rb = _average_ranks(np.asarray(a, np.float64)), _average_ranks(np.asarray(b, np.float64))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def judgement_relation(items, str_index, Zs, T, preds):
    """MASSIVE test item ごとに: 候補の z_c と「正解候補」（target）の Teacher 埋め込みとの cosine sim_j を作り、
    評価時の判断 score（prefix からの継続 pass の softmax 確率 p_j）との関係を見る。
      target = Teacher の top-1 候補（Teacher 一致率の基準）と dataset gold の 2 通り。
      rep_acc    argmax_j sim_j == target の割合（表現だけで判断した場合の一致率）
      judge_acc  argmax_j p_j == target の割合（実際の判断の一致率）
      P(judge ok | rep ok / rep ng)、spearman(log p_j, sim_j) の item 平均（k>=3）、sim_target - mean(sim_others) の平均
    items: [{item_id, candidates}]、preds: preds.npz（test_item_id, test_probs, test_t_probs, test_gold）。"""
    pid = {int(i): n for n, i in enumerate(preds["test_item_id"])}
    probs, tprobs, gold_all = preds["test_probs"], preds["test_t_probs"], preds["test_gold"]
    out = {}
    rows = []
    for it in items:
        n = pid.get(int(it["item_id"]))
        if n is None:
            continue
        idx = [str_index[c] for c in it["candidates"]]
        k = len(idx)
        rows.append((n, idx, k))
    for tname in ("teacher_top1", "gold"):
        rep_ok, jud_ok, sp, mar, kk = [], [], [], [], []
        for n, idx, k in rows:
            tgt = int(np.argmax(tprobs[n, :k])) if tname == "teacher_top1" else int(gold_all[n])
            if tgt < 0 or tgt >= k:
                continue
            sim = cos_rows(Zs[idx], np.repeat(T[idx[tgt]][None, :], k, axis=0))
            p = probs[n, :k]
            rep_ok.append(int(np.argmax(sim)) == tgt)
            jud_ok.append(int(np.argmax(p)) == tgt)
            kk.append(k)
            if k >= 3:
                sp.append(_spearman(np.log(np.maximum(p, 1e-12)), sim))
            mar.append(sim[tgt] - np.delete(sim, tgt).mean())
        rep_ok, jud_ok = np.array(rep_ok, bool), np.array(jud_ok, bool)
        if not len(rep_ok):
            out[tname] = None
            continue
        rand = float(np.mean(1.0 / np.array(kk)))
        both = int((rep_ok & jud_ok).sum())
        phi_den = np.sqrt(rep_ok.mean() * (1 - rep_ok.mean()) * jud_ok.mean() * (1 - jud_ok.mean()))
        phi = float(((rep_ok & jud_ok).mean() - rep_ok.mean() * jud_ok.mean()) / phi_den) if phi_den > 0 else None
        out[tname] = {
            "n_items": int(len(rep_ok)), "random_baseline": _r(rand), "rep_acc": _r(rep_ok.mean()), "judge_acc": _r(jud_ok.mean()),
            "judge_acc_given_rep_ok": _r(jud_ok[rep_ok].mean()) if rep_ok.any() else None,
            "judge_acc_given_rep_ng": _r(jud_ok[~rep_ok].mean()) if (~rep_ok).any() else None,
            "table": {"rep_ok_judge_ok": both, "rep_ok_judge_ng": int((rep_ok & ~jud_ok).sum()),
                      "rep_ng_judge_ok": int((~rep_ok & jud_ok).sum()), "rep_ng_judge_ng": int((~rep_ok & ~jud_ok).sum())},
            "phi_rep_judge": _r(phi), "spearman_logp_vs_sim_mean": _r(np.nanmean(sp)) if sp else None, "n_spearman_items": int(len(sp)),
            "sim_target_minus_others_mean": _r(np.mean(mar)),
        }
    return out


# --------------------------------------------------------------------------------------
# 例（言い換え・無関係候補の距離）
# --------------------------------------------------------------------------------------

def paraphrase_examples(strings, roles, sets, spaces, T, n_per_lang=2, seed=0, topn=3):
    """unseen から seed 固定で ja/en 各 n_per_lang 件。各空間（Teacher 含む）で:
    同 intent・同言語の seen（言い換え）への cosine 平均/最大、異 intent の seen 平均/最大、最近傍 topn（文字列・intent・cos）。"""
    rng = np.random.default_rng(seed)
    pick = []
    s = sets.seen_massive
    for lang in ("ja", "en"):
        # 同言語・同 intent の seen（言い換え）と、異 intent の seen が両方ある unseen だけを候補にする
        cand = [i for i in sets.unseen if sets.lang[i] == lang
                and ((sets.intent[s] == sets.intent[i]) & (sets.lang[s] == lang)).any()
                and ((sets.intent[s] != sets.intent[i]) & (sets.lang[s] == lang)).any()]
        if not cand:
            continue
        pick += sorted(rng.choice(cand, size=min(n_per_lang, len(cand)), replace=False).tolist())
    out = []
    for u in pick:
        ex = {"string": strings[u], "intent": sets.intent[u], "lang": sets.lang[u], "spaces": {}}
        for name, X in {"teacher": T, **spaces}.items():
            sim = cos_mat(X[[u]], X[s])[0]
            samel = sets.lang[s] == sets.lang[u]
            same = (sets.intent[s] == sets.intent[u]) & samel
            diff = (sets.intent[s] != sets.intent[u]) & samel
            o = np.argsort(-np.where(samel, sim, -np.inf))[:topn]
            ex["spaces"][name] = {
                "paraphrase_cos_mean": _r(sim[same].mean()), "paraphrase_cos_max": _r(sim[same].max()),
                "other_intent_cos_mean": _r(sim[diff].mean()), "other_intent_cos_max": _r(sim[diff].max()),
                "nearest": [{"string": strings[s[j]], "intent": sets.intent[s[j]], "cos": _r(sim[j])} for j in o]}
        out.append(ex)
    return out


# --------------------------------------------------------------------------------------
# run の診断
# --------------------------------------------------------------------------------------

def load_checkpoint(path):
    from tb250distill.student import model as M
    cfg, params, meta = M.load_params_npz(path)
    return cfg, params, meta


def diagnose_checkpoint(ckpt, strings, roles, T, ids_lens, lc, preds=None, test_items=None, str_index=None, seed=0, probe_folds=5,
                        unseen_split="test"):
    """1 つの checkpoint × 1 つの Teacher 埋め込み。戻り値 dict（spaces / verdicts / judgement / examples）。"""
    cfg, params, meta = load_checkpoint(ckpt)
    ids, lens, full = ids_lens
    sets = Sets(roles, strings, full, lc, unseen_split)
    H = candidate_hidden(params, cfg, ids, lens)
    has_head = bool(cfg.sem_dim and "sem.wp" in params)
    seen = sets.seen_all
    Hc = H - H[seen].mean(0)
    # probe: seen だけで学習。seen の値は cross-fit、unseen/その他は全 seen で fit したモデル
    groups = [sets.group[i] for i in seen]
    pred_cv, model, alpha, scores = ridge_crossfit(H[seen].astype(np.float64), T[seen].astype(np.float64), groups, probe_folds, seed=seed)
    P = ridge_predict(model, H).astype(np.float32)
    P[seen] = pred_cv.astype(np.float32)
    res = {"ckpt": str(ckpt), "unseen_split": unseen_split, "step": meta.get("step"), "sem_dim": int(cfg.sem_dim), "has_head": has_head,
           "probe": {"alpha_rel": alpha, "cv_cos_by_alpha": {str(k): _r(v) for k, v in scores.items()}, "folds": probe_folds,
                     "fit_strings": int(len(seen)), "note": "seen は phrase 単位 group K-fold の cross-fit、unseen は全 seen で fit"},
           "sets": {"seen_massive": int(len(sets.seen_massive)), "unseen": int(len(sets.unseen)),
                    "unseen_also_in_train": sets.n_unseen_in_train, **{f"seen_{k}": int(len(v)) for k, v in sets.other.items()}},
           "spaces": {}, "verdict": {}}
    spaces = {"probe": P, "h": Hc}
    if has_head:
        Z = head_project(params, H)
        spaces = {"head": Z, **spaces}
    for name, X in spaces.items():
        res["spaces"][name] = space_metrics(X, T, sets, comparable=name in ("head", "probe"))
    tref = paraphrase_metrics(T, sets)
    res["teacher_ref"] = {"paraphrase": tref, "neighbors": neighbor_recall(T, T, sets), "seen_paraphrase": seen_paraphrase_metrics(T, sets)}
    for name in ("head", "probe"):
        if name in res["spaces"]:
            res["verdict"][name] = verdict(res["spaces"][name], tref)
    primary = "head" if has_head else "probe"
    res["primary_space"] = primary
    res["verdict_primary"] = res["verdict"][primary]
    if preds is not None and test_items:
        res["judgement_relation"] = {"space": primary, **judgement_relation(test_items, str_index, spaces[primary], T, preds)}
        if has_head:
            res["judgement_relation_probe"] = judgement_relation(test_items, str_index, spaces["probe"], T, preds)
    res["examples"] = paraphrase_examples(strings, roles, sets, spaces, T)
    return res


def run_name_from_dir(run_dir):
    p = Path(run_dir).resolve().parts
    return f"{p[-3]}_{p[-1]}" if len(p) >= 3 and p[-2] == "common" else Path(run_dir).name


def run_providers(run_dir, ckpt_cfg, all_providers):
    """sem run は config.json の sem.info.provider（学習に使った Teacher）だけ。head の無い run は全 provider。"""
    if run_dir is not None:
        try:
            cfg = json.loads((Path(run_dir) / "config.json").read_text())
            prov = ((cfg.get("sem") or {}).get("info") or {}).get("provider")
            if prov:
                return [prov]
        except (OSError, ValueError):
            pass
    return list(all_providers)


def load_test_items(conn):
    return [{"item_id": int(r["item_id"]), "candidates": json.loads(r["candidates"])}
            for r in conn.execute("SELECT item_id, candidates FROM items WHERE split='test' AND source='massive' ORDER BY item_id")]


def run_diag(run_dir, name, ckpt, providers_all, eval_root, shard, db, out_dir, emb_root="data/emb", seed=0, log=_log, unseen_split="test"):
    from tb250distill import replay
    shard_name = os.path.basename(os.path.normpath(str(shard)))
    tokmeta = json.loads((Path(shard) / "meta.json").read_text())
    lc = int(tokmeta["lc"])
    spm_path = Path(shard) / "spm.model"
    if tokmeta.get("spm_sha1") and hashlib.sha1(spm_path.read_bytes()).hexdigest() != tokmeta["spm_sha1"]:
        raise SystemExit("spm.model が shard meta の spm_sha1 と違う")
    ckpt = ckpt or str(Path(run_dir) / "ckpt" / "best.npz")
    preds = None
    if run_dir is not None and (Path(run_dir) / "preds.npz").exists():
        preds = np.load(Path(run_dir) / "preds.npz", allow_pickle=False)
    conn = replay.connect(db, readonly=True)
    try:
        test_items = load_test_items(conn)
    finally:
        conn.close()
    cfg0 = load_checkpoint(ckpt)[0]
    if cfg0.lc != lc:
        log(f"warning: ckpt の lc={cfg0.lc} と shard の lc={lc} が違う。shard の lc で切る")
    res = {"run": name, "run_dir": None if run_dir is None else str(run_dir), "providers": {}, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "definitions": {"close_enough": f"unseen intent_acc >= {TEACHER_RATIO} x Teacher 空間の値 かつ >= {CHANCE_FACTOR} x chance "
                                           f"かつ unseen cosine >= {SEEN_RATIO} x seen(massive) cosine"}}
    plist = []
    for pname in run_providers(run_dir, cfg0, providers_all):
        plist.append(pname)
        # <provider>__<variant>（例 qwen3_1p7b__batch32: 別の取得条件の評価専用埋め込み）があれば併せて診断する
        plist += sorted(d.name for d in Path(eval_root).glob(f"{pname}__*") if (d / shard_name / "meta.json").exists())
    for pname in plist:
        strings, roles, T, meta = load_eval_dir(Path(eval_root) / pname / shard_name)
        il = tokenize_strings(strings, spm_path, lc)
        str_index = {s: i for i, s in enumerate(strings)}
        r = diagnose_checkpoint(ckpt, strings, roles, T, il, lc, preds=preds, test_items=test_items, str_index=str_index, seed=seed,
                                unseen_split=unseen_split)
        r["eval_dir"] = str(Path(eval_root) / pname / shard_name)
        res["providers"][pname] = r
        v = r["verdict_primary"]
        log(f"[{name}/{pname}] primary={r['primary_space']} cos seen={v['cos_seen_massive']} unseen={v['cos_unseen']} "
            f"intent_acc={v['intent_acc_unseen']} (teacher {v['intent_acc_teacher']}, chance {v['intent_chance']}) -> "
            f"{'近い' if v['close_enough'] else '近くない'}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / f"{name}.json", res)
    return res


# --------------------------------------------------------------------------------------
# 表（SUMMARY.md / docs/DIAG_SEM.md）
# --------------------------------------------------------------------------------------

def _f(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def _order_key(name):
    pri = {"ref": 0, "baseA": 1, "sem06": 2, "sem17": 3}
    for k, v in pri.items():
        if name.startswith(k):
            return (v, name)
    return (9, name)


def render_summary(results):
    """results: [diag json dict]。Markdown 文字列。"""
    L = []
    w = L.append
    w("# Candidate semantic 診断（DIAG_SEM）\n")
    w(f"生成: {time.strftime('%Y-%m-%d %H:%M:%S%z')}。値はすべて実測（np backend の CPU 計算、各 run の best checkpoint）。\n")
    w("## 読み方\n")
    w("- seen = train 候補に出た文字列（MASSIVE train 1200 件 + 他 source の seed 固定抽出）、unseen = MASSIVE test 候補の文字列（train に 0% 出現、240 件）。")
    w("- 空間: **head** = 学習時に cosine loss を取った projection head 出力（sem run のみ）、**probe** = seen だけで学習した ridge 線形プローブ（h_c -> Teacher PCA。"
      "seen 値は phrase 単位 group 5-fold の cross-fit、unseen は全 seen で fit。head の無い Baseline A と sem run を同じ尺度で比べる）、**h** = h_c そのもの（seen 平均で中心化）。")
    w("- cos = 空間の z と評価専用 Teacher PCA 埋め込み（128 次元）の cosine 平均。chance = 同集合内の別文字列との平均 cosine。")
    w("- R@k = unseen 各文字列の Teacher 空間 top-k 近傍集合と Student 空間 top-k 近傍集合の重なり（pool = seen_massive 1200 + unseen 240、自分自身は除外）。chance = k/(pool-1)。")
    w("- intent acc = unseen の最近傍 seen_massive が同 intent の割合（60 intent、chance ≈ 1/60）。Teacher = Teacher 空間での同じ値（上限の目安）。"
      "margin/AUC は同言語の (unseen, seen_massive) ペアで同 intent と異 intent の cosine を比べた差と、同 intent の cosine が高い確率。")
    w(f"- 判定の目安: 「十分近い」= unseen の intent acc >= Teacher 空間の値の {int(TEACHER_RATIO * 100)}% かつ chance の {CHANCE_FACTOR:g} 倍以上、"
      f"かつ unseen の cos >= seen(massive) の cos の {int(SEEN_RATIO * 100)}%。満たさなければ「近くない（表現自体が学べていない）」。\n")
    results = sorted(results, key=lambda r: _order_key(r["run"]))
    provs = sorted({p for r in results for p in r["providers"]})
    for pn in provs:
        w(f"## Teacher = {pn}\n")
        rows = [(r["run"], r["providers"][pn]) for r in results if pn in r["providers"]]
        w("### A. Teacher 埋め込みとの cosine と近傍の再現（seen / unseen）\n")
        w("| run | 空間 | cos seen(massive) | cos seen(synth) | cos seen(jcqa) | cos seen(w2c) | cos **unseen** | chance(unseen) | R@1 | R@5 | R@10 |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, r in rows:
            for sp, m in r["spaces"].items():
                tc, nb = m.get("teacher_cos"), m["neighbors"]["recall"]
                g = lambda k: _f(tc[k]["cos"]) if tc and k in tc else "-"
                w(f"| {name} | {sp} | {g('seen_massive')} | {g('seen_synth')} | {g('seen_jcqa')} | {g('seen_when2call')} | "
                  f"{('**' + g('unseen') + '**') if tc else '-'} | {_f(tc['unseen']['chance']) if tc else '-'} | "
                  f"{_f(nb['@1'])} | {_f(nb['@5'])} | {_f(nb['@10'])} |")
        if rows:
            nbc = rows[0][1]["spaces"][next(iter(rows[0][1]["spaces"]))]["neighbors"]["chance"]
            w(f"| (chance) | | | | | | | | {_f(nbc['@1'], 4)} | {_f(nbc['@5'], 4)} | {_f(nbc['@10'], 4)} |")
        w("")
        w("### A2. cosine の中身（共通成分で説明できる分を除く）\n")
        w("PCA 空間は言語・文体などの共通成分が大きく、cos だけでは intent を当てているか分からない。「他 intent」= 同言語で intent が違う別文字列の Teacher 埋め込みとの cosine 平均。"
          "specificity = cos(自分の Teacher 埋め込み) - cos(他 intent)。")
        w("| run | 空間 | cos seen | 他intent(seen) | specificity seen | cos unseen | 他intent(unseen) | specificity unseen | spec. unseen/seen |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, r in rows:
            for sp, m in r["spaces"].items():
                tc = m.get("teacher_cos")
                if not tc:
                    continue
                a, b = tc["seen_massive"], tc["unseen"]
                ratio = (b["specificity"] / a["specificity"]) if a.get("specificity") else None
                w(f"| {name} | {sp} | {_f(a['cos'])} | {_f(a['cos_other_intent_samelang'])} | {_f(a['specificity'])} | {_f(b['cos'])} | "
                  f"{_f(b['cos_other_intent_samelang'])} | {_f(b['specificity'])} | {_f(ratio, 2)} |")
        w("")
        w("### B. 言い換えの近さ（unseen -> seen_massive の intent 検索）\n")
        w("seen 内 = 学習した文字列どうし（同じ言い回しのラッパー違いの兄弟は除く）の intent 検索。unseen の値が seen 内より大きく下がれば未見の言い回しへの汎化の問題、"
          "seen 内でも低ければ表現が intent を整理できていない。")
        w("| run | 空間 | intent acc (unseen) | (同言語のみ) | seen 内 intent acc | cos 同intent | cos 異intent | margin(同言語) | AUC(同言語) |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        if rows:
            tr = rows[0][1]["teacher_ref"]["paraphrase"]
            ts = rows[0][1]["teacher_ref"]["seen_paraphrase"]
            w(f"| **Teacher 空間（上限の目安）** | teacher | **{_f(tr['intent_acc'])}** | {_f(tr['intent_acc_samelang'])} | {_f(ts['intent_acc'])} | "
              f"{_f(tr['cos_same_intent'])} | {_f(tr['cos_diff_intent'])} | {_f(tr['margin_samelang'])} | {_f(tr['auc_samelang'])} |")
            w(f"| (chance) | | {_f(tr['intent_chance'], 4)} | {_f(tr['intent_chance_samelang'], 4)} | {_f(ts['intent_chance'], 4)} | | | 0 | 0.5 |")
        for name, r in rows:
            for sp, m in r["spaces"].items():
                p = m["paraphrase"]
                w(f"| {name} | {sp} | {_f(p['intent_acc'])} | {_f(p['intent_acc_samelang'])} | {_f(m['seen_paraphrase']['intent_acc'])} | {_f(p['cos_same_intent'])} | "
                  f"{_f(p['cos_diff_intent'])} | {_f(p['margin_samelang'])} | {_f(p['auc_samelang'])} |")
        w("")
        w("### C. 判定（head があれば head、無ければ probe。probe も併記）\n")
        w("| run | 空間 | intent acc | / Teacher | x chance | cos unseen/seen | 条件1 (>=70% Teacher) | 条件2 (>=5x chance) | 条件3 (cos>=80% seen) | 判定 |")
        w("|---|---|---:|---:|---:|---:|:-:|:-:|:-:|---|")
        for name, r in rows:
            for sp, v in r["verdict"].items():
                ok = lambda b: "OK" if b else "NG"
                w(f"| {name} | {sp} | {_f(v['intent_acc_unseen'])} | {_f(v['ratio_to_teacher'], 2)} | {_f(v['x_chance'], 1)} | "
                  f"{_f(v['cos_unseen'])}/{_f(v['cos_seen_massive'])} = {_f(v['ratio_unseen_to_seen'], 2)} | {ok(v['cond_intent_ge_70pct_teacher'])} | "
                  f"{ok(v['cond_intent_ge_5x_chance'])} | {ok(v['cond_cos_unseen_ge_80pct_seen'])} | "
                  f"{'**十分近い**' if v['close_enough'] else '近くない'} |")
        w("")
        w("### D. 判断経路との関係（MASSIVE test item。表現 z_c と Teacher 正解候補の類似 vs 評価時の判断）\n")
        w("target = Teacher top-1 候補（gold 版は JSON）。rep_acc = argmax_j cos(z_j, Teacher 埋め込み(target)) が target の割合（表現だけで選んだ場合）、"
          "judge_acc = 実際の判断（prefix からの継続 pass）の argmax が target の割合（= Teacher 一致率）。")
        w("| run | 空間 | n | random | rep_acc | judge_acc | judge_acc / rep ok | judge_acc / rep ng | phi(rep,judge) | Spearman(log p, sim) | sim(target)-他 |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, r in rows:
            jr = r.get("judgement_relation")
            if not jr or not jr.get("teacher_top1"):
                continue
            for label, j in (("%s" % jr["space"], jr["teacher_top1"]), ("probe", (r.get("judgement_relation_probe") or {}).get("teacher_top1"))):
                if not j:
                    continue
                w(f"| {name} | {label} | {j['n_items']} | {_f(j['random_baseline'])} | {_f(j['rep_acc'])} | {_f(j['judge_acc'])} | "
                  f"{_f(j['judge_acc_given_rep_ok'])} | {_f(j['judge_acc_given_rep_ng'])} | {_f(j['phi_rep_judge'])} | "
                  f"{_f(j['spearman_logp_vs_sim_mean'])} | {_f(j['sim_target_minus_others_mean'])} |")
        w("")
        w("### E. 例（unseen 文字列、同言語の seen_massive への cosine。paraphrase = 同 intent の train 言い回し 10 件、other = 異 intent）\n")
        for name, r in rows:
            ex = r.get("examples") or []
            if not ex:
                continue
            sp = r["primary_space"]
            w(f"**{name}**（student 空間 = {sp}）\n")
            w("| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |")
            w("|---|---|---|---|---|---|---|")
            for e in ex:
                t, s = e["spaces"]["teacher"], e["spaces"][sp]
                n1 = s["nearest"][0]
                w(f"| {e['string']} | {e['intent']} | {_f(t['paraphrase_cos_mean'])}/{_f(t['paraphrase_cos_max'])} | "
                  f"{_f(t['other_intent_cos_mean'])}/{_f(t['other_intent_cos_max'])} | {_f(s['paraphrase_cos_mean'])}/{_f(s['paraphrase_cos_max'])} | "
                  f"{_f(s['other_intent_cos_mean'])}/{_f(s['other_intent_cos_max'])} | {n1['string']} ({n1['intent']}, {_f(n1['cos'])}) |")
            w("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("prepare-eval")
    pe.add_argument("--db", default="data/replay.sqlite")
    pe.add_argument("--shard", default="data/tok/pubA")
    pe.add_argument("--providers", nargs="+", default=["qwen3_emb_0p6b", "qwen3_1p7b"])
    pe.add_argument("--emb-root", default="data/emb")
    pe.add_argument("--eval-root", default="data/emb_eval")
    pe.add_argument("--other-per-source", type=int, default=600)
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--batch", type=int, default=32)
    pe.add_argument("--urls", nargs="*", default=None, help="provider と同順の embedding サーバ URL（既定: preset のポート）")
    rn = sub.add_parser("run")
    rn.add_argument("--run-dir", action="append", default=[], help="runs/<exp>/common/<gpu>（ckpt/best.npz、preds.npz を使う）")
    rn.add_argument("--ckpt", default=None, help="checkpoint を直接指定（--name 必須。preds が無ければ判断経路の関係は出さない）")
    rn.add_argument("--name", default=None)
    rn.add_argument("--providers", nargs="+", default=["qwen3_emb_0p6b", "qwen3_1p7b"])
    rn.add_argument("--db", default="data/replay.sqlite")
    rn.add_argument("--shard", default="data/tok/pubA")
    rn.add_argument("--eval-root", default="data/emb_eval")
    rn.add_argument("--emb-root", default="data/emb")
    rn.add_argument("--out", default="runs/diag_sem")
    rn.add_argument("--seed", type=int, default=0)
    rn.add_argument("--unseen-split", default="test", choices=["test", "val"],
                    help="unseen 文字列集合の出どころ（MASSIVE test / val。候補文字列はほぼ同一）。モデル選択は val、最終は test")
    sm = sub.add_parser("summary")
    sm.add_argument("--out", default="runs/diag_sem")
    sm.add_argument("--docs", default=None, help="同じ表を書き出す先（例 docs/DIAG_SEM.md）")
    a = ap.parse_args(argv)
    if a.cmd == "prepare-eval":
        from tb250distill import replay
        from tb250distill.teacher import embed as EM
        urls = a.urls or [None] * len(a.providers)
        provs = {p: EM.make_provider(p, url=u, batch=a.batch) for p, u in zip(a.providers, urls)}
        conn = replay.connect(a.db, readonly=True)
        try:
            prepare_eval(conn, a.shard, a.providers, provs, a.emb_root, a.eval_root, a.other_per_source, a.seed)
        finally:
            conn.close()
        return 0
    if a.cmd == "run":
        if a.ckpt and not a.name:
            raise SystemExit("--ckpt には --name が必要")
        specs = [(rd, run_name_from_dir(rd), None) for rd in a.run_dir]
        if a.ckpt:
            specs.append((None, a.name, a.ckpt))
        if not specs:
            raise SystemExit("--run-dir か --ckpt を指定")
        for rd, name, ck in specs:
            run_diag(rd, name, ck, a.providers, a.eval_root, a.shard, a.db, a.out, a.emb_root, a.seed, unseen_split=a.unseen_split)
        return 0
    res = []
    for p in sorted(Path(a.out).glob("*.json")):
        d = json.loads(p.read_text())
        if "providers" in d and "run" in d:
            res.append(d)
    md = render_summary(res)
    (Path(a.out) / "SUMMARY.md").write_text(md)
    if a.docs:
        Path(a.docs).write_text(md)
    print(f"SUMMARY.md: {len(res)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
