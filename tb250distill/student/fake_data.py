"""teacher 無しで学習を試すための偽 shard 生成（Tokenized shard 契約と同じ npz 形式）。

teacher logits は「学習可能な構造」を持つ: 各 token に潜在ベクトル（prefix 用 W / 候補用 U、d 次元）があり、
  logit_j = scale * temp * <U 和(候補 j)/sqrt(len), W 和(prefix の有効 token)/sqrt(n)> / sqrt(d)
（temp は item ごとの確信度。小さい item は teacher uncertain）。
robust shard には variant（perm / irrelevant_ctx / ambiguous）と variant_of（元 val item_id）を余分キーとして付ける。
  - irrelevant_ctx: 潜在ベクトル 0 の token（id が ACTIVE の外側）を context に挿入 -> logits は変わらない
  - ambiguous     : logits を 0.2 倍（teacher が迷う）

CLI: python -m tb250distill.student.fake_data --out data/tok/fake [--n-train 4000 ...]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PAD, UNK, BOS, EOS, SEP = 0, 1, 2, 3, 4
FIRST = 5


class FakeTeacher:
    def __init__(self, vocab=8192, active=60, d=3, seed=0, scale=3.0):
        rng = np.random.default_rng(seed + 12345)
        self.vocab, self.active, self.d, self.scale = vocab, active, d, scale
        self.W = np.zeros((vocab, d))
        self.U = np.zeros((vocab, d))
        self.W[FIRST:FIRST + active] = rng.standard_normal((active, d))
        self.U[FIRST:FIRST + active] = rng.standard_normal((active, d))

    def logits(self, prefix_tokens, cands, temp):
        pt = np.asarray(prefix_tokens)
        pt = pt[(pt >= FIRST) & (pt < FIRST + self.active)]
        pv = self.W[pt].sum(0) / np.sqrt(max(1, len(pt)))
        out = []
        for c in cands:
            c = np.asarray(c)
            cv = self.U[c].sum(0) / np.sqrt(max(1, len(c)))
            out.append(self.scale * temp * float(cv @ pv) / np.sqrt(self.d))
        return np.array(out)


def _alloc(n, lp, kmax, lc):
    return dict(
        item_id=np.zeros(n, np.int64), prefix=np.zeros((n, lp), np.int32), prefix_len=np.zeros(n, np.int32),
        cand=np.zeros((n, kmax, lc), np.int32), cand_len=np.zeros((n, kmax), np.int32),
        k=np.zeros(n, np.int32), t_logits=np.zeros((n, kmax), np.float32), gold=np.full(n, -1, np.int32))


def _gen_item(rng, ft, lp, lc, kmax):
    k = int(rng.integers(2, kmax + 1))
    q = rng.integers(FIRST, FIRST + ft.active, int(rng.integers(3, 9)))
    ctx = rng.integers(FIRST, FIRST + ft.active, int(rng.integers(8, min(lp - 12, 40))))
    cands = [rng.integers(FIRST, FIRST + ft.active, int(rng.integers(1, lc // 2 + 1))) for _ in range(k)]
    temp = float(rng.choice([0.25, 0.6, 1.0, 1.0, 1.4]))
    return q, ctx, cands, temp


def _fill(arr, i, item_id, prefix, cands, logits, gold, lp, lc):
    prefix = np.asarray(prefix)[-lp:] if len(prefix) > lp else np.asarray(prefix)
    arr["item_id"][i] = item_id
    arr["prefix"][i, :len(prefix)] = prefix
    arr["prefix_len"][i] = len(prefix)
    k = len(cands)
    arr["k"][i] = k
    for j, c in enumerate(cands):
        c = np.asarray(c)[:lc]
        arr["cand"][i, j, :len(c)] = c
        arr["cand_len"][i, j] = len(c)
    arr["t_logits"][i, :k] = logits
    arr["gold"][i] = gold


def make_split(ft, n, rng, id0, lp, lc, kmax, gold_frac=0.7):
    arr = _alloc(n, lp, kmax, lc)
    items = []
    for i in range(n):
        q, ctx, cands, temp = _gen_item(rng, ft, lp, lc, kmax)
        prefix = np.concatenate([q, [SEP], ctx])
        if len(prefix) > lp:   # context の左（先頭側）を切る
            prefix = np.concatenate([q, [SEP], ctx[-(lp - len(q) - 1):]])
        logits = ft.logits(prefix, cands, temp)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        gold = int(rng.choice(len(cands), p=p)) if rng.random() < gold_frac else -1
        _fill(arr, i, id0 + i, prefix, cands, logits, gold, lp, lc)
        items.append((prefix, cands, logits, gold, temp))
    return arr, items


def make_robust(ft, val_arr, val_items, rng, id0, lp, lc, kmax, n_each):
    n_each = min(n_each, len(val_items))
    rows = []
    for variant in ("perm", "irrelevant_ctx", "ambiguous"):
        for i in range(n_each):
            prefix, cands, logits, gold, temp = val_items[i]
            of = int(val_arr["item_id"][i])
            if variant == "perm":
                pm = rng.permutation(len(cands))
                rows.append((variant, of, prefix, [cands[j] for j in pm], logits[pm],
                             int(np.where(pm == gold)[0][0]) if gold >= 0 else -1))
            elif variant == "irrelevant_ctx":
                junk = rng.integers(FIRST + ft.active, FIRST + ft.active + 200, 20)
                room = lp - len(prefix)
                junk = junk[:max(0, min(20, room))]
                pre2 = np.concatenate([prefix, junk]) if len(junk) else prefix
                rows.append((variant, of, pre2, cands, logits, gold))
            else:
                rows.append((variant, of, prefix, cands, logits * 0.2, gold))
    arr = _alloc(len(rows), lp, kmax, lc)
    for i, (v, of, prefix, cands, logits, gold) in enumerate(rows):
        _fill(arr, i, id0 + i, prefix, cands, logits, gold, lp, lc)
    arr["variant"] = np.array([r[0] for r in rows])
    arr["variant_of"] = np.array([r[1] for r in rows], dtype=np.int64)
    return arr


def write_fake_dataset(out_dir, n_train=6000, n_val=500, n_test=500, n_robust=100, vocab=8192, lp=128,
                       lc=16, kmax=5, seed=0, active=60):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    ft = FakeTeacher(vocab=vocab, active=active, seed=seed)
    ids = {"train": 0, "val": 100000, "test": 200000, "robust": 300000}
    tr, _ = make_split(ft, n_train, rng, ids["train"], lp, lc, kmax)
    va, va_items = make_split(ft, n_val, rng, ids["val"], lp, lc, kmax)
    te, _ = make_split(ft, n_test, rng, ids["test"], lp, lc, kmax)
    rb = make_robust(ft, va, va_items, rng, ids["robust"], lp, lc, kmax, n_robust)
    for name, a in (("train", tr), ("val", va), ("test", te), ("robust", rb)):
        np.savez(os.path.join(out_dir, f"{name}_L{lp}.npz"), **a)
    meta = dict(kind="fake", vocab=vocab, lp=lp, lc=lc, kmax=kmax, seed=seed, active=active,
                n=dict(train=n_train, val=n_val, test=n_test, robust=len(rb["item_id"])))
    with open(os.path.join(out_dir, "fake_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-robust", type=int, default=100, help="variant ごとの件数")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--lp", type=int, default=128)
    ap.add_argument("--lc", type=int, default=16)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--active", type=int, default=60, help="潜在構造を持つ token 数（小さいほど学習が容易）")
    a = ap.parse_args(argv)
    print(json.dumps(write_fake_dataset(a.out, a.n_train, a.n_val, a.n_test, a.n_robust, a.vocab, a.lp,
                                        a.lc, a.kmax, a.seed, a.active)))


if __name__ == "__main__":
    main()
