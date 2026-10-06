"""sentencepiece（unigram）学習と tokenized shard の書き出し（DESIGN.md「Tokenized shard 契約」）。

  python -m tb250distill.tokenize_data --db data/replay.sqlite --name spm8k --vocab 8192 --lp 128 --lc 16

出力 data/tok/<name>/:
  spm.model / spm.vocab         （train split の items テキストから学習。特殊 id: 0=pad 1=unk 2=bos 3=eos 4=<sep>）
  <split>_L<Lp>.npz             item_id, prefix[N,Lp], prefix_len, cand[N,Kmax,Lc], cand_len[N,Kmax], k, t_logits[N,Kmax], gold,
                                source[N]（optional・文字列。source 別評価用。無い旧 shard も有効）
  meta.json                     語彙サイズ・件数・切り詰め率・source 別件数など
--max-per-source "synth=30000"  で train split だけ source ごとに上限をかける（seed 固定のランダム抽出。指定の無い source は全件。
  val/test/robust には適用しない）。抽出は item_id ごとの hash 順位で決まるので、行の並びや teacher の進み具合に依らず決定的。
teacher 行がある item だけを書き出す（teacher が増えたら再実行すれば再作成される。spm は --retrain しない限り再利用）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from tb250distill import replay

PAD, UNK, BOS, EOS, SEP = 0, 1, 2, 3, 4
SPLITS = ("train", "val", "test", "robust")


def _has_source(conn) -> bool:
    return any(r[1] == "source" for r in conn.execute("PRAGMA table_info(items)"))


def corpus_rows(conn, split: str = "train", max_per_source: int = 0, seed: int = 20261006, sources=None) -> list:
    """spm 学習コーパスの元になる items 行。max_per_source>0 なら source ごとに決定的に間引く
    （合成 100k が公開データの語彙を押し流さないようにするため。既定 0=全件で従来どおり）。"""
    sel = "context, question, candidates" + (", source" if _has_source(conn) else ", 'synth' AS source")
    rows = conn.execute(f"SELECT {sel} FROM items WHERE split=? ORDER BY item_id", (split,)).fetchall()
    if sources:
        rows = [r for r in rows if r["source"] in sources or ("public" in sources and r["source"] != "synth")]
    if max_per_source and max_per_source > 0:
        by: dict = {}
        for r in rows:
            by.setdefault(r["source"], []).append(r)
        rows = []
        for src in sorted(by):
            rs = by[src]
            if len(rs) > max_per_source:
                rs = random.Random(f"{seed}:{src}").sample(rs, max_per_source)
            rows += rs
    return rows


def corpus_lines(conn, split: str = "train", max_per_source: int = 0, sources=None):
    for r in corpus_rows(conn, split, max_per_source, sources=sources):
        yield r["question"]
        for sent in re.split(r"(?<=[。.!?！？\n])\s*", r["context"]):
            if sent.strip():
                yield sent.strip()
        for c in json.loads(r["candidates"]):
            yield c


def train_spm(conn, out_dir: Path, vocab: int, threads: int = 2, max_piece_len: int = 6, max_per_source: int = 0, sources=None) -> Path:
    import sentencepiece as spm

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "spm"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        n = 0
        for line in corpus_lines(conn, max_per_source=max_per_source, sources=sources):
            f.write(line.replace("\n", " ") + "\n")
            n += 1
        corpus = f.name
    if n == 0:
        raise SystemExit("no train items to learn the tokenizer from")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=str(prefix), vocab_size=vocab, model_type="unigram", byte_fallback=True,
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS, user_defined_symbols=["<sep>"],
        character_coverage=0.9995, split_digits=True, hard_vocab_limit=False, num_threads=threads,
        max_sentencepiece_length=max_piece_len,
        input_sentence_size=2_000_000, shuffle_input_sentence=True, minloglevel=1,
    )
    Path(corpus).unlink(missing_ok=True)
    return Path(str(prefix) + ".model")


def encode_prefix(sp, question: str, context: str, lp: int) -> list[int]:
    """question <sep> context。context は末尾側を残し、左を切る。"""
    q = sp.encode(question)
    c = sp.encode(context)
    if len(q) + 1 >= lp:  # 質問だけで一杯なら質問を右で切る（context は落ちる）
        return q[: lp - 1] + [SEP]
    room = lp - len(q) - 1
    return q + [SEP] + (c[-room:] if room > 0 else [])


def parse_max_per_source(spec) -> dict:
    """"synth=30000,jnli=5000"（文字列、または文字列のリスト）-> {source: 上限}。上限は 1 以上の整数。"""
    if not spec:
        return {}
    if isinstance(spec, str):
        spec = [spec]
    out: dict = {}
    for chunk in spec:
        for part in str(chunk).split(","):
            part = part.strip()
            if not part:
                continue
            name, eq, val = part.partition("=")
            name = name.strip()
            try:
                n = int(val)
            except ValueError:
                n = 0
            if not eq or not name or n < 1:
                raise ValueError(f"--max-per-source: '{part}' は 'source=上限(1以上の整数)' 形式で書く")
            if name in out:
                raise ValueError(f"--max-per-source: source '{name}' が重複")
            out[name] = n
    return out


def _rank_key(seed, src: str, item_id) -> bytes:
    return hashlib.sha1(f"{seed}:{src}:{int(item_id)}".encode()).digest()


def cap_per_source(rows: list, caps: dict, seed: int = 20261006) -> list:
    """rows（dict、source と item_id を持つ）を source ごとに caps[source] 件へ決定的に間引く。caps に無い source は全件。
    残す item は (seed, source, item_id) の hash 順位が小さい順に cap 件。元の並び順は保つ。"""
    if not caps:
        return rows
    keep: set = set()
    by: dict = {}
    for r in rows:
        src = r.get("source") or "synth"
        by.setdefault(src, []).append(r)
    for src, rs in by.items():
        cap = caps.get(src)
        if cap is not None and len(rs) > cap:
            ranked = sorted(rs, key=lambda r: _rank_key(seed, src, r["item_id"]))[:cap]
            keep.update(r["item_id"] for r in ranked)
        else:
            keep.update(r["item_id"] for r in rs)
    return [r for r in rows if r["item_id"] in keep]


def build_shard(conn, sp, split: str, lp: int, lc: int, kmax: int, teacher_temp: float, sources=None,
                max_per_source=None, sample_seed: int = 20261006) -> tuple[dict, dict]:
    rows = list(replay.iter_scored(conn, split, sources=sources or None))
    rows = [r for r in rows if len(r["candidates"]) <= kmax]
    before_cap = None
    if max_per_source and split == "train":  # val/test/robust には適用しない
        before_cap = {}
        for r in rows:
            sname = r.get("source") or "synth"
            before_cap[sname] = before_cap.get(sname, 0) + 1
        rows = cap_per_source(rows, max_per_source, sample_seed)
    n = len(rows)
    item_id = np.zeros(n, np.int64)
    prefix = np.zeros((n, lp), np.int32)
    prefix_len = np.zeros(n, np.int32)
    cand = np.zeros((n, kmax, lc), np.int32)
    cand_len = np.zeros((n, kmax), np.int32)
    k = np.zeros(n, np.int32)
    t_logits = np.zeros((n, kmax), np.float32)
    gold = np.full(n, -1, np.int32)
    ctx_cut = cand_cut = 0
    source = np.array([r.get("source") or "synth" for r in rows], dtype="U16") if n else np.zeros(0, dtype="U16")
    src_stat: dict = {}
    is_byte = np.array([sp.is_byte(j) for j in range(sp.get_piece_size())], dtype=bool)
    for i, r in enumerate(rows):
        item_id[i] = r["item_id"]
        item_cand_cut = 0
        q_n = len(sp.encode(r["question"]))
        c_n = len(sp.encode(r["context"]))
        ids = encode_prefix(sp, r["question"], r["context"], lp)
        pre_cut = q_n + 1 + c_n > lp
        if pre_cut:
            ctx_cut += 1
        prefix[i, : len(ids)] = ids
        prefix_len[i] = len(ids)
        kk = len(r["candidates"])
        k[i] = kk
        for j, text in enumerate(r["candidates"]):
            cid = sp.encode(text) or [UNK]
            if len(cid) > lc:
                cand_cut += 1
                item_cand_cut += 1
            cid = cid[:lc]
            cand[i, j, : len(cid)] = cid
            cand_len[i, j] = len(cid)
        t_logits[i, :kk] = np.asarray(r["logits"], np.float32) / teacher_temp
        gold[i] = -1 if r["gold"] is None else r["gold"]
        st = src_stat.setdefault(source[i].item(), {"n": 0, "prefix_cut": 0, "items_cand_cut": 0, "prefix_len_sum": 0, "tok": 0, "byte": 0})
        used = np.concatenate([prefix[i, : len(ids)]] + [cand[i, j, : cand_len[i, j]] for j in range(kk)])
        st["tok"] += int(used.size)
        st["byte"] += int(is_byte[used].sum())
        st["n"] += 1
        st["prefix_cut"] += int(pre_cut)
        st["items_cand_cut"] += int(item_cand_cut > 0)
        st["prefix_len_sum"] += len(ids)
    arrays = dict(item_id=item_id, prefix=prefix, prefix_len=prefix_len, cand=cand, cand_len=cand_len, k=k, t_logits=t_logits, gold=gold,
                  source=source)
    counts: dict = {}
    for nm in source.tolist():
        counts[nm] = counts.get(nm, 0) + 1
    stats = {
        "source_counts": dict(sorted(counts.items())),
        "n": n, "prefix_truncated_frac": round(ctx_cut / n, 4) if n else None, "cand_truncated": cand_cut,
        "mean_prefix_len": round(float(prefix_len.mean()), 1) if n else None,
        "k_hist": {int(a): int(b) for a, b in zip(*np.unique(k, return_counts=True))} if n else {},
        "gold_frac": round(float((gold >= 0).mean()), 4) if n else None,
        "by_source": {
            s_: {"n": v["n"], "prefix_truncated_frac": round(v["prefix_cut"] / v["n"], 4),
                 "items_with_truncated_cand_frac": round(v["items_cand_cut"] / v["n"], 4),
                 "mean_prefix_len": round(v["prefix_len_sum"] / v["n"], 1),
                 "byte_fallback_token_frac": round(v["byte"] / max(1, v["tok"]), 4)}
            for s_, v in sorted(src_stat.items())},
    }
    if before_cap is not None:
        stats["max_per_source"] = dict(max_per_source)
        stats["source_counts_before_cap"] = dict(sorted(before_cap.items()))
        stats["sample_seed"] = sample_seed
    return arrays, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/replay.sqlite")
    ap.add_argument("--name", default="spm8k")
    ap.add_argument("--out-root", default="data/tok")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--lp", type=int, default=128, help="prefix（question <sep> context）の最大 token 数")
    ap.add_argument("--lc", type=int, default=16, help="候補 1 つの最大 token 数")
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--max-piece-len", type=int, default=6,
                    help="spm の最大 piece 長（文字）。候補文言が 1 piece に丸ごと入る＝ラベル化するのを避ける")
    ap.add_argument("--sources", default="",
                    help="include フィルタ（カンマ区切り。既定=全 source。'public' は synth 以外）。shard と spm コーパスの両方に効く")
    ap.add_argument("--retrain", action="store_true", help="spm を学習し直す")
    ap.add_argument("--corpus-max-per-source", type=int, default=0,
                    help="spm 学習コーパスの source ごとの train item 上限（0=全件）。合成が圧倒的に多いとき公開データの語彙を確保するため")
    ap.add_argument("--max-per-source", action="append", default=None, metavar="SRC=N",
                    help='train split だけ source ごとの件数上限。例 "synth=30000"（複数は "synth=30000,jnli=5000"）。'
                         "seed 固定のランダム抽出。指定の無い source は全件。val/test/robust には適用しない")
    ap.add_argument("--sample-seed", type=int, default=20261006, help="--max-per-source の抽出 seed")
    ap.add_argument("--teacher-temp", type=float, default=1.0, help="t_logits = logits / この値（既定 1.0=無加工）")
    args = ap.parse_args(argv)

    import sentencepiece as spm

    sources = [x for x in args.sources.split(",") if x] or None
    try:
        max_per_source = parse_max_per_source(args.max_per_source)
    except ValueError as e:
        raise SystemExit(str(e))
    out_dir = Path(args.out_root) / args.name
    conn = replay.connect(args.db, readonly=True)
    model = out_dir / "spm.model"
    if args.retrain or not model.exists():
        t0 = time.time()
        train_spm(conn, out_dir, args.vocab, max_piece_len=args.max_piece_len, max_per_source=args.corpus_max_per_source, sources=sources)
        print(f"trained spm in {time.time() - t0:.1f}s -> {model}")
    sp = spm.SentencePieceProcessor(model_file=str(model))
    assert sp.piece_to_id("<sep>") == SEP and sp.pad_id() == PAD and sp.unk_id() == UNK and sp.bos_id() == BOS and sp.eos_id() == EOS, "special id contract broken"
    if sp.get_piece_size() != args.vocab:
        print(f"warning: actual vocab {sp.get_piece_size()} != requested {args.vocab}", file=sys.stderr)

    meta = {
        "name": args.name, "vocab": sp.get_piece_size(), "lp": args.lp, "lc": args.lc, "kmax": args.kmax,
        "teacher_temp": args.teacher_temp, "spm_sha1": hashlib.sha1(model.read_bytes()).hexdigest(),
        "special_ids": {"pad": PAD, "unk": UNK, "bos": BOS, "eos": EOS, "sep": SEP}, "splits": {},
        "corpus_max_per_source": args.corpus_max_per_source, "sources": sources,
        "max_per_source": max_per_source, "sample_seed": args.sample_seed if max_per_source else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for split in [s for s in args.splits.split(",") if s]:
        arrays, stats = build_shard(conn, sp, split, args.lp, args.lc, args.kmax, args.teacher_temp, sources,
                                     max_per_source=max_per_source, sample_seed=args.sample_seed)
        if stats["n"] == 0:
            print(f"{split}: no scored items, skipped")
            continue
        path = out_dir / f"{split}_L{args.lp}.npz"
        np.savez(path, **arrays)
        meta["splits"][split] = stats
        print(f"{split}: {stats['n']} items -> {path} (prefix cut {stats['prefix_truncated_frac']}, mean len {stats['mean_prefix_len']}) by source {stats['source_counts']}")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    (out_dir / f"meta_L{args.lp}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))  # Lp ごとの統計（meta.json は最後の実行のもの）
    return 0


if __name__ == "__main__":
    sys.exit(main())
