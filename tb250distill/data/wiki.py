"""Wikipedia（日本語）→ LM 事前学習用 tokenized shard（Student 語彙 = pubA の sentencepiece をそのまま使う）。

独立実験「Wikipedia 事前学習」用。意味表現蒸留（sem 系）とは無関係。

  # 1) Mac: 記事を id%mod==rem で間引いた subset parquet を作る（pyarrow が必要。出力は小さいので tb250 へ転送する）
  python -m tb250distill.data.wiki subset --src ~/dev/data/wikipedia_ja/20231101.ja --out <dir> --mod 8 --rem 0

  # 2) tb250（nice 推奨）: 整形 → pubA spm でトークン化 → 長さ T の連続トークン列 shard（uint16）
  python -m tb250distill.data.wiki build --src /mnt/hdd/wikipedia_ja_subset --out data/wiki/wikiA \
      --spm data/tok/pubA/spm.model --seq-len 128 --target-tokens 50000000

出力 `<out>/`:
  train_00000.npy ...   uint16 (rows, T)。1 行 = ストリーム上の連続 T トークン（行頭からモデルは h0=0 で読む）。
  val_00000.npy         held-out（記事単位。crc32(id) % 100 < val_pct）。
  meta.json             記事数・トークン数・byte-fallback 率・時間・spm sha1 など。
ストリーム = 記事ごとに [bos] + 段落トークン列 + [eos] を連結したもの（段落間に区切りは入れない）。train/val は別ストリーム。
行の学習への使い方: 入力 = row[:-1]、教師 = row[1:]（pretrain_lm.py）。

ライセンス: Wikipedia テキストは CC BY-SA 4.0（出典: https://huggingface.co/datasets/wikimedia/wikipedia 20231101.ja）。
subset parquet / shard は派生物。公開版モデルへの採用可否はユーザー判断（docs/WIKI_PRETRAIN.md）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

BOS, EOS = 2, 3
SOURCE_NOTE = {"dataset": "wikimedia/wikipedia", "config": "20231101.ja", "license": "CC BY-SA 4.0 (and GFDL)",
               "url": "https://huggingface.co/datasets/wikimedia/wikipedia"}

_EMPTY_PAREN = re.compile(r"[（(][\s、,，・;；:：]*[）)]")
_SPACES = re.compile(r"[ \t　]+")
_JA = re.compile(r"[぀-ヿ㐀-鿿]")


# ---------------------------------------------------------------------------
# 整形（純関数）
# ---------------------------------------------------------------------------

def clean_paragraphs(text: str, min_ja_frac: float = 0.3, short_len: int = 40) -> list[str]:
    """記事本文 → 段落（1 行 = 1 段落）のリスト。
    - 空行・見出し（句点が無く short_len 文字未満の行）・日本語文字が少ない行（参考文献/英語の表など）を捨てる。
    - 中身が空の括弧（読み・原綴りが消えた残骸「（）」「（、）」）を除き、空白を詰める。"""
    out = []
    for line in text.replace("\r", "\n").split("\n"):
        s = _EMPTY_PAREN.sub("", line)
        s = _SPACES.sub(" ", s).strip()
        if not s:
            continue
        if "。" not in s and len(s) < short_len:
            continue
        if len(_JA.findall(s)) < min_ja_frac * len(s):
            continue
        out.append(s)
    return out


def is_val_article(art_id: str, val_pct: int) -> bool:
    return (zlib.crc32(str(art_id).encode()) % 100) < val_pct


def order_key(art_id: str, seed: int = 20261006) -> int:
    """記事の処理順（ファイル順バイアスを避けるための決定的シャッフル）。"""
    return zlib.crc32(f"{seed}:{art_id}".encode())


def article_tokens(sp, paras: list[str], max_tokens: int) -> tuple[list[int], bool, int]:
    """段落を順に encode して連結し、max_tokens を超えたら（段落境界で）打ち切る。最初の段落が単独で超える場合だけ途中で切る。
    戻り値 (token ids, 打ち切りの有無, 使った段落数)。"""
    ids: list[int] = []
    enc = sp.encode(paras)
    for i, p in enumerate(enc):
        if ids and len(ids) + len(p) > max_tokens:
            return ids, True, i
        ids.extend(p)
        if len(ids) >= max_tokens:
            return ids[:max_tokens], (len(ids) > max_tokens or i + 1 < len(enc)), i + 1
    return ids, False, len(enc)


class RowWriter:
    """トークンストリームを長さ T の行へ詰めて uint16 の .npy shard に書く（shard あたり shard_rows 行）。端数は捨てる。"""

    def __init__(self, out_dir: Path, split: str, T: int, shard_rows: int = 131072):
        self.out_dir, self.split, self.T, self.shard_rows = out_dir, split, T, shard_rows
        self.buf: list[np.ndarray] = []
        self.buf_n = 0
        self.rows = 0
        self.tokens = 0          # 書き出した（行に詰めた）トークン数
        self.shards = 0
        self.stream_tokens = 0   # 受け取った全トークン数（端数を含む）

    def add(self, ids: list[int]):
        a = np.asarray(ids, dtype=np.uint16)
        self.stream_tokens += int(a.size)
        self.buf.append(a)
        self.buf_n += int(a.size)
        if self.buf_n >= self.shard_rows * self.T:
            self._flush(final=False)

    def _flush(self, final: bool):
        if not self.buf_n:
            return
        a = np.concatenate(self.buf)
        nrow = a.size // self.T
        if not final:
            nrow = min(nrow, self.shard_rows)
        if nrow > 0:
            rows = a[: nrow * self.T].reshape(nrow, self.T)
            np.save(self.out_dir / f"{self.split}_{self.shards:05d}.npy", rows)
            self.shards += 1
            self.rows += nrow
            self.tokens += nrow * self.T
        rest = a[nrow * self.T:]
        self.buf = [rest] if rest.size else []
        self.buf_n = int(rest.size)

    def close(self):
        # 残りを全部（shard_rows を超えていても）吐く
        while self.buf_n >= self.T:
            self._flush(final=False)
            if self.buf_n < self.T:
                break
        self._flush(final=True)


# ---------------------------------------------------------------------------
# subset（Mac）
# ---------------------------------------------------------------------------

def cmd_subset(args) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.src), "*.parquet")))
    if not files:
        print(f"parquet が無い: {args.src}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_in = n_out = 0
    chars_out = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        dst = out / ("subset-" + os.path.basename(fp).split("-")[1] + ".parquet")
        if dst.exists():
            print(f"skip（既存）: {dst}")
            continue
        w = None
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["id", "title", "text"]).to_pylist()
            keep = []
            for r in t:
                n_in += 1
                try:
                    ok = int(r["id"]) % args.mod == args.rem
                except ValueError:
                    ok = zlib.crc32(r["id"].encode()) % args.mod == args.rem
                if not ok:
                    continue
                text = r["text"]
                if args.max_chars and len(text) > args.max_chars:
                    cut = text.rfind("\n", 0, args.max_chars)
                    text = text[: cut if cut > 0 else args.max_chars]
                keep.append({"id": r["id"], "title": r["title"], "text": text})
                chars_out += len(text)
            if keep:
                tab = pa.Table.from_pylist(keep)
                if w is None:
                    w = pq.ParquetWriter(str(dst) + ".tmp", tab.schema, compression="zstd")
                w.write_table(tab)
                n_out += len(keep)
        if w is not None:
            w.close()
            os.replace(str(dst) + ".tmp", dst)
        print(f"{os.path.basename(fp)}: -> {dst.name} (累計 in {n_in} out {n_out})", flush=True)
    meta = {"source": SOURCE_NOTE, "mod": args.mod, "rem": args.rem, "max_chars": args.max_chars,
            "articles_in": n_in, "articles_out": n_out, "chars_out": chars_out, "seconds": round(time.time() - t0, 1),
            "note": "id % mod == rem の記事のみ。text は max_chars で段落境界（改行）切り。CC BY-SA 4.0 の派生物。"}
    (out / "SUBSET.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# build（tb250）
# ---------------------------------------------------------------------------

def iter_articles(src: str):
    import pyarrow.parquet as pq
    for fp in sorted(glob.glob(os.path.join(src, "*.parquet"))):
        pf = pq.ParquetFile(fp)
        for rg in range(pf.metadata.num_row_groups):
            tab = pf.read_row_group(rg, columns=["id", "text"])
            for i_, t_ in zip(tab.column("id").to_pylist(), tab.column("text").to_pylist()):
                yield i_, t_


def cmd_build(args) -> int:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    if sp.get_piece_size() != args.vocab:
        print(f"spm の語彙 {sp.get_piece_size()} != --vocab {args.vocab}", file=sys.stderr)
        return 2
    is_byte = np.array([sp.is_byte(i) for i in range(sp.get_piece_size())])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("*.npy")) and not args.force:
        print(f"{out} に shard が既にある（上書きしない。--force か別の --out）", file=sys.stderr)
        return 2
    t0 = time.time()
    arts = list(iter_articles(args.src))
    t_read = time.time() - t0
    arts.sort(key=lambda a: order_key(a[0], args.seed))
    print(f"read {len(arts)} articles in {t_read:.1f}s", flush=True)
    W = {"train": RowWriter(out, "train", args.seq_len, args.shard_rows),
         "val": RowWriter(out, "val", args.seq_len, args.shard_rows)}
    st = {"articles_read": len(arts), "dropped_empty": 0, "dropped_short": 0, "articles_used": {"train": 0, "val": 0},
          "articles_truncated": 0, "chars": {"train": 0, "val": 0},
          "byte_tokens": 0, "unk_tokens": 0, "content_tokens": 0, "paragraphs": 0}
    t1 = time.time()
    for k, (aid, text) in enumerate(arts):
        if W["train"].stream_tokens >= args.target_tokens and (args.val_tokens is None or
                                                                W["val"].stream_tokens >= args.val_tokens):
            break
        split = "val" if is_val_article(aid, args.val_pct) else "train"
        if split == "train" and W["train"].stream_tokens >= args.target_tokens:
            continue
        paras = clean_paragraphs(text)
        if not paras:
            st["dropped_empty"] += 1
            continue
        if sum(len(p) for p in paras) < args.min_chars:
            st["dropped_short"] += 1
            continue
        ids, trunc, n_used = article_tokens(sp, paras, args.max_article_tokens)
        st["articles_truncated"] += int(trunc)
        st["paragraphs"] += n_used
        a = np.asarray(ids, dtype=np.int64)
        st["content_tokens"] += int(a.size)
        st["byte_tokens"] += int(is_byte[a].sum())
        st["unk_tokens"] += int((a == 1).sum())
        st["chars"][split] += sum(len(p) for p in paras[:n_used])
        W[split].add([BOS] + ids + [EOS])
        st["articles_used"][split] += 1
        if (k + 1) % 20000 == 0:
            print(f"  {k + 1}/{len(arts)} train tokens {W['train'].stream_tokens} val {W['val'].stream_tokens} "
                  f"{time.time() - t1:.0f}s", flush=True)
    for w in W.values():
        w.close()
    tok_total = W["train"].stream_tokens + W["val"].stream_tokens
    meta = {"name": out.name, "source": SOURCE_NOTE, "src_dir": args.src, "seq_len": args.seq_len, "vocab": args.vocab,
            "spm": args.spm, "spm_sha1": hashlib.sha1(open(args.spm, "rb").read()).hexdigest(),
            "special_ids": {"bos": BOS, "eos": EOS}, "dtype": "uint16",
            "stream": "per article [bos] + paragraph tokens + [eos]; train/val separate streams; rows = consecutive T tokens, remainder dropped",
            "max_article_tokens": args.max_article_tokens, "min_chars": args.min_chars, "val_pct": args.val_pct,
            "target_tokens": args.target_tokens, "seed": args.seed,
            "train": {"rows": W["train"].rows, "tokens": W["train"].tokens, "shards": W["train"].shards,
                      "stream_tokens": W["train"].stream_tokens},
            "val": {"rows": W["val"].rows, "tokens": W["val"].tokens, "shards": W["val"].shards,
                    "stream_tokens": W["val"].stream_tokens},
            "stats": st, "byte_token_frac": st["byte_tokens"] / max(1, st["content_tokens"]),
            "unk_token_frac": st["unk_tokens"] / max(1, st["content_tokens"]),
            "tokens_per_char": st["content_tokens"] / max(1, sum(st["chars"].values())),
            "seconds": {"read_parquet": round(t_read, 1), "tokenize_pack": round(time.time() - t1, 1),
                        "total": round(time.time() - t0, 1)},
            "note": "CC BY-SA 4.0 の派生物。公開版モデルへの採用可否は未判断（docs/WIKI_PRETRAIN.md）。"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# 読み込み（pretrain_lm.py から使う）
# ---------------------------------------------------------------------------

def load_rows(data_dir: str, split: str, mmap: bool = True) -> list[np.ndarray]:
    """`<data_dir>/<split>_*.npy` を shard 順の memmap のリストで返す。"""
    fs = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.npy")))
    return [np.load(f, mmap_mode="r" if mmap else None) for f in fs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("subset", help="parquet を id%%mod==rem で間引いた subset を作る（Mac。pyarrow 必要）")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mod", type=int, default=8)
    s.add_argument("--rem", type=int, default=0)
    s.add_argument("--max-chars", type=int, default=8000, help="記事 text の最大文字数（段落境界で切る。0=切らない）")
    b = sub.add_parser("build", help="subset → 整形・トークン化・行詰め（tb250）")
    b.add_argument("--src", required=True, help="subset parquet のディレクトリ")
    b.add_argument("--out", required=True)
    b.add_argument("--spm", required=True, help="pubA の spm.model（Student の語彙と同一であること）")
    b.add_argument("--vocab", type=int, default=8192)
    b.add_argument("--seq-len", type=int, default=128)
    b.add_argument("--target-tokens", type=int, default=50_000_000, help="train ストリームの目標トークン数")
    b.add_argument("--val-tokens", type=int, default=None, help="val ストリームの上限（既定: train が目標に達した時点で止める）")
    b.add_argument("--val-pct", type=int, default=1, help="記事単位の held-out 割合（%%）")
    b.add_argument("--max-article-tokens", type=int, default=1024)
    b.add_argument("--min-chars", type=int, default=200)
    b.add_argument("--shard-rows", type=int, default=131072)
    b.add_argument("--seed", type=int, default=20261006)
    b.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    return {"subset": cmd_subset, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
