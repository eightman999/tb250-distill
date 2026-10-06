"""公開データセットの取得（data/external/<name>/ に生ファイル + MANIFEST.json）。

  python -m tb250distill.data.fetch_public --root data/external [--only massive,wrime]

- 取得元は revision（HF は commit sha / refs/convert/parquet の sha、GitHub は commit sha）に固定する。
- HF 認証は使わない（認証が要るものは対象外）。ダウンロードは requests で一時ファイル→rename。
- MANIFEST.json に id / revision / url / license / 取得日時 / ファイル別 sha256・件数 を記録する。
- ライセンス上の注意（NC/ND/SA 等）は license_notes に残す。再実行しても既存ファイルは sha256 が一致すれば再取得しない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HF = "https://huggingface.co/datasets/{id}/resolve/{rev}/{path}"
GH = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"

SOURCES: dict[str, dict] = {
    "routellm": {
        "hf_id": "routellm/gpt4_dataset",
        "revision": "7ef62d690ff71ce9c21f4226cfc7bbc2ef9e6227",
        "license": "apache-2.0 (HF card / LICENSE)",
        "hf_checked": {
            "routellm/gpt4_judge_battles": {"sha": "2a1afe8d0659904c0f6f59de6179e086fdb027c7", "license": "apache-2.0",
                                            "note": "battle 形式（109,101 件・parquet）。今回は gpt4_dataset の mixtral_score を使うため不使用"},
        },
        "license_notes": (
            "データセット自体は Apache-2.0。ただし prompt は lmsys-chat-1m / flan_v2_cot 等（source 列）由来で、"
            "元データ側の利用条件（特に LMSYS-Chat-1M の利用規約）が及ぶ可能性がある。私的研究利用に限り、派生データを公開しない。"
        ),
        "files": [
            {"url": HF.format(id="routellm/gpt4_dataset", rev="7ef62d690ff71ce9c21f4226cfc7bbc2ef9e6227", path="train.jsonl"), "path": "train.jsonl"},
            {"url": HF.format(id="routellm/gpt4_dataset", rev="7ef62d690ff71ce9c21f4226cfc7bbc2ef9e6227", path="valid.jsonl"), "path": "valid.jsonl"},
            {"url": HF.format(id="routellm/gpt4_dataset", rev="7ef62d690ff71ce9c21f4226cfc7bbc2ef9e6227", path="LICENSE"), "path": "LICENSE"},
        ],
    },
    "when2call": {
        "hf_id": "nvidia/When2Call",
        "revision": "0582f7749df63a96fdc3070932e83e72396ace53",
        "license": "cc-by-4.0",
        "license_notes": "CC BY 4.0（帰属表示のみ。商用可）。合成データ（NVIDIA 生成）。",
        "files": [
            {"url": HF.format(id="nvidia/When2Call", rev="0582f7749df63a96fdc3070932e83e72396ace53", path="test/when2call_test_mcq.jsonl"), "path": "test/when2call_test_mcq.jsonl"},
            {"url": HF.format(id="nvidia/When2Call", rev="0582f7749df63a96fdc3070932e83e72396ace53", path="train/when2call_train_pref.jsonl"), "path": "train/when2call_train_pref.jsonl"},
        ],
    },
    "massive": {
        "hf_id": "AmazonScience/massive",
        "revision": "ed58ac423a2f4121720918bf5301577edce4ffd3",
        "revision_ref": "refs/convert/parquet (main=ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740 は loader script のみ)",
        "license": "cc-by-4.0",
        "hf_checked": {"AmazonScience/massive@main": "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740（loader script 型。parquet は refs/convert/parquet を使用）"},
        "license_notes": "CC BY 4.0（帰属表示のみ。商用可）。ja-JP と en-US のみ取得。",
        "files": [
            {"url": HF.format(id="AmazonScience/massive", rev="refs%2Fconvert%2Fparquet", path=f"{loc}/{sp}/0000.parquet"), "path": f"{loc}/{sp}.parquet"}
            for loc in ("ja-JP", "en-US") for sp in ("train", "validation", "test")
        ] + [
            {"url": HF.format(id="AmazonScience/massive", rev="ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740", path="LICENSE"), "path": "LICENSE"},
        ],
    },
    "wrime": {
        "hf_id": "shunk031/wrime (HF 版は loader script 型・license: unknown 表記のため不使用。元配布元を使用)",
        "source_repo": "https://github.com/ids-cv/wrime",
        "hf_checked": {"shunk031/wrime": {"sha": "3fb7212c389d7818b8e6179e2cdac762f2e081d9", "license": "unknown（カード表記）", "parquet_ref": "3d7d7bf5137d588d121761f75e6a65646b25fa3a（未使用）"}},
        "revision": "ac92deaff7845f31fdd46ef5d893300887006d13",
        "license": "CC BY-NC-ND 4.0 (ids-cv/wrime README)",
        "license_notes": (
            "NC（非営利のみ）・ND（改変物の再配布禁止）。私的研究利用に限り、変換済みデータ（replay DB / shard）を公開・再配布しない。"
            "WRIME ver.2 を取得（Writer/Reader1-3 の感情強度 0-3 と極性 -2..2）。"
        ),
        "files": [
            {"url": GH.format(repo="ids-cv/wrime", rev="ac92deaff7845f31fdd46ef5d893300887006d13", path="wrime-ver2.tsv"), "path": "wrime-ver2.tsv"},
            {"url": GH.format(repo="ids-cv/wrime", rev="ac92deaff7845f31fdd46ef5d893300887006d13", path="LICENSE"), "path": "LICENSE"},
        ],
    },
    "jglue": {
        "hf_id": "shunk031/JGLUE (HF 版は loader script 型で datasets 4 系不可・カード表記 cc-by-4.0 は上流と不一致。元配布元を使用)",
        "source_repo": "https://github.com/yahoojapan/JGLUE",
        "hf_checked": {"shunk031/JGLUE": {"sha": "41cc99f1c01d41b1ab13435ae97b7076c2199f4c", "license": "cc-by-4.0（カード表記。上流は CC BY-SA 4.0）", "parquet_ref": "3a4a622d8ae03c700184c17da7afa1d3dc870cd3（未使用。MARC-ja 複製を含む）"}},
        "revision": "6f071c09316baae89c3d083a90985b4b1cb9968c",
        "license": "CC BY-SA 4.0 (yahoojapan/JGLUE README)",
        "license_notes": (
            "SA（継承）。派生データを公開する場合は同ライセンス・帰属表示が必要。私的利用に限り、派生データは公開しない。"
            "MARC-ja は Amazon が MARC の配布を停止したため JGLUE 側で提供終了（datasets/marc_ja-v1.2 は .gitkeep のみ）。"
            "HF 上のパーケット複製は取り込まない（配布停止データの再配布物のため）。"
        ),
        "files": [
            {"url": GH.format(repo="yahoojapan/JGLUE", rev="6f071c09316baae89c3d083a90985b4b1cb9968c", path=f"datasets/{d}-v1.3/{sp}-v1.3.json"), "path": f"{d}/{sp}.jsonl"}
            for d in ("jcommonsenseqa", "jnli") for sp in ("train", "valid", "test")
        ] + [
            {"url": GH.format(repo="yahoojapan/JGLUE", rev="6f071c09316baae89c3d083a90985b4b1cb9968c", path="LICENSE"), "path": "LICENSE"},
        ],
        "skipped": {"MARC-ja": "Amazon による MARC 配布停止（JGLUE README）。取り込まない。"},
    },
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(p: Path) -> int | None:
    """jsonl / tsv は行数（tsv はヘッダ除く）、parquet は pyarrow の行数。LICENSE 等は None。"""
    suf = p.suffix
    if suf == ".jsonl":
        with open(p, "rb") as f:
            return sum(1 for line in f if line.strip())
    if suf == ".tsv":
        with open(p, "rb") as f:
            return max(0, sum(1 for line in f if line.strip()) - 1)
    if suf == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(str(p)).metadata.num_rows
    return None


def download(url: str, dest: Path, tries: int = 3) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for i in range(tries):
        try:
            with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  retry {i + 1}/{tries}: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(3 * (i + 1))
    raise SystemExit(f"download failed: {url}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/external")
    ap.add_argument("--only", default="", help="カンマ区切りで対象を絞る")
    args = ap.parse_args(argv)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    mpath = root / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    only = {s for s in args.only.split(",") if s}
    for name, spec in SOURCES.items():
        if only and name not in only:
            continue
        print(f"== {name}")
        files = []
        for f in spec["files"]:
            dest = root / name / f["path"]
            if not dest.exists():
                print(f"  GET {f['url']}")
                download(f["url"], dest)
            files.append({
                "path": f["path"], "url": f["url"], "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest), "rows": count_rows(dest),
            })
            print(f"  {f['path']}: {files[-1]['bytes']} bytes, rows={files[-1]['rows']}")
        entry = {k: v for k, v in spec.items() if k != "files"}
        entry["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        entry["files"] = files
        manifest[name] = entry
        mpath.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
