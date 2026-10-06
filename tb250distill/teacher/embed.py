"""Teacher 側の文 embedding 取得（Candidate Semantic Distillation 用 target）。

  # llama-server（embedding 用、別ポート 18085）の起動・停止（RX 6400 = Vulkan0。teacher.produce 動作中は拒否）
  python -m tb250distill.teacher.embed serve start --provider qwen3_emb_0p6b [--cpu]
  python -m tb250distill.teacher.embed serve stop|status --provider qwen3_emb_0p6b
  # train shard の候補文字列だけを embedding -> data/emb/<provider>/<shard_name>/
  python -m tb250distill.teacher.embed extract --provider qwen3_emb_0p6b --shard data/tok/pubA --db data/replay.sqlite \
      [--lp 256] [--dim 128] [--batch 32] [--url http://127.0.0.1:18085]

出力 data/emb/<provider>/<shard_name>/:
  strings.json        train split の候補文字列の unique 集合（train 行 × 候補位置の初出順。index = 行番号）
  emb_raw.npy         float32 [U, D]  provider の出力そのまま（pooled。サーバが L2 正規化して返す場合はその値。meta に記録）
  pca.npz             mean[D] components[d,D] explained_variance[d] explained_variance_ratio[d] total_variance l2_normalize
  emb_pca.npy         float32 [U, d]  = (L2 正規化した emb_raw - mean) @ components.T（学習側が読む導出物。pca.npz と emb_raw から再計算可）
  cand_idx_train.npy  int32 [N, Kmax] shard の train 行 × 候補位置 -> unique index（候補無しの位置は -1）
  meta.json           provider, model, sha256, pooling, D, d, U, 説明分散比, 取得時間, item_id_sha1 など

PCA の扱い（学習側と共通の契約）:
  1. emb_raw の各行を L2 正規化する（pooled 出力の絶対スケールの違いを除く）
  2. train の unique 集合の平均を引く（LLM embedding は平均方向が支配的で、そのままだと cosine が全て ~1 になるため）
  3. 平均を引いた集合の共分散の上位 d 主成分へ射影（whiten しない。符号は最大絶対値成分が正になるよう固定）
  PCA 後は再正規化しない（Student 側の cosine loss は target のスケールに不変）。PCA は train unique 集合だけで学習する。

リーク防止（実装で強制）:
  - 入力は train shard（`train_L*.npz`）の item_id だけ。DB から引いた行の split が全て 'train' でなければ LeakError。
  - val/test/robust の shard・候補文字列は読まない・embedding しない・PCA に使わない（split を選ぶ引数も無い）。
--exclude-truncated（意味表現蒸留 loss だけから、候補が Lc token を超えて切り詰められる文字列を除く）:
  既存の data/emb/<provider>/<shard>/ は上書きしない。別名 <shard>_notrunc/ に cand_idx_train.npy（spm token 長 > Lc の候補文字列の位置を -1）と
  meta.json（exclude_truncated: Lc・source 別の除外件数）を作る。strings.json / emb_raw.npy / pca.npz / emb_pca.npy は元をハードリンク
  （再抽出しない）。判断用の KD/CE 損失は shard の候補全体を使い続けるので影響しない（sem target の有効位置だけが減る）。
  元の <shard>/ が無ければ先に通常の extract を行う。
再開: 取得は --chunk 件ずつ <out>/parts/chunk_*.npy に保存し、再実行で取得済み chunk を飛ばす（strings.json の sha1 が違えば拒否）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HOME = Path.home()
LLAMA_DIR = HOME / "bench" / "llama-bin" / "llama-b11384"
ICD = "/usr/share/vulkan/icd.d/radeon_icd.json"
REPO = Path(__file__).resolve().parents[2]
RUN_DIR = Path(os.environ.get("TB250_RUNS", REPO / "runs")) / "embed"
TEACHER_PORT = 18080  # 既存 teacher の llama-server。embedding 用はこれと別ポート


class LeakError(RuntimeError):
    """train 以外の split の文字列を embedding / PCA に使おうとした。"""


# --------------------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    """名前付きプリセット。llama-server の起動情報とクライアント設定を持つ。"""
    name: str
    model_path: str                 # gguf（~ 展開）。サーバを自分で起動しない場合は参照だけ
    model_id: str                   # 出典（HF repo / ファイル名）
    license: str
    pooling: str                    # llama-server --pooling（mean|last|cls）
    prefix: str = ""                # 各文字列の前に付ける（instruction 等）
    suffix: str = ""                # 後ろに付ける（EOS を明示的に足したい場合）
    port: int = 18085
    ctx: int = 2048   # 全 slot 合計。parallel=4 で 1 slot 512 token（候補文字列は高々 ~100 token）
    parallel: int = 4
    ubatch: int = 1024
    note: str = ""


PRESETS = {
    # 既存 Teacher と同じ chat モデルの最終 hidden を mean pooling（Teacher の内部表現そのものに最も近い）
    "qwen3_1p7b": ProviderSpec(
        name="qwen3_1p7b", model_path="~/bench/models/Qwen3-1.7B-Q4_K_M.gguf", model_id="Qwen3-1.7B-Q4_K_M.gguf (Teacher と同一)",
        license="Apache-2.0", pooling="mean",
        note="decoder LLM の全 token hidden の平均。sentence embedding 専用学習はされていない"),
    # sentence embedding 専用モデル（比較用）
    "qwen3_emb_0p6b": ProviderSpec(
        name="qwen3_emb_0p6b", model_path="~/tb250-distill/models/Qwen3-Embedding-0.6B-Q8_0.gguf",
        model_id="Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf", license="Apache-2.0", pooling="last",
        note="モデルカード指定: last token pooling。文書側（候補文字列）には instruction を付けない。次元は最大 1024"),
}


class EmbeddingProvider:
    """embedding 取得の抽象。embed(texts) -> float32 [n, D]（同じ入力に同じ出力）。"""
    name = "base"

    def embed(self, texts):  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"provider": self.name}


class LlamaServerProvider(EmbeddingProvider):
    """llama-server の `--embeddings` サーバへ HTTP で問い合わせる。

    endpoint="v1": POST /v1/embeddings（OpenAI 互換。サーバ側で L2 正規化される）
    endpoint="native": POST /embedding（embd_normalize=-1 で未正規化の pooled 値）
    """
    name = "llama_server"

    def __init__(self, url="http://127.0.0.1:18085", model="", pooling="mean", prefix="", suffix="", batch=32,
                 endpoint="native", timeout=120.0, retries=3, preset=None, model_path=None):
        self.url = url.rstrip("/")
        self.model = model
        self.pooling = pooling
        self.prefix, self.suffix = prefix, suffix
        self.batch = int(batch)
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries
        self.preset = preset
        self.model_path = model_path
        self._sess = None
        self.norms = []          # 観測した出力 norm（サーバが正規化して返したかの記録用）

    @classmethod
    def from_preset(cls, name, **over):
        sp = PRESETS[name]
        kw = dict(url=f"http://127.0.0.1:{sp.port}", model=os.path.basename(sp.model_path), pooling=sp.pooling,
                  prefix=sp.prefix, suffix=sp.suffix, preset=name, model_path=os.path.expanduser(sp.model_path))
        kw.update({k: v for k, v in over.items() if v is not None})
        return cls(**kw)

    def _http(self):
        if self._sess is None:
            import requests  # 遅延 import（Mac の .venv には無い。provider を使わないテストを通すため）
            self._sess = requests.Session()
        return self._sess

    def _post(self, texts):
        inputs = [self.prefix + t + self.suffix for t in texts]
        if self.endpoint == "v1":
            path, payload = "/v1/embeddings", {"input": inputs, "model": self.model}
        else:
            path, payload = "/embedding", {"content": inputs, "embd_normalize": -1}
        last = None
        for a in range(self.retries):
            try:
                r = self._http().post(self.url + path, json=payload, timeout=self.timeout)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                return r.json()
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (a + 1))
        raise RuntimeError(f"embedding request failed after {self.retries} tries: {last}")

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            chunk = list(texts[i:i + self.batch])
            res = self._post(chunk)
            if self.endpoint == "v1":
                rows = sorted(res["data"], key=lambda d: d["index"])
                vecs = [np.asarray(d["embedding"], np.float32) for d in rows]
            else:
                rows = sorted(res, key=lambda d: d["index"])
                vecs = []
                for d in rows:
                    e = np.asarray(d["embedding"], np.float32)
                    if e.ndim == 2:   # pooling none なら token ごと。pooled は [1, D]
                        if e.shape[0] != 1:
                            raise RuntimeError(f"pooled embedding を期待したが token ごとの出力 {e.shape}（サーバの --pooling を確認）")
                        e = e[0]
                    vecs.append(e)
            if len(vecs) != len(chunk):
                raise RuntimeError(f"返ってきた embedding 数 {len(vecs)} != 要求 {len(chunk)}")
            arr = np.stack(vecs).astype(np.float32)
            if not np.isfinite(arr).all():
                raise RuntimeError("embedding に非有限値")
            self.norms.append(float(np.linalg.norm(arr, axis=1).mean()))
            out.append(arr)
        if not out:
            return np.zeros((0, 0), np.float32)
        return np.concatenate(out)

    def describe(self):
        return {"provider": self.name, "preset": self.preset, "url": self.url, "model": self.model, "pooling": self.pooling,
                "prefix": self.prefix, "suffix": self.suffix, "endpoint": self.endpoint, "batch": self.batch,
                "server_output_norm_mean": float(np.mean(self.norms)) if self.norms else None}


def make_provider(name, **over):
    if name not in PRESETS:
        raise SystemExit(f"unknown provider preset {name!r}; available: {sorted(PRESETS)}")
    return LlamaServerProvider.from_preset(name, **over)


# --------------------------------------------------------------------------------------
# train 文字列の収集（リーク防止の唯一の入口）
# --------------------------------------------------------------------------------------

def sha1_json(obj) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_train_shard_arrays(shard_dir, lp=None):
    """data/tok/<name>/ の **train** shard から item_id と k だけを読む（他 split は開かない）。"""
    from tb250distill.student import model as M
    path = M.find_shard(shard_dir, "train", lp)
    if path is None:
        raise SystemExit(f"train shard not found in {shard_dir}")
    z = np.load(path, allow_pickle=False)
    return path, z["item_id"].astype(np.int64), z["k"].astype(np.int32)


def collect_train_strings(conn, item_id, k):
    """train shard の行 item_id/k に対応する候補文字列を DB から引き、(strings, cand_idx[N,Kmax]) を返す。
    DB の行の split が全て 'train' で、候補数が shard の k と一致しなければ拒否する。
    strings は train 行 × 候補位置の初出順の unique（完全一致）。"""
    n = len(item_id)
    kmax = int(k.max()) if n else 0
    rows = {}
    ids = [int(x) for x in item_id]
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT item_id, split, candidates FROM items WHERE item_id IN ({q})", chunk):
            rows[int(r[0])] = (r[1], json.loads(r[2]))
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"shard の item_id が DB に無い: {len(missing)} 件（例 {missing[:3]}）")
    bad = sorted({rows[i][0] for i in ids} - {"train"})
    if bad:
        raise LeakError(f"train shard の item が train 以外の split を含む: {bad}（val/test/robust の文字列は embedding しない）")
    index, strings = {}, []
    cand_idx = np.full((n, kmax), -1, np.int32)
    for r, iid in enumerate(ids):
        cands = rows[iid][1]
        if len(cands) != int(k[r]):
            raise SystemExit(f"item {iid}: DB の候補数 {len(cands)} != shard の k {int(k[r])}（shard と DB が不整合）")
        for j, c in enumerate(cands):
            u = index.get(c)
            if u is None:
                u = index[c] = len(strings)
                strings.append(c)
            cand_idx[r, j] = u
    return strings, cand_idx


# --------------------------------------------------------------------------------------
# 取得（再開可能）と PCA
# --------------------------------------------------------------------------------------

def _atomic_save_npy(path, arr):
    tmp = str(path) + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _log(msg):
    print(msg, flush=True)


def fetch_embeddings(provider, strings, out_dir, chunk=512, log=_log):
    """strings を chunk 件ずつ provider へ。<out_dir>/parts/chunk_<i>.npy に保存し、再実行時は取得済みを飛ばす。
    戻り値: (emb_raw[U,D] float32, stats)。provider へ渡す文字列は strings の部分列だけ（リーク防止テストで検証）。"""
    parts = Path(out_dir) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    nchunks = (len(strings) + chunk - 1) // chunk
    t0 = time.perf_counter()
    done = fetched = 0
    arrs = []
    for ci in range(nchunks):
        p = parts / f"chunk_{ci:06d}.npy"
        lo, hi = ci * chunk, min(len(strings), (ci + 1) * chunk)
        if p.exists():
            a = np.load(p)
            if a.shape[0] != hi - lo:
                raise SystemExit(f"{p}: 行数 {a.shape[0]} != {hi - lo}（chunk サイズが前回と違う? --chunk を揃えるか parts を別名へ退避）")
            done += 1
        else:
            a = provider.embed(strings[lo:hi]).astype(np.float32)
            if a.shape[0] != hi - lo:
                raise RuntimeError("provider の出力行数が不一致")
            _atomic_save_npy(p, a)
            fetched += hi - lo
        arrs.append(a)
        if (ci + 1) % 20 == 0 or ci + 1 == nchunks:
            el = time.perf_counter() - t0
            log(f"  embedded {min(len(strings), (ci + 1) * chunk)}/{len(strings)} ({el:.0f}s, resumed chunks {done})")
    emb = np.concatenate(arrs) if arrs else np.zeros((0, 0), np.float32)
    dims = {a.shape[1] for a in arrs}
    if len(dims) > 1:
        raise RuntimeError(f"chunk 間で次元が違う: {dims}")
    return emb, {"seconds_this_run": time.perf_counter() - t0, "resumed_chunks": done, "chunks": nchunks,
                 "strings_embedded_this_run": fetched}


def fit_pca(emb_raw, d, l2_normalize=True):
    """train の unique 集合だけで PCA。戻り値: dict(mean, components[d,D], explained_variance[d], explained_variance_ratio[d],
    total_variance, l2_normalize) と、射影済み emb_pca[U,d] float32。"""
    x = np.asarray(emb_raw, np.float64)
    U, D = x.shape
    if U < 2:
        raise SystemExit("PCA には 2 文字列以上が必要")
    if l2_normalize:
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    mean = x.mean(axis=0)
    xc = x - mean
    d = int(min(d, D, U - 1))
    cov = (xc.T @ xc) / (U - 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    comp = v[:, :d].T.copy()
    # 符号の固定（最大絶対値の成分が正）
    sgn = np.sign(comp[np.arange(d), np.abs(comp).argmax(axis=1)])
    sgn[sgn == 0] = 1.0
    comp *= sgn[:, None]
    total = float(w.clip(min=0).sum())
    ev = w[:d].clip(min=0)
    pca = {"mean": mean, "components": comp, "explained_variance": ev,
           "explained_variance_ratio": ev / max(total, 1e-300), "total_variance": np.float64(total),
           "l2_normalize": np.int64(1 if l2_normalize else 0)}
    emb_pca = (xc @ comp.T).astype(np.float32)
    return pca, emb_pca


def apply_pca(emb_raw, pca):
    """fit_pca と同じ変換を新しい文字列へ適用する（評価専用 embedding を学習用 PCA で 128 次元へ射影する）。
    pca は np.load した pca.npz（mean / components / l2_normalize）。戻り値 float32 [n, d]。"""
    x = np.asarray(emb_raw, np.float64)
    if int(pca["l2_normalize"]):
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return ((x - pca["mean"]) @ pca["components"].T).astype(np.float32)


def file_sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def extract(provider, shard_dir, db_path, out_root="data/emb", provider_name=None, lp=None, dim=128, chunk=512,
            log=_log, conn=None):
    """train shard の候補文字列 -> embedding -> PCA -> data/emb/<provider>/<shard_name>/ に書く。meta の dict を返す。"""
    from tb250distill import replay
    shard_dir = str(shard_dir)
    shard_name = os.path.basename(os.path.normpath(shard_dir))
    pname = provider_name or getattr(provider, "preset", None) or provider.name
    out = Path(out_root) / pname / shard_name
    out.mkdir(parents=True, exist_ok=True)
    tr_path, item_id, k = load_train_shard_arrays(shard_dir, lp)
    own = conn is None
    if own:
        conn = replay.connect(db_path, readonly=True)
    try:
        strings, cand_idx = collect_train_strings(conn, item_id, k)
    finally:
        if own:
            conn.close()
    sh1 = sha1_json(strings)
    sj = out / "strings.json"
    if sj.exists():
        if sha1_json(json.loads(sj.read_text())) != sh1:
            raise SystemExit(f"{sj} が今回の train 文字列集合と違う。別の出力先を使うか parts/ と一緒に退避する")
    else:
        tmp = str(sj) + ".tmp"
        Path(tmp).write_text(json.dumps(strings, ensure_ascii=False))
        os.replace(tmp, sj)
    log(f"train items {len(item_id)}, unique candidate strings {len(strings)} -> {out}")
    t_all = time.perf_counter()
    emb_raw, st = fetch_embeddings(provider, strings, out, chunk=chunk, log=log)
    pca, emb_pca = fit_pca(emb_raw, dim, l2_normalize=True)
    d = int(pca["components"].shape[0])
    _atomic_save_npy(out / "emb_raw.npy", emb_raw)
    tmp = str(out / "pca.tmp.npz")
    np.savez(tmp, **pca)
    os.replace(tmp, out / "pca.npz")
    _atomic_save_npy(out / "emb_pca.npy", emb_pca)
    _atomic_save_npy(out / "cand_idx_train.npy", cand_idx)
    desc = provider.describe()
    model_path = getattr(provider, "model_path", None)
    meta = {
        **desc, "provider": pname, "shard": os.path.basename(os.path.normpath(shard_dir)), "train_shard_file": os.path.basename(tr_path),
        "split": "train", "n_items": int(len(item_id)), "item_id_sha1": hashlib.sha1(item_id.astype(np.int64).tobytes()).hexdigest(),
        "strings_sha1": sh1, "U": int(emb_raw.shape[0]), "D": int(emb_raw.shape[1]), "d": d,
        "explained_variance_ratio": float(pca["explained_variance_ratio"].sum()),
        "explained_variance_ratio_first": [float(x) for x in pca["explained_variance_ratio"][:8]],
        "pca": {"fit_on": "unique train candidate strings only", "l2_normalize_before": True, "center": "mean of train unique set",
                "whiten": False, "l2_normalize_after": False,
                "note": "Student の cosine loss は target のスケールに不変なので PCA 後は再正規化しない"},
        "model_file": os.path.basename(model_path) if model_path else None,
        "sha256": file_sha256(model_path) if model_path and os.path.isfile(model_path) else None,
        "extract_seconds_total": time.perf_counter() - t_all, **st,
        "kmax": int(cand_idx.shape[1]), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": ["strings.json", "emb_raw.npy", "pca.npz", "emb_pca.npy", "cand_idx_train.npy", "meta.json"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    shutil.rmtree(out / "parts", ignore_errors=True)
    log(f"done: U={meta['U']} D={meta['D']} d={d} explained_variance_ratio={meta['explained_variance_ratio']:.4f} "
        f"({meta['extract_seconds_total']:.1f}s)")
    return meta


# --------------------------------------------------------------------------------------
# --exclude-truncated: 切り詰められる候補文字列を意味表現蒸留 loss から除く
# --------------------------------------------------------------------------------------

UNK_ID = 1   # tokenize_data.py の encode と同じ（空 encode は [UNK] 1 token）


def spm_token_lengths(strings, spm_model):
    """tokenize_data.build_shard と同じ規則（`sp.encode(text) or [UNK]`、特殊 token なし）の token 長 int32[U]。"""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(spm_model))
    return np.array([len(sp.encode(t) or [UNK_ID]) for t in strings], np.int32)


def _link_or_copy(src, dst):
    """dst が無ければ src のハードリンク（別 fs 等で不可なら symlink）。既にあれば中身が同じことだけ確認する（上書きしない）。"""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size or file_sha256(dst) != file_sha256(src):
            raise SystemExit(f"{dst} が既にあり {src} と内容が違う（上書きしない）")
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        os.symlink(os.path.abspath(src), dst)
        return "symlink"


def make_notrunc_variant(base_dir, out_dir, spm_model, lc, sources, log=_log):
    """base_dir（extract の出力）から out_dir に --exclude-truncated 版を作る。戻り値: meta dict。
    sources: train shard 行ごとの source 名の配列[N]（source 別の除外件数用。None なら全て "all"）。
    base_dir / out_dir は同一不可。out_dir に cand_idx_train.npy があれば、同じ内容の再実行のみ許す。"""
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    if os.path.abspath(base_dir) == os.path.abspath(out_dir):
        raise SystemExit("--exclude-truncated の出力先が元と同じ（元は上書きしない）")
    for f in ("strings.json", "emb_raw.npy", "pca.npz", "emb_pca.npy", "cand_idx_train.npy", "meta.json"):
        if not (base_dir / f).exists():
            raise SystemExit(f"{base_dir}/{f} が無い（先に通常の extract を行う）")
    strings = json.loads((base_dir / "strings.json").read_text())
    ci = np.load(base_dir / "cand_idx_train.npy")
    bmeta = json.loads((base_dir / "meta.json").read_text())
    if bmeta.get("exclude_truncated", {}).get("applied"):
        raise SystemExit(f"{base_dir} は既に --exclude-truncated 版（元の extract 出力を渡す）")
    n, kmax = ci.shape
    src = np.asarray(sources) if sources is not None else np.array(["all"] * n)
    if len(src) != n:
        raise SystemExit(f"source の行数 {len(src)} != cand_idx の行数 {n}")
    tl = spm_token_lengths(strings, spm_model)
    trunc_u = tl > int(lc)                                  # unique 文字列ごとの「切り詰められる」
    valid = ci >= 0
    trunc_slot = valid & trunc_u[np.where(valid, ci, 0)]
    new_ci = np.where(trunc_slot, -1, ci).astype(np.int32)
    by_src = {}
    for sname in sorted(set(src.tolist())):
        m = src == sname
        us = np.unique(ci[m][valid[m]])
        by_src[sname] = {"candidate_slots": int(valid[m].sum()), "excluded_slots": int(trunc_slot[m].sum()),
                         "excluded_slot_frac": float(trunc_slot[m].sum() / max(1, valid[m].sum())),
                         "unique_strings": int(len(us)), "excluded_unique_strings": int(trunc_u[us].sum()),
                         "items": int(m.sum()), "items_with_excluded": int(trunc_slot[m].any(axis=1).sum())}
    out_dir.mkdir(parents=True, exist_ok=True)
    cp = out_dir / "cand_idx_train.npy"
    if cp.exists() and not np.array_equal(np.load(cp), new_ci):
        raise SystemExit(f"{cp} が既にあり今回の結果と違う（上書きしない。別名にする）")
    if not cp.exists():
        _atomic_save_npy(cp, new_ci)
    links = {f: _link_or_copy(base_dir / f, out_dir / f) for f in ("strings.json", "emb_raw.npy", "pca.npz", "emb_pca.npy")}
    spm_sha1 = hashlib.sha1(Path(spm_model).read_bytes()).hexdigest()
    meta = dict(bmeta)
    meta.update({
        "variant": "notrunc", "base_dir": str(base_dir),
        "exclude_truncated": {
            "applied": True, "lc": int(lc), "rule": "spm token 長（sp.encode(text) or [UNK]）> Lc の候補文字列の位置を cand_idx で -1 にする",
            "spm_model": str(spm_model), "spm_sha1": spm_sha1,
            "excluded_slots": int(trunc_slot.sum()), "candidate_slots": int(valid.sum()),
            "excluded_unique_strings": int(trunc_u[np.unique(ci[valid])].sum()), "unique_strings_referenced": int(len(np.unique(ci[valid]))),
            "excluded_by_source": by_src,
            "scope": "意味表現蒸留 loss の target 位置だけ。判断用 KD/CE は shard の全候補を使う（影響なし）",
            "embedding_files": links,
        },
        "files": ["strings.json", "emb_raw.npy", "pca.npz", "emb_pca.npy", "cand_idx_train.npy", "meta.json"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    ex = meta["exclude_truncated"]
    log(f"notrunc: Lc={lc} excluded slots {ex['excluded_slots']}/{ex['candidate_slots']} "
        f"(unique strings {ex['excluded_unique_strings']}/{ex['unique_strings_referenced']}) -> {out_dir}")
    for sname, v in by_src.items():
        log(f"  {sname}: excluded slots {v['excluded_slots']}/{v['candidate_slots']} ({v['excluded_slot_frac']:.3f}), "
            f"unique {v['excluded_unique_strings']}/{v['unique_strings']}")
    return meta


def load_train_shard_sources(shard_dir, lp=None):
    """train shard の source 配列（無ければ None）と cand の Lc。"""
    from tb250distill.student import model as M
    path = M.find_shard(shard_dir, "train", lp)
    z = np.load(path, allow_pickle=False)
    return (z["source"].astype(str) if "source" in z.files else None), int(z["cand"].shape[2])


# --------------------------------------------------------------------------------------
# llama-server の起動・停止（embedding 用。既存 teacher の pid/port には触れない）
# --------------------------------------------------------------------------------------

def _pid_file(name):
    return RUN_DIR / f"{name}.pid"


def _read_pid(name):
    try:
        pid = int(_pid_file(name).read_text().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def produce_running() -> bool:
    # "-m tb250distill.teacher.produce" は produce の起動コマンドにだけ現れる（単に teacher.produce を含む別プロセスの引数を拾わない）
    r = subprocess.run(["pgrep", "-f", "--", "-m tb250distill[.]teacher[.]produce"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def server_cmd(sp: ProviderSpec, port=None, device="Vulkan0", cpu=False):
    return [str(LLAMA_DIR / "llama-server"), "-m", os.path.expanduser(sp.model_path), "--embeddings", "--pooling", sp.pooling,
            "-c", str(sp.ctx), "-np", str(sp.parallel), "-b", str(sp.ubatch), "-ub", str(sp.ubatch), "-t", "2",
            "--host", "127.0.0.1", "--port", str(port or sp.port)] + (["-ngl", "0"] if cpu else ["--device", device, "-ngl", "99"])


def serve_start(name, port=None, device="Vulkan0", cpu=False, wait=180.0):
    sp = PRESETS[name]
    port = port or sp.port
    if port == TEACHER_PORT:
        raise SystemExit(f"port {TEACHER_PORT} は既存 teacher 用。別ポートを使う")
    import requests
    if not cpu and produce_running():
        raise SystemExit("teacher.produce が動作中。RX 6400 は終了後に使う（CPU で API だけ試すなら --cpu）")
    if not Path(os.path.expanduser(sp.model_path)).exists():
        raise SystemExit(f"model not found: {sp.model_path}")
    if _read_pid(name):
        print("already running pid", _read_pid(name))
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(LLAMA_DIR) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["VK_ICD_FILENAMES"] = ICD   # NVIDIA 390 の Vulkan ICD が llama.cpp を落とすため必須
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log = open(RUN_DIR / f"{name}.log", "ab")
    proc = subprocess.Popen(server_cmd(sp, port, device, cpu), env=env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    _pid_file(name).write_text(str(proc.pid))
    t0 = time.time()
    while time.time() - t0 < wait:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}); see {RUN_DIR / (name + '.log')}")
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                print(f"started pid={proc.pid} port={port} pooling={sp.pooling}")
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    raise TimeoutError("llama-server did not become healthy")


def serve_stop(name, timeout=20.0):
    pid = _read_pid(name)
    if pid is None:
        print("not running")
        return
    os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.3)
    else:
        os.kill(pid, signal.SIGKILL)
    _pid_file(name).unlink(missing_ok=True)
    print("stopped")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve")
    sv.add_argument("action", choices=["start", "stop", "status"])
    sv.add_argument("--provider", required=True, choices=sorted(PRESETS))
    sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--device", default="Vulkan0")
    sv.add_argument("--cpu", action="store_true", help="GPU を使わず CPU で起動（API 確認用。teacher.produce 動作中でも可）")
    ex = sub.add_parser("extract")
    ex.add_argument("--provider", required=True, choices=sorted(PRESETS))
    ex.add_argument("--shard", required=True, help="data/tok/<name>（train shard だけを読む）")
    ex.add_argument("--db", default="data/replay.sqlite")
    ex.add_argument("--lp", type=int, default=None)
    ex.add_argument("--dim", type=int, default=128, help="PCA 次元 d")
    ex.add_argument("--out-root", default="data/emb")
    ex.add_argument("--url", default=None)
    ex.add_argument("--batch", type=int, default=32, help="1 リクエストの文字列数")
    ex.add_argument("--chunk", type=int, default=512, help="再開単位（parts に保存する文字列数）")
    ex.add_argument("--endpoint", default="native", choices=["native", "v1"])
    ex.add_argument("--exclude-truncated", action="store_true",
                    help="spm token 長 > Lc の候補文字列を意味表現蒸留 loss の対象から外した <shard>_notrunc/ を作る（元は上書きしない・再抽出しない）")
    ex.add_argument("--spm", default=None, help="--exclude-truncated 用 spm.model（既定 <shard>/spm.model）")
    ex.add_argument("--lc", type=int, default=None, help="--exclude-truncated の Lc（既定: train shard の候補 pad 長）")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        if a.action == "start":
            serve_start(a.provider, a.port, a.device, a.cpu)
        elif a.action == "stop":
            serve_stop(a.provider)
        else:
            print(json.dumps({"pid": _read_pid(a.provider), "spec": asdict(PRESETS[a.provider])}, indent=1))
        return 0
    shard_name = os.path.basename(os.path.normpath(a.shard))
    base = Path(a.out_root) / a.provider / shard_name
    if a.exclude_truncated and (base / "meta.json").exists():
        _log(f"{base} は作成済み。再抽出せず notrunc 版だけ作る")
    else:
        prov = make_provider(a.provider, url=a.url, batch=a.batch, endpoint=a.endpoint)
        extract(prov, a.shard, a.db, a.out_root, provider_name=a.provider, lp=a.lp, dim=a.dim, chunk=a.chunk)
    if a.exclude_truncated:
        sources, lc_shard = load_train_shard_sources(a.shard, a.lp)
        make_notrunc_variant(base, Path(a.out_root) / a.provider / (shard_name + "_notrunc"),
                             a.spm or os.path.join(a.shard, "spm.model"), a.lc or lc_shard, sources)
    return 0


if __name__ == "__main__":
    sys.exit(main())
