"""投機的デコードベンチの構成（BenchConfig）・llama-server のコマンドライン・実験プラン。

必須環境変数: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json（build_env が設定する）。
llama.cpp b11384 のフラグだけを使う（古い --draft-max 等は削除済みでエラーになる）:
  -md/--spec-draft-model, -devd/--spec-draft-device, -ngld/--spec-draft-ngl, --spec-type,
  --spec-draft-n-max / --spec-draft-n-min / --spec-draft-p-min
`-fit off` は必須（既定 on だと ngl 等を勝手に調整して条件が固定できない）。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

HOME = Path.home()
LLAMA_DIR = HOME / "bench" / "llama-bin" / "llama-b11384"
MODEL_DIR = HOME / "bench" / "models"
ICD = "/usr/share/vulkan/icd.d/radeon_icd.json"
DEFAULT_PORT = 18190

T8_Q2 = str(MODEL_DIR / "qwen3-8b-base-q2_k.gguf")
T8_Q3 = str(MODEL_DIR / "qwen3-8b-base-q3_k_s.gguf")
D17 = str(MODEL_DIR / "Qwen3-1.7B-Q4_K_M.gguf")      # Qwen3 系なので 8B と語彙互換
# Qwen3-0.6B（2026-10-07 取得。~/bench/specbench-prep/prep06.sh）。I = 公式 GGUF（post-trained）、B = Base を自前で量子化
D06I_Q8 = str(MODEL_DIR / "qwen3-0.6b-q8_0.gguf")
D06B_Q8 = str(MODEL_DIR / "qwen3-0.6b-base-q8_0.gguf")
D06B_Q4 = str(MODEL_DIR / "qwen3-0.6b-base-q4_k_m.gguf")

SPEC_TYPES = ("none", "draft-simple", "ngram-simple", "ngram-mod")
DRAFT_DEFAULT_N_MAX = 3                               # llama-server の --spec-draft-n-max 既定値


@dataclass
class BenchConfig:
    name: str
    target_model: str
    target_devices: str = "Vulkan0"          # "Vulkan0" / "Vulkan0,Vulkan1"
    split_mode: str = "none"                 # "none" / "layer"
    tensor_split: str | None = None          # 例 "3,1"
    ngl: str = "all"
    draft_model: str | None = None
    draft_device: str = "Vulkan1"            # "Vulkan0" / "Vulkan1" / "none"（CPU）
    ngld: str = "all"
    spec_type: str = "none"                  # none / draft-simple / ngram-simple / ngram-mod
    n_max: int | None = None
    n_min: int | None = None
    p_min: float | None = None
    ctx: int = 1024
    cache_type: str | None = None            # 例 "q8_0" -> -ctk/-ctv
    threads: int = 2
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def uses_draft_model(self) -> bool:
        return self.spec_type != "none" and self.draft_model is not None

    @property
    def n_max_effective(self) -> int | None:
        """受理長の推定に使うドラフト長の上限。draft-simple は未指定なら既定値 3、ngram 系は不明（None）。"""
        if self.n_max is not None:
            return self.n_max
        return DRAFT_DEFAULT_N_MAX if self.spec_type == "draft-simple" else None

    def models(self) -> list[str]:
        return [m for m in (self.target_model, self.draft_model if self.uses_draft_model else None) if m]


def build_env(llama_dir: str | os.PathLike = LLAMA_DIR) -> dict:
    env = dict(os.environ)
    ld = str(llama_dir)
    env["LD_LIBRARY_PATH"] = ld + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["VK_ICD_FILENAMES"] = ICD
    return env


def build_cmd(cfg: BenchConfig, llama_dir: str | os.PathLike = LLAMA_DIR, port: int = DEFAULT_PORT,
              host: str = "127.0.0.1") -> list[str]:
    """llama-server のコマンドライン。spec_type="none" のとき draft 系フラグは一切付けない。"""
    if cfg.spec_type not in SPEC_TYPES:
        raise ValueError(f"{cfg.name}: unknown spec_type {cfg.spec_type!r}")
    if cfg.split_mode not in ("none", "layer"):
        raise ValueError(f"{cfg.name}: unknown split_mode {cfg.split_mode!r}")
    if cfg.spec_type == "draft-simple" and not cfg.draft_model:
        raise ValueError(f"{cfg.name}: draft-simple には draft_model が必要")
    cmd = [
        str(Path(llama_dir) / "llama-server"),
        "-m", cfg.target_model,
        "-dev", cfg.target_devices,
        "-ngl", cfg.ngl,
        "-sm", cfg.split_mode,
    ]
    if cfg.tensor_split:
        cmd += ["-ts", cfg.tensor_split]
    cmd += ["-c", str(cfg.ctx), "-np", "1", "-t", str(cfg.threads), "-fit", "off"]
    if cfg.cache_type:
        cmd += ["-ctk", cfg.cache_type, "-ctv", cfg.cache_type]
    if cfg.spec_type != "none":
        cmd += ["--spec-type", cfg.spec_type]
        if cfg.draft_model:
            cmd += ["-md", cfg.draft_model, "-devd", cfg.draft_device, "-ngld", cfg.ngld]
        if cfg.n_max is not None:
            cmd += ["--spec-draft-n-max", str(cfg.n_max)]
        if cfg.n_min is not None:
            cmd += ["--spec-draft-n-min", str(cfg.n_min)]
        if cfg.p_min is not None:
            cmd += ["--spec-draft-p-min", str(cfg.p_min)]
    cmd += ["--host", host, "--port", str(port)]
    return cmd


# --------------------------------------------------------------------------------------
# プラン
# --------------------------------------------------------------------------------------

def _smoke() -> tuple[str, list[BenchConfig]]:
    base = BenchConfig("d17_rx", D17, note="smoke baseline: D17 を Vulkan0 単独、投機なし")
    spec = BenchConfig("d17_rx__d17wx_n4", D17, draft_model=D17, draft_device="Vulkan1", spec_type="draft-simple",
                       n_max=4, note="smoke: draft=D17 on Vulkan1（ハーネス確認用）")
    return base.name, [base, spec]


def _main() -> tuple[str, list[BenchConfig]]:
    ref = BenchConfig("t8q2_rx", T8_Q2, note="reference: T8_Q2 を Vulkan0 単独、投機なし")

    def draft(name: str, *, device: str = "Vulkan1", n_max: int | None = 4, p_min: float | None = None,
              note: str = "") -> BenchConfig:
        return BenchConfig(name, T8_Q2, draft_model=D17, draft_device=device, spec_type="draft-simple",
                           n_max=n_max, p_min=p_min, note=note)

    cfgs = [
        ref,
        draft("t8q2_rx__d17wx_n2", n_max=2, note="draft D17 を WX 2100（Vulkan1）、n_max 2"),
        draft("t8q2_rx__d17wx_n4", n_max=4, note="draft D17 を WX 2100、n_max 4"),
        draft("t8q2_rx__d17wx_n8", n_max=8, note="draft D17 を WX 2100、n_max 8"),
        draft("t8q2_rx__d17wx_n8_p075", n_max=8, p_min=0.75, note="n_max 8 + p_min 0.75（自信が低いドラフトを打ち切る）"),
        draft("t8q2_rx__d17rx_n4", device="Vulkan0", n_max=4,
              note="draft を同じ RX 6400 に置く。VRAM 不足で起動失敗する可能性あり（それも結果）"),
        draft("t8q2_rx__d17cpu_n4", device="none", n_max=4,
              note="draft を CPU（Celeron G3930、AVX 無し）。遅い対照"),
        BenchConfig("t8q2_rx__ngram_simple", T8_Q2, spec_type="ngram-simple", note="ドラフトモデル無し、既定パラメータ"),
        BenchConfig("t8q2_rx__ngram_mod", T8_Q2, spec_type="ngram-mod", note="ドラフトモデル無し、既定パラメータ"),
        BenchConfig("t8q2_split", T8_Q2, target_devices="Vulkan0,Vulkan1", split_mode="layer",
                    note="T8_Q2 を 2 GPU に -sm layer（分割のオーバーヘッドだけを見る）"),
        BenchConfig("t8q3_split", T8_Q3, target_devices="Vulkan0,Vulkan1", split_mode="layer",
                    note="T8_Q3 を 2 GPU に -sm layer（第 2 GPU を容量として使い、より良い量子化を載せる案）"),
        BenchConfig("t8q3_rx", T8_Q3, cache_type="q8_0", note="T8_Q3 を Vulkan0 単独、KV q8_0（載らない可能性あり）"),
        BenchConfig("t8q3_rx__d17wx_n4", T8_Q3, cache_type="q8_0", draft_model=D17, draft_device="Vulkan1",
                    spec_type="draft-simple", n_max=4, note="t8q3_rx の上に draft D17 を WX 2100"),
    ]
    return ref.name, cfgs


def _draft06() -> tuple[str, list[BenchConfig]]:
    """main1 で 1.7B ドラフトは WX 2100 上で遅すぎた。より小さい 0.6B（Base / post-trained、Q8_0 / Q4_K_M）で再検証する。"""
    ref = BenchConfig("t8q2_rx", T8_Q2, note="reference: T8_Q2 を Vulkan0 単独、投機なし（同一セッションで再計測）")

    def draft(name: str, model: str, *, target: str = T8_Q2, device: str = "Vulkan1", n_max: int = 4,
              cache_type: str | None = None, note: str = "") -> BenchConfig:
        return BenchConfig(name, target, draft_model=model, draft_device=device, spec_type="draft-simple",
                           n_max=n_max, cache_type=cache_type, note=note)

    cfgs = [
        ref,
        draft("t8q2_rx__d06b8wx_n2", D06B_Q8, n_max=2, note="draft 0.6B-Base Q8_0 を WX 2100、n_max 2"),
        draft("t8q2_rx__d06b8wx_n4", D06B_Q8, n_max=4, note="draft 0.6B-Base Q8_0 を WX 2100、n_max 4"),
        draft("t8q2_rx__d06b8wx_n8", D06B_Q8, n_max=8, note="draft 0.6B-Base Q8_0 を WX 2100、n_max 8"),
        draft("t8q2_rx__d06b4wx_n4", D06B_Q4, note="draft 0.6B-Base Q4_K_M を WX 2100（帯域律速なら Q8 より速い）"),
        draft("t8q2_rx__d06iwx_n4", D06I_Q8, note="draft 0.6B（post-trained、公式 Q8_0）を WX 2100。Base との受理率差を見る"),
        draft("t8q2_rx__d06b4rx_n4", D06B_Q4, device="Vulkan0",
              note="draft を同じ RX 6400 に同居（VRAM に収まるか、GTT へはみ出すかを gtt_mb で見る）"),
        draft("t8q2_rx__d06b4cpu_n4", D06B_Q4, device="none", note="draft を CPU（Celeron、AVX 無し）。対照"),
        BenchConfig("t8q2_rx__ngram_mod", T8_Q2, spec_type="ngram-mod", note="main1 の最良。同一セッションで再計測"),
        BenchConfig("t8q3_rx", T8_Q3, cache_type="q8_0", note="T8_Q3 を Vulkan0 単独、KV q8_0（Q3 側の基準）"),
        draft("t8q3_rx__d06b4wx_n4", D06B_Q4, target=T8_Q3, cache_type="q8_0",
              note="T8_Q3（KV q8_0）+ draft 0.6B-Base Q4_K_M を WX 2100"),
    ]
    return ref.name, cfgs


PLANS: dict[str, tuple[str, list[BenchConfig]]] = {
    "smoke": _smoke(),
    "main": _main(),
    "draft06": _draft06(),
}


def get_plan(name: str, only: list[str] | None = None) -> tuple[str, list[BenchConfig]]:
    """(reference 名, configs)。only があればその名前だけに絞る（プラン内の順序を保つ。未知の名前は ValueError）。"""
    if name not in PLANS:
        raise ValueError(f"unknown plan {name!r}（{sorted(PLANS)}）")
    ref, cfgs = PLANS[name]
    if only:
        names = {c.name for c in cfgs}
        unknown = [o for o in only if o not in names]
        if unknown:
            raise ValueError(f"プラン {name} に無い config: {unknown}（--list で確認）")
        cfgs = [c for c in cfgs if c.name in set(only)]
    return ref, list(cfgs)


def describe(cfg: BenchConfig) -> str:
    parts = [f"target={Path(cfg.target_model).name}@{cfg.target_devices}"]
    if cfg.split_mode != "none":
        parts.append(f"sm={cfg.split_mode}")
    if cfg.spec_type != "none":
        parts.append(f"spec={cfg.spec_type}")
        if cfg.draft_model:
            parts.append(f"draft={Path(cfg.draft_model).name}@{cfg.draft_device}")
        if cfg.n_max is not None:
            parts.append(f"n_max={cfg.n_max}")
        if cfg.p_min is not None:
            parts.append(f"p_min={cfg.p_min}")
    if cfg.cache_type:
        parts.append(f"kv={cfg.cache_type}")
    return " ".join(parts)
