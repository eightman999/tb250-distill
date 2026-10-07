# SPECBENCH: 異種 GPU 投機的デコードのベンチ

llama.cpp（b11384、Vulkan）の投機的デコードを、RX 6400（Vulkan0、4 GB）と Radeon Pro WX 2100（Vulkan1、2 GB）の組み合わせで測る。実装は `tb250distill/specbench/`。

## 目的

「主 GPU（RX 6400）に本体モデル、弱い第 2 GPU（WX 2100）にドラフトモデルを置く投機的デコードは速くなるか。第 2 GPU は『ドラフト置き場』と『容量（layer 分割）』のどちらに使うのが良いか」を、条件を固定して測る。

## 仮説

1. WX 2100 上の Qwen3-1.7B ドラフトは、RX 6400 上の Qwen3-8B-Base（Q2_K）より速く動くなら、本体の decode を上回る速度向上が出る。ただし WX 2100 は帯域が小さく、ドラフト生成が律速になって逆に遅くなる可能性がある。
2. `n_max` は大きいほど良いとは限らない。受理率が落ちると検証が無駄になる。`p_min` で自信の低いドラフトを打ち切ると改善する可能性がある。
3. 受理率はカテゴリで大きく変わる。コード・JSON・反復列挙は高く、日本語/英語の散文は低い見込み。
4. ドラフトを同じ RX 6400 に置くと VRAM が足りず起動しない可能性がある（それも結果として記録する）。
5. CPU（Celeron G3930、2 コア、AVX 無し）のドラフトは遅く、対照にしかならない。
6. 第 2 GPU を layer 分割に使う案は、分割のオーバーヘッドで単独より遅くなる。ただし Q3_K_S のようにより良い量子化が載せられるなら、品質と速度のトレードオフが生じる。

## 条件

共通: `-ngl all -sm none|layer -c 1024 -np 1 -t 2 -fit off`（`-fit off` で ngl 等の自動調整を止め、条件を固定する）。モデルは次の 3 つ。

| 定数 | ファイル（tb250 の `~/bench/models/`） | 役割 |
|---|---|---|
| T8_Q2 | `qwen3-8b-base-q2_k.gguf` | 本体（Q2_K） |
| T8_Q3 | `qwen3-8b-base-q3_k_s.gguf` | 本体（Q3_K_S） |
| D17 | `Qwen3-1.7B-Q4_K_M.gguf` | ドラフト（Qwen3 系なので語彙互換） |

`main` プラン（reference は `t8q2_rx`）:

| config | 本体 | ドラフト | 備考 |
|---|---|---|---|
| `t8q2_rx` | T8_Q2 @ Vulkan0 | なし | reference |
| `t8q2_rx__d17wx_n2` / `_n4` / `_n8` | T8_Q2 @ Vulkan0 | D17 @ Vulkan1（draft-simple）、n_max 2 / 4 / 8 | 主な検証 |
| `t8q2_rx__d17wx_n8_p075` | T8_Q2 @ Vulkan0 | D17 @ Vulkan1、n_max 8、p_min 0.75 | |
| `t8q2_rx__d17rx_n4` | T8_Q2 @ Vulkan0 | D17 @ Vulkan0、n_max 4 | VRAM 不足で起動失敗しうる |
| `t8q2_rx__d17cpu_n4` | T8_Q2 @ Vulkan0 | D17 @ CPU（`-devd none`）、n_max 4 | 遅い対照 |
| `t8q2_rx__ngram_simple` / `_ngram_mod` | T8_Q2 @ Vulkan0 | なし（ngram 投機、既定パラメータ） | ドラフトモデル無し |
| `t8q2_split` | T8_Q2 @ Vulkan0,Vulkan1（`-sm layer`） | なし | 分割のオーバーヘッドだけを見る |
| `t8q3_split` | T8_Q3 @ Vulkan0,Vulkan1（`-sm layer`） | なし | 第 2 GPU を容量に使い、より良い量子化を載せる案 |
| `t8q3_rx` | T8_Q3 @ Vulkan0、KV `q8_0` | なし | 載らない可能性あり |
| `t8q3_rx__d17wx_n4` | T8_Q3 @ Vulkan0、KV `q8_0` | D17 @ Vulkan1、n_max 4 | |

`smoke` プランは D17 を本体にした 2 本（単独と、D17 を Vulkan1 のドラフトにしたもの）で、ハーネスの確認用。

プロンプトは 8 本（code_py / code_c / json / list / ja_explain / ja_prose / en_explain / en_story、`prompts.py`）。

## 実行手順

1. モデル準備（tb250）: `~/bench/specbench-prep/prep.sh` が Qwen3-8B-Base を HDD 上で f16 GGUF に変換し、Q2_K / Q3_K_S を `~/bench/models/` に作る。`Qwen3-1.7B-Q4_K_M.gguf` は Teacher 用に既にある。
2. Mac で変更して tb250 へ同期する（同期の方法は既存の `scripts/` に従う）。
3. tb250 で、他の llama-server（Teacher 等）を止めてから実行する。起動前に他の llama-server プロセスが居れば中止する。

```bash
# コマンドラインの確認（存在チェック・起動なし）
python -m tb250distill.specbench.run --plan main --out /tmp/x --dry-run
python -m tb250distill.specbench.run --plan main --list

# ハーネス確認
python -m tb250distill.specbench.run --plan smoke --out runs/specbench/smoke --reps 1

# 本番（中断しても --resume で続きから）
python -m tb250distill.specbench.run --plan main --out runs/specbench/main1 --reps 2 --n-predict 128
python -m tb250distill.specbench.run --plan main --out runs/specbench/main1 --resume
python -m tb250distill.specbench.run --plan main --out runs/specbench/main1 --only t8q3_rx,t8q3_rx__d17wx_n4 --resume

# 集計だけやり直す
python -m tb250distill.specbench.report runs/specbench/main1
```

主なオプション: `--only a,b`（絞り込み）、`--reps`、`--n-predict`、`--port`（既定 18190）、`--llama-dir`、`--load-timeout`（既定 300 秒）、`--settle`（config 間の VRAM 解放待ち、既定 5 秒）、`--resume`、`--retry-failed`（`--resume` 時に前回 load_failed/load_timeout の config も再試行）。

起動前チェックは、Vulkan ICD・llama-server・モデルの存在、ポートの空き、`--list-devices` で Vulkan0 が RX 6400・Vulkan1 が WX 2100 であること、他の llama-server が居ないことを確認する。1 つでも違えば何も起動せず中止する。

## 出力（`<out>/`）

| ファイル | 内容 |
|---|---|
| `plan.json` | プラン名、reference、config の全パラメータ、実行引数 |
| `env.json` | 起動前チェックの結果（`--version`、`--list-devices`、モデルサイズ）、git commit、ホスト情報 |
| `status.jsonl` | config ごとの結果。`ok` / `load_failed`（health 待ち中にプロセスが落ちた。ログ末尾 40 行を含む）/ `load_timeout` / `crashed` / `request_failed` / `interrupted` / `error` |
| `results.jsonl` | request ごと。壁時計、timings、tokens_predicted、content の sha256、stop_type、`match_ref`、`draft_n` / `draft_n_accepted`、`energy_j`（合計と `_rx` / `_wx`）、request 中の最高温度、温度ガードで待った秒数 |
| `outputs.jsonl` | request ごとの生成本文 |
| `telemetry.csv` | 0.5 秒ごとの RX 6400 / WX 2100（vram_mb, temp_c, power_w, busy_pct, sclk_mhz）。`config` 列で config と対応 |
| `server-<config>.log` | llama-server の標準出力・標準エラー |
| `summary.json` / `summary.md` | 集計（下記） |

集計の指標:

- gen tok/s: `timings.predicted_per_second` の中央値。wall tok/s は HTTP 往復込み。speedup は reference 比（gen tok/s の中央値同士）。
- 受理率: Σ`draft_n_accepted` / Σ`draft_n`。受理長（推定）は受理率 × n_max（検証回数は取れないので、ドラフト長が常に n_max だと仮定した推定）。
- J/token: Σ`energy_j` / Σ`tokens_predicted`。request の時間窓で電力を台形積分して求める。prompt 処理も含み、2 枚の GPU の合計と RX/WX の内訳を出す。GPU ボードの sysfs 値で、システム全体（壁）の電力ではない。
- 各 GPU のピーク VRAM と最高温度（telemetry.csv。起動中も含む）。
- カテゴリ別 gen tok/s の表。
- `match_ref` 率: content の sha256 が reference の rep0 と一致した割合。

## 注意

- 生成長は `ignore_eos: true` と `n_predict` で固定する。EOS で止まらないので、本文の意味はあまりない（速度と受理率を測るためのもの）。
- greedy（`temperature 0`、`seed 42`、`cache_prompt false`）。ただしデバイスや量子化の経路が違うと浮動小数の差で出力が分岐しうる。`match_ref` は失敗ではなく指標。
- 投機的デコードの出力は、理論上は本体モデル単独の greedy と一致するはずだが、バッチ形状の違いで数値が変わり、分岐することがある。
- Celeron G3930（2 コア、AVX 無し）の CPU ドラフトは遅く、あくまで対照。
- 起動に失敗する config（VRAM 不足など）も `load_failed` として記録し、次の config へ進む。
- 温度ガード: いずれかの GPU が 88 ℃ 以上なら 78 ℃ 未満になるまで request を待つ（最大 30 分。待った時間は results に記録）。
- 停止は PID 指定の SIGTERM -> 待機 -> SIGKILL。`pkill -f` / `pgrep -f` は使わない。例外・Ctrl-C・SIGTERM でも finally でサーバを止める。
- `-fit off` を付けているので、VRAM に載らない構成は自動調整されずに起動失敗する。

## GT 430 / GT 710 / GT 730 を対象外にした理由

2026-10-07 に検討し、GT 430 はこのベンチの対象にしないと決めた（ユーザー判断）。根拠は次のとおり。

**llama-server のドラフト置き場（`-devd`）に指定できない（実機で確認）**
- llama.cpp b11384 には、Fermi（sm_21）で動くバックエンドが無い。
  - Vulkan: tb250 の NVIDIA は 390.157 ドライバで、その Vulkan ICD は llama.cpp を落とす。そのため `VK_ICD_FILENAMES=radeon_icd.json` を必須にしており、Vulkan デバイスは RX 6400 と WX 2100 だけになる。Fermi はそもそも Vulkan 非対応と理解している（未確認）。
  - CUDA: 現行の llama.cpp は CUDA 11 以降が前提。`~/bench/kepler-compat.patch` の Kepler 移植でも sm_35 が下限。Fermi を扱える CUDA は 8 までで、組み合わせられない。
  - OpenCL: 現行の OpenCL バックエンドは Adreno / Intel 向けで、GT 430 の OpenCL 1.1 では使えない見込み（未確認）。
- GT 710 / GT 730（Kepler）も 390 ドライバ下では同じ理由で Vulkan デバイスにならない。CUDA 経路（sm_35 ビルド＋470 ドライバへの切替）は、ドライバの変更を伴うため扱わない。

**迂回路はあるが、速くならない**
- GT 430 で LLM を動かせるのは、Fermi/CLBlast 移植版 `~/bench/llama-legacy`（llama.cpp 2e6cd4b、2023-05）だけ。これを使うには次の 2 つが必要になる。
  - 外部ドラフトを受け取る投機的デコードのツール（`speculative-simple` を改造し、b11384 のライブラリにリンクする）。
  - 語彙がそろうモデルの組み合わせ。移植版は GGJT v3 形式で、GQA に非対応。そのため Qwen3 / TinyLlama は使えず、ドラフトは OpenLLaMA-3B、本体も同じトークナイザの OpenLLaMA-7B/13B（要ダウンロード）に限られる。
- 移植版 README の GT 430 での実測（OpenLLaMA-3B Q4_0、ctx 512）は次のとおり。PCIe x1 のため、オフロード 1 層あたり約 +55 ms/token かかり、GPU を使うより CPU だけのほうが速い。
  - GPU（ngl 4〜10）: 2.40〜1.35 t/s
  - CPU のみ: 4.91 t/s
- 投機的デコードで速くなるには、ドラフトが本体よりも数倍速い必要がある。GT 430 のドラフトは本体より遅いので、受理率によらず速度は落ちる。速くなる余地があるのは本体が 1 t/s を切る場合くらいだが、その場合も本体とドラフトが Celeron 2 コアを奪い合う。また、同じドラフトなら CPU で動かしたほうが速い。
- 同じ傾向は smoke（`runs/specbench/smoke2`）でも確認した。WX 2100（GT 430 よりかなり速い）に Qwen3-1.7B のドラフトを置いた結果は次のとおりで、第 2 GPU の速度が律速になった。
  - RX 6400 単独: 80.3 tok/s
  - ドラフトを WX 2100 に置いた場合: 17.4 tok/s（0.22x）

再検討するのは、Fermi で動くバックエンドが現行の llama.cpp に入った場合か、「動くが遅い」ことの動作実証そのものを目的にする場合（OpenLLaMA-7B の取得が必要）に限る。

## 結果

### main1（2026-10-07、`runs/specbench/main1`）

条件は次のとおり。

- プラン: `main`、reps 2、n_predict 128、8 prompts
- reference: `t8q2_rx`
- llama.cpp: b11384（Vulkan、RADV）
- 13 config すべて `ok`

| config | gen tok/s | speedup | 受理率 | J/token 合計 (RX / WX) | VRAM peak MiB (RX / WX) |
|---|---:|---:|---:|---|---|
| t8q2_rx（reference） | 29.73 | 1.00x | - | 1.578 (1.459 / 0.119) | 3181 / 5 |
| t8q2_rx__d17wx_n2 | 12.52 | 0.42x | 0.79 | 2.772 (0.992 / 1.781) | 3181 / 1216 |
| t8q2_rx__d17wx_n4 | 11.80 | 0.40x | 0.69 | 2.940 (0.932 / 2.008) | 3181 / 1216 |
| t8q2_rx__d17wx_n8 | 7.28 | 0.24x | 0.53 | 4.641 (1.771 / 2.870) | 3181 / 1216 |
| t8q2_rx__d17wx_n8_p075 | 10.34 | 0.35x | 0.91 | 3.061 (1.117 / 1.945) | 3182 / 1216 |
| t8q2_rx__d17rx_n4 | 1.24 | 0.04x | 0.68 | 11.551 (8.728 / 2.823) | 3366 / 5（GTT に約 1 GB、下記） |
| t8q2_rx__d17cpu_n4 | 1.77 | 0.06x | 0.68 | 3.670 (1.488 / 2.182) | 3185 / 10 |
| t8q2_rx__ngram_simple | 32.70 | 1.10x | 0.62 | 1.371 (1.265 / 0.106) | 3182 / 5 |
| **t8q2_rx__ngram_mod** | **34.88** | **1.17x** | 0.70 | **1.228** (1.133 / 0.095) | 3181 / 5 |
| t8q2_split | 12.64 | 0.42x | - | 2.936 (1.080 / 1.857) | 1938 / 1410 |
| t8q3_split | 10.06 | 0.34x | - | 3.572 (1.394 / 2.178) | 2220 / 1534 |
| t8q3_rx（KV q8_0） | 22.59 | 0.76x | - | 2.118 (1.930 / 0.188) | 3519 / 5 |
| t8q3_rx__d17wx_n4 | 9.05 | 0.30x | 0.63 | 3.272 (1.124 / 2.148) | 3519 / 1216 |

カテゴリ別の値と match_ref は `runs/specbench/main1/summary.md` にある。

**実測から分かったこと**
- **WX 2100 にドラフトモデルを置くと、どの設定でも遅くなった（0.24〜0.42x）。** p_min 0.75 で受理率を 0.91 まで上げても 0.35x だった。律速は受理率ではなく、ドラフト側（WX 2100 上の 1.7B）の 1 トークンあたりの時間と、その間の待ちである。smoke でも、RX 6400 単独の 1.7B は 80.3 tok/s、ドラフトを WX 2100 に置くと 17.4 tok/s だった。
- **WX 2100 を容量として使う layer 分割も遅い。**
  - Q2_K を分割すると、RX 6400 単独の 29.7 tok/s が 12.6 tok/s（0.42x）に落ちた。
  - Q3_K_S は分割すると 10.1 tok/s だが、KV を q8_0 にすれば RX 6400 単独に載り、22.6 tok/s（0.76x）になる。
  - この構成では、第 2 GPU を使わないほうが速い。
- **速くなったのはドラフトモデルなしの n-gram だけ。** ngram-mod が 1.17x、ngram-simple が 1.10x で、J/token も最小だった。ただし、カテゴリによるばらつきが大きい（ngram-mod は 30〜55 tok/s）。また後述のとおり、`ignore_eos` で反復が増えて、n-gram に有利な条件になっている可能性がある（推測）。
- **ドラフトを同じ RX 6400 に置いた構成（`t8q2_rx__d17rx_n4`）は 1.24 tok/s。** 起動はするが VRAM（4 GB）に収まらず、ドラフトがシステムメモリへはみ出していた。同じ構成を再起動して `mem_info_gtt_used` を読み、GTT が 13 MiB から 1066 MiB に増えることを確認した。このとき VRAM は 3363 MiB だった。この表の VRAM 欄は `mem_info_vram_used` だけで、GTT は含まない。
- **CPU ドラフト（Celeron、AVX 無し）は 1.77 tok/s で、対照として期待どおり遅い。**
- **match_ref について:**
  - reference 自身は 8/8 一致し、再現する。
  - ドラフトの有無や、分割・量子化の違いで greedy 出力が分岐する（0.12〜0.75）。Q3_K_S は量子化が違うので一致しないのが当然。

**注意（解釈の限界）**
- `ignore_eos` で 128 トークンに固定したため、base モデルは途中から反復しやすい。n-gram の利得と受理率は、実際の用途より高く出ている可能性がある。
- ドラフトモデルは Qwen3-1.7B（post-trained）で、本体の Qwen3-8B-Base と系統がずれている。Qwen3-0.6B のような、より小さく速いドラフトは未検証（手元に無い）。
- ただし、WX 2100 上で 1.7B を動かす速度が本体の速度と同程度なので、受理率を上げても速くはならない。小さいドラフトで逆転するかどうかは別の問題として残る（→ draft06 で検証）。

### draft06（2026-10-07、`runs/specbench/draft06`）

条件は次のとおり。

- プラン: `draft06`、reps 2、n_predict 128、8 prompts
- reference: `t8q2_rx`（同じセッションで再計測、29.75 tok/s。main1 の 29.73 と一致）
- ドラフトは Qwen3-0.6B の 3 種
  - Base を自前で量子化した Q8_0 / Q4_K_M
  - 公式 GGUF（post-trained）の Q8_0
- 11 config すべて `ok`。この run から、テレメトリに GTT（`mem_info_gtt_used`）を記録している。

| config | gen tok/s | speedup | 受理率 | J/token 合計 (RX / WX) | VRAM peak MiB (RX / WX) | GTT peak MiB (RX) |
|---|---:|---:|---:|---|---|---:|
| t8q2_rx（reference） | 29.75 | 1.00x | - | 1.592 (1.470 / 0.121) | 3181 / 5 | 63 |
| t8q2_rx__d06b8wx_n2 | 17.36 | 0.58x | 0.82 | 1.971 (0.933 / 1.038) | 3181 / 751 | 60 |
| t8q2_rx__d06b8wx_n4 | 18.27 | 0.61x | 0.70 | 1.990 (0.873 / 1.117) | 3181 / 751 | 71 |
| t8q2_rx__d06b8wx_n8 | 12.38 | 0.42x | 0.55 | 3.040 (1.264 / 1.776) | 3181 / 751 | 64 |
| t8q2_rx__d06b4wx_n4 | 19.35 | 0.65x | 0.72 | 1.884 (0.871 / 1.013) | 3181 / 519 | 71 |
| t8q2_rx__d06iwx_n4 | 17.58 | 0.59x | 0.68 | 2.037 (0.888 / 1.149) | 3181 / 751 | 61 |
| **t8q2_rx__d06b4rx_n4** | **38.85** | **1.31x** | 0.72 | **1.177** (1.052 / 0.125) | 3693 / 5 | 100 |
| t8q2_rx__d06b4cpu_n4 | 4.16 | 0.14x | 0.70 | 2.113 (1.203 / 0.909) | 3181 / 5 | 66 |
| t8q2_rx__ngram_mod | 34.83 | 1.17x | 0.70 | 1.222 (1.127 / 0.095) | 3181 / 5 | 101 |
| t8q3_rx（KV q8_0） | 22.57 | 0.76x | - | 2.083 (1.933 / 0.150) | 3519 / 5 | 63 |
| t8q3_rx__d06b4wx_n4 | 15.60 | 0.52x | 0.68 | 2.101 (1.039 / 1.063) | 3519 / 519 | 61 |

カテゴリ別 gen tok/s（中央値）は次のとおり。

| config | code_c | code_py | en_explain | en_story | ja_explain | ja_prose | json | list |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t8q2_rx | 29.88 | 29.82 | 29.79 | 29.79 | 29.73 | 29.71 | 29.69 | 29.69 |
| t8q2_rx__d06b4wx_n4 | 20.11 | 22.18 | 10.94 | 18.37 | 18.01 | 19.47 | 19.23 | 23.10 |
| t8q2_rx__d06b4rx_n4 | 40.61 | 44.43 | 21.77 | 36.64 | 35.51 | 39.10 | 38.56 | 46.19 |
| t8q2_rx__ngram_mod | 48.25 | 38.61 | 29.88 | 40.62 | 35.00 | 53.91 | 36.45 | 54.83 |

**実測から分かったこと**
- **最速は、0.6B-Base Q4_K_M を本体と同じ RX 6400 に同居させた構成で、1.31x（38.85 tok/s）。** J/token も最小（1.177）で、ngram-mod の 1.17x を上回った。
  - RX 6400 の VRAM は 3693 MiB で、4 GB に収まった。
  - GTT は 100 MiB で、reference（63）や ngram-mod（101）と同程度。main1 の 1.7B 同居（+1 GB）のような、システムメモリへのはみ出しは起きていない。
  - en_explain だけは 21.8 tok/s と遅くなった（受理率が低い）。その他のカテゴリは 35〜46 tok/s。
- **WX 2100 に置くと、0.6B でも遅くなる（0.42〜0.65x）。** 1.7B（0.24〜0.42x）よりは改善したが、どの設定でも 1x に届かない。
  - Q4_K_M は Q8_0 より速かった（0.65x 対 0.61x）。WX 2100 のドラフトは帯域律速と考えられる（推測）。
  - 受理率は RX 同居と同程度（0.70〜0.72）。差はドラフト 1 トークンあたりの時間と、GPU 間の受け渡しにあると考えられる（推測。内訳は測っていない）。
- **Base と post-trained の差は小さい。** Q8_0・n_max 4 では、Base が受理率 0.70・0.61x、post-trained が 0.68・0.59x だった。
- **n_max は 2〜4 が良く、8 では受理率 0.55 に落ちて遅くなる。**
- **CPU ドラフトは 0.6B でも 0.14x で、対照として遅い。**
- **本体が Q3_K_S（KV q8_0）の場合、WX 2100 にドラフトを置くと 0.52x。** RX 6400 単独の Q3_K_S（0.76x）より遅い。Q3_K_S と 0.6B を RX 6400 に同居させる構成は未計測で、VRAM の余裕（3519 MiB + 約 500 MiB）が足りるかは不明。

**結論（この構成の範囲）**
- 弱い第 2 GPU（WX 2100）は、ドラフト置き場にしても容量として使っても、RX 6400 単独より遅くなった。
- 速くなるのは、小さいドラフト（0.6B Q4_K_M）を本体と同じ GPU に同居させる構成と、ドラフトモデルを使わない n-gram 方式だった。ただし、n-gram の利得は `ignore_eos` の反復で高めに出ている可能性がある（main1 の注意を参照）。
