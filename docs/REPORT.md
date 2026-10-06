# TB250 旧GPU System-One 蒸留実験 レポート（2026-10-06 時点）

指示書: [INSTRUCTIONS.md](INSTRUCTIONS.md)／設計: [../DESIGN.md](../DESIGN.md)／変更点ごとの記録: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)

## 1. 結論（要約）

1. **end-to-end パイプラインは成立した。** RX 6400 の Teacher（Qwen3-1.7B, Vulkan）→ SQLite replay → GT 430 / GT 710 / GT 730 / Radeon Pro WX 2100 の Student（手書き GRU、OpenCL + CLBlast、FP32）で、指示書の成功条件 Milestone A〜F をすべて満たした。
2. **Fermi（GT 430）でも Kepler と同品質の Student を学習できた**（CUDA 不要、OpenCL 1.1）。ただし連続学習で 85〜90 ℃に達して温度ガードが頻発し、1 件推論 p50 は約 100 ms。**10/06 17:50 に実験から除外した**（ユーザー判断）。
3. **学習分布内の判断は蒸留できる。** 合成データでは Teacher 一致率 0.80 前後。合成 10 万件で学習すると gold 正解率 0.78 となり、Teacher 自身（0.736）を上回った。
4. **蒸留できているのは判断の意味ではなく表層パターン。** 候補の言い回しが未見になると一致率はほぼランダムに落ちる（合成の候補言い換え 0.35〜0.47、MASSIVE 0.36、random 0.26）。JCommonsenseQA はランダム並み（gold 0.23、random 0.20）。
5. **容量や CPU/GPU 性能より、表現と教師信号が律速。** 8.3 M params の Student も 1.5 M と同水準だった。GPU 間の品質差は run ごとのばらつき（±0.03）の範囲に収まった。学習速度は Celeron 2 コアの CPU 律速だった（WX2100 の GPU 使用率は約 1 %）。
6. **候補意味表現の蒸留（Qwen3-Embedding-0.6B）は、現時点で判断の改善に至っていない。** 損失を修正すると表現自体は改善した（未見言い回しの intent 検索 0.05 → 0.21）。それでも Teacher 空間（0.858）の約 25 % にとどまる。判断と同時に学習する本番比較は、途中で中断した。
7. **Wikipedia LM 事前学習（1.5 M GRU、約 46 M token）は JCQA を改善しなかった。**

## 2. 環境

| 項目 | 内容 |
|---|---|
| マザーボード / CPU / RAM | BIOSTAR TB250-BTC PRO / Celeron G3930（2 コア, AVX 無し）/ 32 GB |
| OS | Debian 13, kernel 6.12.107 |
| Teacher GPU | RX 6400 4 GB（PCIe x16）、RADV Vulkan、llama.cpp b11384 |
| Student GPU | GT 430（Fermi, 1 GB, OpenCL 1.1）、GT 710（Kepler, 2 GB）、GT 730（Kepler, 1 GB）。いずれも PCIe x1、nvidia 390.157 の OpenCL<br>WX 2100（Polaris, 2 GB, x1）は Mesa rusticl の OpenCL 3.0 で途中から追加 |
| 制約 | ドライバ・システムパッケージは無変更<br>NVIDIA 390 の Vulkan ICD は vulkaninfo / llama.cpp をクラッシュさせるため、`VK_ICD_FILENAMES=radeon_icd.json` を必須にした<br>1 プロセス 1 GPU（同一プロセスで 2 枚目を開くと INVALID_DEVICE） |

## 3. 実装したもの

- **Teacher scorer**: Qwen3 chat 形式で候補に番号を振り、次トークンの番号 logprob を K 候補で再正規化する。候補順を恒等と逆順の 2 通りで採点して平均し、位置バイアスを打ち消す。soft 分布（logits/probs）と生 logprob を必ず保存する。
- **Replay**: SQLite（WAL）に保存する。Teacher と Student は非同期で動く。tokenize 済み npz shard を作り、Celeron で毎 epoch tokenize しない。
- **Student**: `f(context, question, candidate) → score`。prefix（question+context）を GRU で 1 回読み、候補は prefix の状態から継続して読む。候補位置の情報は入らないため、構造的に permutation 等変になっている。fwd/bwd は手書きで、numpy と pyopencl+CLBlast の 2 backend。数値勾配チェックと np/cl の一致（誤差 1e-7 級）を確認した。
- **学習**: KD（T=2）、gold ありは 0.8KD+0.2CE。AdamW、grad clip、温度ガード（88 ℃停止、78 ℃再開）。checkpoint は 0/10/25/50/75/100 %、best、last を保存し、resume できる。
- **計測**: run ごとに config / hardware / environment / metrics.csv / eval.json を出す。amdgpu は sysfs で VRAM・温度・電力（J/sample 実測）を取る。
- **評価**: agreement、KL、NLL、gold acc、Brier、ECE、robust 6 種、source 別の指標、推論 latency。
- **coordinator**: GPU ごとに別プロセスで起動・監視・評価する。cascade 合成と report 集計のコードもあるが、cascade は未実行。

## 4. Phase 0: Gate test

| GPU | GEMM | SGEMM GFLOPS | 10 分連続負荷 | 温度 |
|---|---|---:|---|---|
| GT 730 | CLBlast | 198 | PASS（10 回ビット一致） | 最大 57 ℃ |
| GT 710 | CLBlast | 99 | PASS | 最大 47 ℃ |
| GT 430 | CLBlast（OpenCL 1.1） | 37 | PASS | 10 分で 92 ℃、延長すると 13 分で 96 ℃（試験を停止） |
| WX 2100 | CLBlast（rusticl） | **342** | 正当性 PASS | — |
| RX 6400 | llama.cpp Vulkan | — | Teacher 採点を連続実行し安定 | 最大 66 ℃ |

逆伝播に必要な基本演算 13 種（atomic_cmpxchg による scatter-add を含む）は、全 GPU で numpy と一致した。RX 6400 を含む 4 GPU 同時 10 分の負荷試験も PASS した。

## 5. Teacher（Qwen3-1.7B Q4_K_M, RX 6400）

- 速度は 3.6〜4.0 samples/s（平均約 270 ms）、VRAM 1.15 GB、約 60 ℃。律速は Python/HTTP で、GPU 使用率は低い。
- 採点件数は合計 166,738 件、エラー 0。内訳は合成 103,800（train 10 万、val/test 各 1,000、robust 1,800）と公開 62,938。
- 分布は極端に尖っている（max prob > 0.99 が 75 %）。位置バイアスが強く、恒等順と逆順で argmax が一致するのは約 65 % だけ。
- source 別の Teacher gold 正解率（test）:

| source | 正解率 | random |
|---|---:|---:|
| MASSIVE | 0.79 | 0.26 |
| JCQA | 0.77 | 0.20 |
| 合成 | 0.74 | 0.35 |
| RouteLLM | 0.55 | 0.50 |
| When2Call | 0.51 | 約 0.43 |
| WRIME | 0.51 | 0.33 |
| JNLI | **0.32** | 0.42 |

JNLI では高確信のまま誤答しており、ランダムを下回る。

## 6. 合成データでの Student 結果

### 6.1 Common-S 3 GPU 比較（合成 train 1 万件、8 epoch、同一 init・seed、1.49 M params）

| GPU | samples/s | wall | best val KL | test agree | test KL | gold | ECE | cand 言い換え | 未見候補 | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GT 430 | 51 | 27.7 分 | 0.589 | 0.761 | 0.651 | 0.714 | 0.171 | 0.400 | 0.457 | 100.7 ms |
| GT 710 | 98 | 14.7 分 | 0.614 | 0.773 | 0.678 | 0.725 | 0.172 | 0.420 | 0.453 | 29.1 ms |
| GT 730 | 165 | 8.4 分 | 0.632 | 0.739 | 0.692 | 0.695 | 0.144 | 0.393 | 0.407 | 19.2 ms |

- random は約 0.35、Teacher の gold は 0.736。
- 70 step 時点では 3 GPU の値が小数 4 桁まで一致した。終盤は float atomic の非決定性で ±0.04 ばらついた。

### 6.2 GPU 別最適化 Student（合成 1 万件）

| run | params | 構成 | test agree | test KL | gold | 備考 |
|---|---:|---|---:|---:|---:|---|
| S430 | 1.26 M | emb128 / H128×2 | 0.777 | 0.691 | 0.737 | 最終 weight |
| S710 | 3.35 M | emb256 / H256×3, Lp192 | 0.788 | 0.745 | 0.743 | 最終 weight |
| S730 | 8.27 M | emb256 / H512×4, Lp256 | 0.773 | 0.654 | 0.728 | best ckpt |

- S710 と S730 は、指示書の寸法では目標 params に届かなかったため寸法を拡大した。
- S730 は batch 32 で VRAM 不足となり（CLBlast の一時バッファ確保に失敗）、batch 16 にした。lr 2e-3 では学習が進まず（4 epoch で agree 0.60）、lr 1e-3 で完走させた（約 2 時間、p50 182 ms）。
- **params を 5.5 倍にしても品質はほぼ不変だった。**

### 6.3 Phase 8: 合成 10 万件（Common-S、2 epoch）

| GPU | test agree | test KL | gold | ECE | cand 言い換え | 未見候補 |
|---|---:|---:|---:|---:|---:|---:|
| GT 430 | 0.803 | 0.438 | 0.759 | 0.136 | 0.373 | 0.417 |
| GT 710 | 0.805 | 0.457 | **0.780** | 0.125 | 0.350 | 0.397 |

分布内の指標は大きく改善し、gold は Teacher を超えた。一方で、候補の言い換えと未見候補は 1 万件のとき（0.42 / 0.45 前後）より**悪化**した。暗記が進んだと考えられる。

## 7. 公開データと Baseline A

### 7.1 導入データ（[RELEASE_POLICY.md](RELEASE_POLICY.md)）

| データ | ライセンス | 公開版モデル |
|---|---|---|
| JCommonsenseQA | CC BY-SA 4.0 | 採用（ユーザー承認） |
| MASSIVE（ja/en） | CC BY 4.0 | 採用 |
| When2Call | CC BY 4.0 | 採用 |
| JNLI | CC BY-SA 4.0 | 未承認 |
| WRIME | CC BY-NC-ND 4.0 | 除外 |
| RouteLLM | Apache-2.0、prompt は LMSYS 由来 | 除外 |

MARC-ja は配布終了のため取り込んでいない。replay DB・shard・teacher 出力は公開しない。

### 7.2 Baseline A（固定。詳細は [BASELINE_A.md](BASELINE_A.md)）

- データは `pubA`: 合成 3 万 + JCQA 8,938 + MASSIVE 12,000 + When2Call 10,000、Lp256。
- 学習条件: Common-S、3 epoch、best val KL の checkpoint で評価。

| GPU | Arch | samples/s | J/sample | best val KL | MASSIVE agree | JCQA gold | 合成 gold | W2C gold | p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GT 430 | Fermi | 38.6 | n/a | 0.909 | 0.370 | 0.220 | 0.741 | 0.492 | 100.5 ms |
| GT 710 | Kepler | 69.4 | n/a | 0.941 | 0.358 | 0.234 | 0.716 | 0.510 | 25.9 ms |
| GT 730 | Kepler | 109.3 | n/a | 0.913 | 0.394 | 0.234 | 0.685 | 0.492 | 28.1 ms |
| WX 2100 | Polaris | 83.0 | **0.22** | 0.915 | 0.332 | 0.212 | 0.748 | 0.504 | 41.7 ms |

- **MASSIVE の Teacher 一致率は mean 0.363 / sd 0.026（random 0.262）。** これを以降の比較基準とした。
- When2Call は Teacher 一致率 0.85 と高いが、gold は約 0.50（Teacher 自身も 0.51）。

## 8. 候補意味表現の蒸留

| ID | 内容 | 結果 |
|---|---|---|
| S06 | Qwen3-Embedding-0.6B（PCA128, cosine, λ=0.5）、判断と同時学習 | MASSIVE 0.388 / 0.336（mean 0.362）。**改善なし** |
| S17 | Qwen3-1.7B（mean pooling） | MASSIVE 0.310（WX2100）。改善なし |
| D1 | 診断（[DIAG_SEM.md](DIAG_SEM.md)） | **全 run「近くない」** |
| SF | 損失の修正とスクリーニング（[SEMFIX.md](SEMFIX.md)） | 未見 intent 検索 0.142 → **0.21**（Teacher 0.858） |
| SP1/SP2 | 修正した損失で本番比較 | **ユーザー指示で中断**（last ckpt を保存済み） |

### 8.1 診断（D1）

- 学習した文字列では意味の判別力が付く（specificity 0.43）。未見の言い回しでは 0.09 で、学習なしの run と同等だった。
- 未見の言い回しで同じ intent を引き当てる割合は 0.04〜0.05。近傍は意味ではなく定型ラッパー語で決まっていた。
- Qwen3-1.7B の mean pooling は target として不適だった。Teacher 空間でも intent 検索が 0.229 しかなく、同じリクエスト内の他文字列に値が左右される。
- When2Call の候補は 99 % が Lc=16 を超えて切れていた。そのため意味表現 loss から除外した（`pubA_notrunc`）。

### 8.2 損失の修正（SF、意味表現だけを学習、MASSIVE val）

| 条件 | 未見 intent 検索 | 既見 intent 検索 |
|---|---:|---:|
| 判断バッチ内・cos（従来相当） | 0.142 | 0.79 |
| 独立ミニバッチ・cos | 0.188 / 0.163 | 0.79 |
| 類似度行列蒸留（rkd、PCA128） | **0.209 / 0.205** | 0.80 |
| InfoNCE + rkd | 0.197 / 0.197 | 0.70 |
| InfoNCE 単独 | 0.117 | 0.34 |

- 独立ミニバッチは効果があった。損失の種類による差は seed のばらつきと同程度だった。
- 既見 0.8 → 未見 0.2 の落差が本質的な問題。MASSIVE の train 側の言い回しは 240 種類しかなく、多様性不足が律速していると推測する。
- 注意: MASSIVE の val と test は候補文字列がほぼ同一（239/240 が共通）。そのため、表現指標での val 選択は独立した確認にならない。

## 9. Wikipedia 事前学習（独立 branch、WX2100）

- jawiki 86k 記事、約 46 M token、1 epoch。112 分で val ppl は 8205 → 50.6（GPU 22.3 W、0.207 J/sample）。
- 事前学習後に Baseline A と同条件で蒸留した。JCQA gold 0.228（Baseline A 平均 0.225）、MASSIVE 0.348、合成 agree 0.647（悪化）、候補言い換え 0.377（悪化）。**改善なし。**

## 10. 指示書の成功条件

| Milestone | 判定 | 根拠 |
|---|---|---|
| A: RX6400 で teacher probability | 達成 | 166,738 件、エラー 0 |
| B: 1000 件で CPU Student の KL 低下 | 達成 | val KL 1.007 → 0.887 |
| C/D/E: GT730 / GT710 / GT430 で学習 | 達成 | 3 枚とも同一の学習曲線 |
| F: random を明確に超える agreement | 達成 | 合成 0.74〜0.81（random 0.35） |

## 11. 指示書 §23 の結論項目

1. **Fermi で学習できたか**: できた。品質は Kepler と同等。ただし熱（最大 90 ℃、温度ガードが頻発）と速度（Kepler の 1/2〜1/3）の面で実運用には不向き。
2. **Kepler との差**: 学習速度は GT730 : GT710 : GT430 ≒ 2.8 : 1.8 : 1（Baseline A）。品質差はノイズの範囲。
3. **params と quality**: 1.3 M〜8.3 M でほぼ横ばい。データ量（1 万 → 10 万）は分布内の品質を大きく上げたが、汎化は上げなかった。
4. **学習速度**: Celeron 2 コアの CPU 律速。i7-6700 への交換を推奨する（週末予定）。
5. **推論 latency（batch 1, p50）**: GT730 19〜28 ms、GT710 26〜29 ms、WX2100 42 ms、GT430 約 100 ms、S730 182 ms。
6. **calibration**: Student の ECE は 0.13〜0.20。gold に対する NLL・Brier は Teacher より良い（Teacher の自信過剰を KD が緩和）。
7. **cascade**: **未実施**。GT430 が最も遅く、1 段目に置く前提が崩れた。順序と構成の再設計が必要。
8. **常駐エージェント等の常時 decision engine への適用**: 候補が固定で分布が既知の判断（続行/停止、検索/回答など）は、Kepler で 20〜30 ms・一致率約 0.8 で実用圏。候補が自由に変わる判断や知識が要る判断は、現状不可。

## 12. 未実施・残件・既知の問題

- 意味表現蒸留の本番比較（SP1/SP2）は中断中。21:38 の停止後、サブエージェントが誤って 21:48 に再開し、21:49 に再停止した（GT730 rkd_pca は step 1417、GT710 infonce_rkd は step 750、last ckpt 保存済み。cl の非決定性により再開の継ぎ目はビット一致しない）。残り 2 本は未開始。再開は tb250 で `RESUME=1 scripts/launch_semfix_prod_resume.sh`。
- 大規模な一般テキストでの埋め込み蒸留（語彙・言い回しの多様性を補う案）は未着手。使うコーパス（pubA の文脈 / Wikipedia / その他）がユーザー判断待ち。
- 判断経路への semantic score 接続、文脈側の意味表現蒸留、言い換えの水増しは、手順上まだ実施条件を満たしていない。
- Teacher の強化（4B 級モデル、JNLI の扱い）は未実施。
- cascade（Phase 9）は未実施。
- GeForce（390）は電力を取得できず、J/sample は WX2100 と RX6400 のみ。
- OpenCL の float atomic により、run は bit 再現しない（±0.03）。差の判定には複数 run が必要。
- git commit は未作成。各 run の commit hash は `uncommitted-<内容 hash>`。

## 13. 成果物の場所

- コード: `~/dev/sandbox/tb250-distill`（Mac 正本）→ `tb250:~/tb250-distill`
- データ（tb250）:
  - `data/replay.sqlite`（バックアップ: `/mnt/hdd/replay-*.sqlite`）
  - `data/tok/`、`data/emb/`、`data/emb_eval/`（評価専用）、`data/external/`（MANIFEST）
- run（tb250 の `runs/`）:

| ディレクトリ | 内容 |
|---|---|
| `hw/` | Gate test |
| `common/` | 合成 Common-S |
| `optimized/` | S430 / S710 / S730 |
| `phase8/` | 合成 10 万件 |
| `pub/` | 公開データ途中版 |
| `baseA/` | Baseline A |
| `sem06/`, `sem17/` | 候補意味表現蒸留 |
| `diag_sem/` | 診断 |
| `semfix/` | 損失修正（screen/prod） |
| `wikiA/` | Wikipedia branch |

- 主な docs: [BASELINE_A.md](BASELINE_A.md)、[DIAG_SEM.md](DIAG_SEM.md)、[SEMFIX.md](SEMFIX.md)、[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)、[RELEASE_POLICY.md](RELEASE_POLICY.md)
