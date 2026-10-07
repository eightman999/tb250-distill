# tb250-distill

**大型 LLM の判断能力を、旧 GPU 上で学習・推論できる小型の判断モデル（System-One decision model）へ蒸留する実験基盤です。**

*Distilling an LLM teacher's decision distributions into tiny GRU scorers trained on old GPUs (GeForce GT 430 / GT 710 / GT 730, Radeon Pro WX 2100) with hand-written backprop on OpenCL + CLBlast.*

```
RX 6400 Teacher (Qwen3 GGUF, llama.cpp Vulkan)
        │  候補ごとの soft 分布（logits / probs）
        ▼
   SQLite replay buffer ──▶ tokenized shard
        │
        ├──▶ Student on GT 730  (OpenCL 1.2 + CLBlast)
        ├──▶ Student on GT 710  (OpenCL 1.2 + CLBlast)
        ├──▶ Student on WX 2100 (Mesa rusticl OpenCL 3.0 + CLBlast)
        └──▶ Student on GT 430  (Fermi, OpenCL 1.1)  ※途中で除外
```

モデルが計算するのは `f(state/context, question, candidate) → score` で、K 個の候補の score を softmax して判断分布にします。Student はテキストを生成しない（autoregressive ではない）ので、KV cache や decode は要りません。

## 特徴

- **Teacher と Student を切り離した構成**: Teacher の採点結果は replay buffer に保存し、各 Student は別プロセス・別 GPU でそれを独立に読む。Teacher の採点を待たずに Student を学習できる。
- **soft 分布の蒸留**: argmax（正解ラベル）だけでなく、Teacher の「迷い」まで KL で蒸留する（温度 T=2）。gold がある問題は KD 0.8 + CE 0.2 で学習する。
- **CUDA 非依存**: GRU の forward/backward を手書きし、numpy と pyopencl + CLBlast の 2 backend で動かす。GEMM は CLBlast、gate・損失・AdamW・埋め込みの scatter-add は OpenCL 1.1 互換のカーネルで実装した。Fermi（GT 430）でも動く。
- **正当性の検証**: 数値勾配チェック（float64）と、numpy / OpenCL の一致テストを行う（誤差 1e-7 級）。
- **位置バイアス対策**: Teacher は候補の並び順を変えて複数回採点し、平均する。Student は候補を 1 つずつ独立に採点するので、候補の順序に構造上依存しない。
- **計測**:
  - run ごとに config / hardware / environment / metrics.csv / eval.json を出力する。
  - 温度ガード（88 ℃で学習を止め、78 ℃で再開）。
  - amdgpu は sysfs から電力を取り、J/sample を実測する。
- **評価**:
  - Teacher との一致率、KL、gold 正解率、NLL、Brier、ECE、データセット別の指標を出す。
  - robust 評価として、候補の並び替え、候補の言い換え、文脈の言い換え、無関係な文の挿入、質問の曖昧化、未見の候補の 6 種を測る。

## 主な結果

詳細は [docs/REPORT.md](docs/REPORT.md) を参照してください。

- end-to-end のパイプラインが成立した。Teacher の採点は 16.7 万件（エラー 0）。4 GPU 同時の 10 分負荷試験も PASS した。
- Fermi（GT 430）でも Kepler と同等の品質で学習できた。ただし熱（最大 90 ℃）と速度の面で実運用には向かない。
- 学習データと同じ分布の判断は蒸留できる。合成データでは Teacher 一致率が約 0.80。10 万件で学習すると gold 正解率 0.78 となり、Teacher 自身（0.736）を上回った。
- **未見の言い回しへの汎化はできていない。** 候補の文言が変わると一致率はほぼランダムまで落ちる（MASSIVE 0.36、ランダム 0.26）。モデルを 8.3 M params に増やしても、Wikipedia で事前学習しても改善しなかった。Teacher の文埋め込みを蒸留する試みは継続中。
- 1 件あたりの推論時間（p50）: GT 730 は約 20〜28 ms、GT 710 は約 26〜29 ms、WX 2100 は 42 ms、GT 430 は約 100 ms。
- 学習速度は、Celeron G3930（2 コア）の CPU が律速だった。

各段階の差分は [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)、基準値は [docs/BASELINE_A.md](docs/BASELINE_A.md) にあります。

## 動作環境

検証済みの環境は以下です。

| 項目 | 内容 |
|---|---|
| OS | Debian 13（kernel 6.12） |
| Python | 3.13 |
| Teacher | RX 6400 4 GB、llama.cpp（Vulkan ビルド） |
| Student | NVIDIA 390 系ドライバの OpenCL（GT 430 / GT 710 / GT 730）、Mesa rusticl（WX 2100） |

Python パッケージ: `numpy`, `pyopencl`, `pyclblast`（システムの CLBlast が必要）, `sentencepiece`, `requests`, `pyarrow`（Wikipedia の前処理のみ）, `pytest`。

注意:

- NVIDIA 390 の Vulkan ICD と RADV が共存する環境では、Vulkan を使うプロセスに `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` を指定する（指定しないと vulkaninfo / llama.cpp が落ちる）。
- WX 2100 などの Mesa rusticl デバイスには `RUSTICL_ENABLE=radeonsi` を指定する。
- 1 プロセスで扱う GPU は 1 枚にする。OpenCL デバイスは名前の部分一致（例 `"GT 710"`）で選ぶ。

## 使い方

```bash
# 0. ハードウェア情報と Gate test
python -m tb250distill.hw.collect --out runs/hw
python -m tb250distill.hw.gate_cl --device "GT 710" --out runs/hw --minutes 10

# 1. データ: 合成データと公開データ（任意）を replay DB へ
python -m tb250distill.data.synth --db data/replay.sqlite
python -m tb250distill.data.fetch_public --root data/external
python -m tb250distill.data.public build --root data/external --db data/replay.sqlite

# 2. Teacher 採点（RX 6400、llama-server を自動起動。中断しても続きから再開できる）
python -m tb250distill.teacher.produce --db data/replay.sqlite --splits val,test,train,robust --start-server

# 3. トークン化
python -m tb250distill.tokenize_data --db data/replay.sqlite --name pubA --vocab 8192 --lp 256 \
    --sources synth,jcqa,massive,when2call --max-per-source synth=30000 --retrain \
    --splits train,val,test,robust

# 4. 学習（GPU ごとに別プロセス。全 GPU で同じ初期重みを使う）
python -m tb250distill.coordinator init --config common_s --seed 0 --out runs/common/init.npz
python -m tb250distill.coordinator launch --plan common --data data/tok/pubA --epochs 3 \
    --backend cl --gpus "GT 710,GT 730" --extra "--lp 256"
python -m tb250distill.coordinator status --plan common

# 5. 評価（best checkpoint）
python -m tb250distill.student.evaluate --backend cl --device "GT 730" --data data/tok/pubA --lp 256 \
    --ckpt runs/common/gt730/ckpt/best.npz --splits val test robust --out eval.json
```

単体の学習は `python -m tb250distill.student.train --help` を参照してください（numpy backend `--backend np` なら GPU なしで動きます）。

- llama.cpp の異種 GPU 投機的デコード（RX 6400 に本体、WX 2100 にドラフト）のベンチ: [docs/SPECBENCH.md](docs/SPECBENCH.md)。
  `python -m tb250distill.specbench.run --plan main --out runs/specbench/<名前>`、集計は `python -m tb250distill.specbench.report <out>`。

- 候補の意味表現の蒸留: `--sem-cand-weight` / `--sem-emb` / `--sem-loss` / `--sem-batch` を指定する。Teacher の埋め込みは `tb250distill.teacher.embed` で抽出する。
- Wikipedia での LM 事前学習: `tb250distill.student.pretrain_lm`。

テスト:

```bash
python -m pytest -q
# OpenCL 一致テスト（実機）
TB250_CL_DEVICE="GT 710" python -m pytest -q tests/test_student_cl.py
```

## リポジトリ構成

```
tb250distill/
  hw/        ハードウェア収集、OpenCL Gate test、amdgpu テレメトリ
  data/      合成データ、公開データ取得・変換、Wikipedia 前処理
  teacher/   llama-server 管理、候補採点、replay への生産、埋め込み抽出
  specbench/ 異種 GPU 投機的デコードのベンチ（run / report / configs / prompts）
  student/   numpy / OpenCL backend、GRU モデル、学習・評価、意味表現蒸留、LM 事前学習
  replay.py  tokenize_data.py  coordinator.py  cascade.py  report.py  diag_sem.py
tests/       単体テスト（勾配チェック、np/cl 一致、データ変換など）
scripts/     同期・実験起動用スクリプト
docs/        レポート、実験ログ、診断、データ方針
```

## データとライセンス

- コードとドキュメントは [Apache License 2.0](LICENSE) です。
- 学習データ・Teacher の出力・学習済みモデルは、このリポジトリに含まれていません。replay DB、shard、外部データは `data/`、`runs/` に生成され、git 管理外です。
- 公開データセットはそれぞれのライセンスに従います。MASSIVE / When2Call は CC BY 4.0、JGLUE（JCommonsenseQA / JNLI）は CC BY-SA 4.0、WRIME は CC BY-NC-ND 4.0、RouteLLM データは Apache-2.0（元の prompt の条件に注意）、Wikipedia は CC BY-SA 4.0 です。
- 派生データや学習済みモデルを公開する場合の扱いは [docs/RELEASE_POLICY.md](docs/RELEASE_POLICY.md) を参照してください。
- Teacher モデル（Qwen3-1.7B、Qwen3-Embedding-0.6B）は Apache-2.0 です。
