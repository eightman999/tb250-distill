# TB250 OpenJev 蒸留・旧GPU Student学習 実装指示書（ユーザー提示 2026-10-06、要約せず要点を保持）

## 目的
BIOSTAR TB250-BTC PRO 上で異種GPUを同時利用した System-One / Jev 型モデルの蒸留実験環境を構築する。
- Teacher: AMD Radeon RX 6400 4GB / Student 1: GT 730 / Student 2: GT 710 / Student 3: GT 430
RX 6400 で OpenJev 互換教師を動かし、各入力に対する候補ごとの logit / probability distribution を生成。
GT 730/710/430 でそれぞれ独立した小型 Student を学習。最終目的は `state + question + candidate → score` を計算する
小型 System-One decision model を旧GPU上で学習・推論可能にすること。

## 0. 基本方針
1. Teacher と Student を同期実行させない。2. RX 6400 は教師データ生成に専念。3. 教師結果は replay buffer / dataset として保存。
4. 各 Student は保存済みデータを独立して読む。5. Student 同士でGPUを共有しない。6. 最初の比較では3GPUに完全に同じモデル。
7. 共通比較完了後、GPUごとにモデルサイズを最適化。8. CUDA依存を極力避け、GT430まで扱えるバックエンドを優先。
9. Studentは autoregressive LLM にしない。10. 最終生成トークンではなく教師の確率分布そのものを蒸留する。

## 1. ハードウェア確認
起動時に CPU, RAM, kernel, 各GPU名, PCI ID, PCIe link width/speed, VRAM, driver, Vulkan/OpenCL/CUDA(可能なら) device 一覧,
温度, power state, GPUクロックを自動取得しログ保存。PCIe x1 は許容。GPU間で weight を頻繁に直接転送する実装は禁止。

## 2. Gate test
RX6400: Vulkan認識 / llama.cpp系から利用可 / 小型GGUFをGPU offload / inference 10分以上安定。
GT730, GT710: OpenCL認識 / CLBlast SGEMM成功 / 連続負荷10分。
GT430（最重要）: OpenCL device 認識 / CLBlastまたは使用可能なOpenCL GEMM / FP32 SGEMM / backwardに必要な基本演算。CUDA不可でも問題としない。
4 GPU 同時負荷で最低10分安定。Gate を通らないGPUがあっても他GPUの作業は止めない。失敗原因とログを保存して先へ進む。

## 3. Teacher
RX 6400、OpenJev互換 scorer、Qwen系 0.6B～4B GGUF、Vulkan。小モデルでパイプライン完成→4B級へ拡大。
context 256 → 512 → 必要なら 1024。`score(context, question, candidates[]) -> {logits[], probabilities[]}`。
候補は A/B/C の文字そのものではなく独立した意味内容。soft probability を必ず保存。argmax のみ保存は禁止。

## 4. Replay buffer
Teacher と Student を切り離す。Student 開始前に最低 10,000 samples を先行生成。50k/100k/500k まで拡張可能に。
SQLite / binary+index / memmap。巨大 JSONL を最終形式にしない。tokenize済み input を保存。Celeron で毎epoch tokenize しない。

## 5. Dataset カテゴリ（目安）
NLI 20%, intent/routing 20%, state→action 20%, ranking/preference 15%, sentiment/priority 10%, ambiguous/uncertainty 10%, agent gate 5%。
continue/stop, answer/search, ask-user/act, remember/discard, retrieve-memory/skip, tool-A/tool-B/no-tool, safe-action/defer, local-model/larger-model を多く。

## 6. Candidate permutation
候補順序をランダム化。Student がラベル文字や position を学習していないことを評価。permutation test を held-out eval に必須。

## 7. Student architecture
テキスト生成モデルにしない。`f(context, question, candidate) -> scalar` を同一 network で各候補に適用し softmax。
最初は小型 GRU（FP32, SGEMM中心, CLBlast, Fermi対応, KV cache不要, decode不要）。

## 8. Common-S
vocab 8192, emb 128, GRU hidden 192, layers 2, context 128, FP32, 約2M params。初期 weight を一度作り3GPUで同じ checkpoint。
seed, sample order, batch size, optimizer, lr, loss, validation dataset も統一。

## 9. Distillation loss
KL(Teacher||Student), temperature KD。T=2.0, KD 0.8, CE 0.2。gold 無しは KD のみ。教師の迷いも蒸留対象。

## 10. Optimizer
Adam/AdamW。重ければ SGD+momentum, Adagrad と比較可。FP16前提にしない。FP32。

## 11. Common experiment
C430/C710/C730 同条件。計測: training loss, val loss, KL, top-1 teacher agreement, gold accuracy, Brier, ECE, samples/s, tokens/s,
step latency, max VRAM, GPU util, wall-clock, 温度, 電力(可能なら), energy/sample(可能なら)。

## 12. GPU optimized Student
GT430 Student-S: vocab 4k-8k, emb 96-128, hidden 128, 2層, ctx 96-128, 1-2M, latency 重視。
GT710 Student-M: vocab 8k, emb 160, hidden 192, 3層, ctx 192, 3-5M。
GT730 Student-L: vocab 8k-16k, emb 256, hidden 256, 3-4層, ctx 256, 8-12M（余裕あれば16M/24M）。

## 13. 基本run
C430, C710, C730 (Common-S 2M), S430 (1-2M), S710 (3-5M), S730 (8-12M)。

## 14. Evaluation
Decision quality: agreement, gold acc, KL, NLL。Calibration: Brier, ECE。
Robustness: candidate permutation, candidate wording paraphrase, context paraphrase, irrelevant context insertion, ambiguous question, unseen candidates。
Confidence: 高entropy sample 抽出、Teacher/Student entropy 比較、「Teacher uncertain / Student confident」を重点調査。

## 15. Cascade
GT430 → (entropy high) GT710 → GT730 → Large model/Teacher。threshold は validation から（例 max prob >= 0.85 で採用）。
accuracy, agreement, average latency, escalation rate, energy per decision を比較。

## 16. プロセス構成
teacher_rx6400, student_gt730, student_gt710, student_gt430, coordinator の5プロセス以上。GPU明示固定。ログ/checkpoint別ディレクトリ。

## 17. Checkpoint
0/10/25/50/75/100%, best-validation。resume可能。weights, optimizer state, step, RNG state, dataset position, config, git commit hash。

## 18. Reproducibility
config.json, hardware.json, environment.txt, metrics.csv, train.log, eval.json, 可能なら git diff。

## 19. 優先順位
Phase 0 4GPU認識・負荷 / 1 Teacher完成・100 sample / 2 CPU Student fwd/bwd / 3 GT730 / 4 GT710 / 5 GT430 /
6 Common-S 3GPU比較 / 7 GPU別最適化 / 8 100k以上へ拡張 / 9 cascade。

## 20. 最初の成功条件
A: RX6400で context/question/3-5 candidates から teacher probability 取得。B: 1000 sample で CPU Student の KL 低下。
C/D/E: GT730/GT710/GT430 で学習可能。F: 全GPUで random baseline を明確に超える teacher agreement。

## 21. 禁止
既存環境を破壊しない。既存GPUドライバを不用意に更新しない。いきなり500k生成しない。最初から大型Studentを作らない。
Teacher待ちでStudentをblockしない。hard labelのみ保存しない。accuracyだけで成功判定しない。3GPUで別々のdataset禁止。
GPU比較時にモデル条件を変えない。問題発生時に他GPUの実験まで停止しない。

## 22/23. 報告
各Phase: Phase/Status/GPU/Backend/成功/失敗/性能/VRAM/samples/s/次。
最終表: GPU | Architecture | Student params | Backend | samples/s | Agreement | KL | ECE | VRAM。
結論: Fermi学習可否, Keplerとの差, params vs quality, 学習速度, inference latency, calibration, cascade効率, 常駐エージェント等の常時decision engineへの利用可能性。

最重要: 大型モデルの判断能力を旧GPU上で高速に動く小型 decision model へ蒸留できるかを実証する。精度より end-to-end pipeline 完成を優先。
