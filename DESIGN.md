# TB250 System-One 蒸留 設計（実装の共通契約）

元の指示書: docs/INSTRUCTIONS.md。ここは各モジュール間の契約だけを定める。

## 実機（2026-10-06 確認済み）

- ホスト: tb250（Debian 13, kernel 6.12.107）。Macからは `ssh -o BatchMode=yes -o HostName=<tb250 の Tailscale IP> tb250`（IP は git 管理外の `scripts/host.local` に `TB250_HOSTNAME=...` で置く）（`tb250.local` は解決不可）。
  - **別の学習機（llm_master）には絶対に接続しない。**
- CPU Celeron G3930 2コア / RAM 31GiB / SSD空き75GB、`/mnt/hdd` 空き227GB。
- GPU（PCI / link / driver）:
  - 03:00.0 RX 6400 4GB, x16 Gen4, amdgpu。Vulkan(RADV) / OpenCL(Clover 1.1)
  - 04:00.0 GT 430 (GF108 Fermi) 1GB, x1 2.5GT/s, nvidia 390.157, OpenCL 1.1
  - 07:00.0 GT 730 (GK208B) 1GB, x1, nvidia 390.157, OpenCL 1.2
  - 08:00.0 GT 710 (GK208B) 2GB, x1, nvidia 390.157, OpenCL 1.2
  - 06:00.0 Radeon Pro WX 2100（今回未使用）、Intel HD610（未使用）
- **ドライバ・カーネルモジュール・システムパッケージを変更しない**（apt install/remove, modprobe, driver-swap 禁止）。
- NVIDIA 390 の Vulkan ICD は vulkaninfo / llama.cpp を stack smashing で落とす。Vulkanを使う全プロセスで
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` を必須とする。このとき Vulkan0=RX6400, Vulkan1=WX2100。
- llama.cpp: `~/bench/llama-bin/llama-b11384/`（Vulkanビルド, `LD_LIBRARY_PATH` に同ディレクトリ）。`llama-server` あり。
- 既存モデル: `~/bench/models/Qwen3-1.7B-Q4_K_M.gguf`（Teacher初期）。`~/bench` は既存資産、書き換え禁止（読むだけ）。
- OpenCLデバイスの順番は不定（現状 GT730, GT430, GT710）。**必ずデバイス名の部分一致で選ぶ**（"GT 430" 等）。
- GeForce + 390 では nvidia-smi の utilization/power は N/A。温度・VRAM使用量・クロックは取得可。

- **2026-10-06 以降 GT430 は実験から除外**（ユーザー指示）。Student 用 GPU は GT710 / GT730 / WX2100（WX2100 は `RUSTICL_ENABLE=radeonsi` 必須）。

## 配置

- ローカル正本: `~/dev/sandbox/tb250-distill`（Mac, git）。
- リモート: `tb250:~/tb250-distill`。同期は Mac → tb250 の一方向:
  `rsync -a --exclude .git --exclude .venv --exclude runs --exclude data --exclude __pycache__ ./ tb250:~/tb250-distill/`
  （`scripts/sync.sh` 経由）。`--delete` は使わない。
- リモート venv: `~/tb250-distill/.venv`（numpy 2.5, pyopencl 2026.1, pyclblast, sentencepiece, requests 導入済み）。
  追加パッケージは venv 内 pip のみ可。Mac ではローカル `.venv` を作ってよい（numpy, sentencepiece, pytest）。
- 生成物（リモートのみ）: `~/tb250-distill/data/`（replay DB, tokenizer, tokenized shards）、`~/tb250-distill/runs/`。

```
runs/
  hw/                 Phase 0 hardware.json, gate結果
  teacher/            teacher log, config
  common/{gt430,gt710,gt730}/
  optimized/{gt430,gt710,gt730}/
  cpu/                Phase 2
```

## パッケージ構成（Python, `tb250distill/`）

| モジュール | 役割 |
|---|---|
| `hw/collect.py` | hardware.json / environment.txt 収集（CPU, RAM, kernel, PCI ID, link, VRAM, driver, Vulkan/OpenCL/CUDA 一覧, 温度, clock, pstate） |
| `hw/gate_cl.py` | OpenCL GEMM/逆伝播基本演算の正当性 + 連続負荷（`--device "GT 430" --minutes 10`） |
| `hw/gate_teacher.sh` | RX6400 llama-server 連続推論 |
| `data/synth.py` | カテゴリ別の raw item 生成（日本語/英語混在） |
| `replay.py` | SQLite replay buffer（下記スキーマ） |
| `teacher/server.py` | llama-server の起動・停止・ヘルスチェック |
| `teacher/scorer.py` | `score(context, question, candidates) -> {logits, probs}` |
| `teacher/produce.py` | raw item → teacher → replay DB（再開可能、Student を待たない） |
| `tokenize_data.py` | sentencepiece 学習 + tokenized shard 書き出し |
| `student/backend_np.py` / `backend_cl.py` | 同一 op インターフェース（numpy / pyopencl + CLBlast） |
| `student/model.py` | GRU scorer（手書き forward/backward） |
| `student/train.py` | KD 学習・checkpoint・metrics |
| `student/evaluate.py` | 評価・robustness・calibration |
| `coordinator.py` | プロセス起動・監視（teacher/students を別プロセスで起動） |

## Replay DB 契約（SQLite, `data/replay.sqlite`, WAL モード）

```sql
CREATE TABLE items(              -- teacher 前の raw item
  item_id INTEGER PRIMARY KEY,
  split TEXT NOT NULL,           -- train | val | test | robust
  category TEXT NOT NULL,        -- nli | intent | state_action | ranking | sentiment | ambiguous | agent_gate | commonsense | routing
  lang TEXT NOT NULL,            -- ja | en
  context TEXT NOT NULL,
  question TEXT NOT NULL,
  candidates TEXT NOT NULL,      -- JSON list[str]（意味内容を持つ文字列。A/B/C 等のラベル文字ではない）
  gold INTEGER,                  -- candidates の index、無ければ NULL
  variant_of INTEGER,            -- robust 用: 元 item_id
  variant TEXT,                  -- NULL | perm | cand_paraphrase | ctx_paraphrase | irrelevant_ctx | ambiguous | unseen_cand
  gen_seed INTEGER NOT NULL,
  source TEXT NOT NULL DEFAULT 'synth',  -- synth | jcqa | jnli | massive | wrime | when2call | routellm（公開データ導入で追加。ALTER TABLE 済み）
  extra TEXT                     -- JSON: 元 ID・元ラベル・soft 分布など（synth は NULL）
);
CREATE TABLE teacher(            -- teacher 出力（soft 分布必須）
  item_id INTEGER PRIMARY KEY REFERENCES items(item_id),
  teacher_model TEXT NOT NULL,   -- gguf ファイル名
  method TEXT NOT NULL,          -- 例 "label_logprob_perm2"
  logits TEXT NOT NULL,          -- JSON list[float]、candidates と同じ順序
  probs TEXT NOT NULL,           -- JSON list[float]、softmax(logits)
  raw TEXT,                      -- JSON: permutation ごとの生 logprob 等
  latency_ms REAL,
  created_at TEXT NOT NULL
);
```

- 公開データ（`tb250distill/data/public.py`）: 取得は `data/fetch_public.py`（`data/external/<name>/` + `MANIFEST.json`）、
  変換・insert は `python -m tb250distill.data.public build`。val/test の候補文言は train と別の言い回し集合。
  `python -m tb250distill.data.public stats` で source×split ごとの Teacher gold 一致率を出す。
  produce の段階指定は `split[:上限][@source]`（`public` = synth 以外。例 `val@public,test@public,train@public,val,test,train`）。
- `items` は生成時に一括 insert（決定的 seed）。`teacher` は produce.py が未処理 item を順に埋める。Student は `teacher` に行がある item だけ読む。
- argmax だけの保存は禁止。`logits` と `probs` を必ず保存。

## Tokenized shard 契約（`data/tok/<tok_name>/`）

- `spm.model`（sentencepiece unigram, byte_fallback, vocab = 8192 等）。特殊 id: 0=pad, 1=unk, 2=bos, 3=eos, 4=`<sep>`。
- shard は item_id 集合ごとに `<split>.npz`:
  - `item_id` int64[N]
  - `prefix` int32[N, Lp]（`question <sep> context` を tokenize、context は**末尾側を残す**よう左を切る。右 pad 0）、`prefix_len` int32[N]
  - `cand` int32[N, Kmax, Lc]（Lc=16 既定）、`cand_len` int32[N, Kmax]、`k` int32[N]
  - `t_logits` float32[N, Kmax]（pad は -inf ではなく 0、`k` でマスク）、`gold` int32[N]（無しは -1）
- optional キー `source`（文字列配列[N]）: 公開データ導入後の shard に入る。`Shard.extra["source"]`、`evaluate.py` の `by_source` 出力で source 別評価。
  無い旧 shard も有効（挙動不変）。spm は `--corpus-max-per-source` で source ごとにコーパスを間引ける。
- Lp = モデルの context 長（Common-S は 128）。shard は Lp ごとに作る（`<split>_L128.npz`）。
- Celeron で毎 epoch tokenize しない。shard 作成は teacher 追加分に対して増分または再作成で良い。

## Student 契約

- `f(context, question, candidate) -> scalar`。prefix（question+sep+context）を GRU で一度読み、各 candidate token 列を
  prefix の最終 hidden（全層）から継続して読み、最終 hidden → head → scalar。K 候補を softmax。
  候補位置の情報はモデルに入らない（構造的に permutation 等変）。
- Common-S: vocab 8192, emb 128, GRU hidden 192 × 2 層, Lp 128, Lc 16, FP32。head = Linear(H,H)+tanh+Linear(H,1)。
- 損失: `T=2.0`。KD = `T^2 * KL(softmax(t/T) || softmax(s/T))`。gold あり: `0.8*KD + 0.2*CE(s, gold)`、gold 無し: `KD`。
- Optimizer AdamW（lr 2e-3, wd 0.01, betas 0.9/0.999）, grad clip 1.0。
- 同一 init checkpoint・seed・sample order・batch size を全 GPU で使用（order は seed から決定、デバイス非依存）。
- checkpoint（npz + json）: weights, optimizer state, step, RNG state, dataset position, config, git commit hash。
  0/10/25/50/75/100% と best-val を保存。`--resume` 可能。
- 各 run dir: `config.json hardware.json environment.txt metrics.csv train.log eval.json`（+ `git.diff`）。

## Candidate Semantic Distillation（候補文の意味表現の蒸留）

目的: 候補を表面文字列で照合する Student が、学習時と言い回しの違う候補（MASSIVE test のラベル言い回し）に汎化するよう、
Teacher 側の文 embedding を候補の GRU 表現へ蒸留する。

- embedding 取得: `teacher/embed.py`（provider 抽象 `llama_server` + 名前付きプリセット `qwen3_1p7b`（mean pooling）/ `qwen3_emb_0p6b`（last pooling））。
  対象は **train shard の候補文字列の unique 集合だけ**（val/test/robust は読まない・embedding しない・PCA に使わない。DB 側 split が train 以外なら LeakError）。
  出力 `data/emb/<provider>/<shard_name>/`: `strings.json emb_raw.npy pca.npz emb_pca.npy cand_idx_train.npy meta.json`。
  PCA = L2 正規化 → train unique 集合の平均を引く → 上位 d 主成分（既定 d=128、whiten なし、PCA 後は再正規化しない）。
  `cand_idx_train.npy[N,Kmax]` は shard の train 行順 × 候補位置 -> unique index（候補無しは -1）。meta の `item_id_sha1` で shard との対応を検証する。
- Student（`--sem-cand-weight λ --sem-emb DIR [--sem-dim d]`）: 候補を文脈なし（全層の初期 hidden=0）で同じ GRU（重み共有）に通した最終層の最終 hidden を
  `Linear(H, d)`（`sem.wp/sem.bp`、`Config.sem_dim>0` のときだけ param に追加。seed 決定的な別乱数 stream で初期化）で z に写し、
  `L_total = L_KD/CE + λ * mean_valid(1 - cos(z, t))`（t = PCA 済み embedding、train バッチ内の有効候補で平均）。
  追加 pass は学習時のみ（判断用 score の経路・推論は不変）。λ=0 / 未指定は追加 forward を行わず従来とビット一致（np backend）。
  `--sem-ctx-weight`（Context Semantic Distillation: prefix 最終 hidden -> context embedding）は予約のみ・未実装（0 以外はエラー）。
  metrics.csv 末尾に `sem_loss` 列（λ 倍前の値）。head は学習専用（推論の parameter 数 = total - sem_head）。
- `--exclude-truncated`（`embed extract --exclude-truncated`）: spm token 長 > Lc の候補文字列（When2Call の長文が大半）の位置を cand_idx で -1 にした
  `data/emb/<provider>/<shard>_notrunc/` を別名で作る（既存ディレクトリは上書きしない。strings/emb/pca は元のハードリンクで再抽出しない。
  meta.json `exclude_truncated` に Lc・source 別の除外件数）。除外は意味表現蒸留 loss の target 位置だけで、判断用 KD/CE には影響しない。
  `--sem-emb data/emb/<provider>/pubA_notrunc` で使う（meta の宣言があるときだけ `SemTargets.allow_masked`）。
- 評価専用 Teacher 埋め込み `data/emb_eval/<provider>/<shard>/`（`diag_sem prepare-eval`。MASSIVE val/test 候補と train 候補の一部）は
  **学習から読めない**: `model.load_sem_dir` / `train.py --sem-emb` は、パス（symlink 解決後も）に `emb_eval` を含む・`EVAL_ONLY` マーカー・meta `eval_only` を拒否する。
  診断は `python -m tb250distill.diag_sem run|summary`（結果 `runs/diag_sem/`、表は docs/DIAG_SEM.md）。
- 評価: eval.json top-level `key_metrics.massive_test_agreement`（test の source=massive の Teacher 一致率。source キーが無い shard では null）と
  train.log の `FINAL key_metrics:` 行。best checkpoint の選択は従来どおり val_kl。

## 評価指標

teacher top-1 agreement, gold accuracy, KL(teacher||student), NLL(gold), Brier, ECE(15 bins),
random baseline（= mean(1/k)）、entropy 比較・「teacher uncertain / student confident」抽出、
samples/s, tokens/s, step latency, max VRAM（nvidia-smi）, 温度, wall-clock。
robustness: robust split の各 variant で agreement/KL を元 item と比較。
