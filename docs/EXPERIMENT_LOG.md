# 実験ログ（変更点ごとの差分）

固定条件（特記なき限り）: data `pubA`（Lp256/Lc16）、Common-S 1.49M、init `runs/common/init.npz`、seed 0、batch 32、AdamW lr 2e-3、T=2、KD0.8/CE0.2、3 epoch、val 250 step 毎、best val KL ckpt で評価。
主指標: MASSIVE test Teacher 一致率（unseen wording）。run 間ばらつき ±0.03 程度（OpenCL float atomic の非決定性）。

| ID | 日時 | 変更点（前段からの差分のみ） | run | MASSIVE agree | JCQA gold | 備考 |
|---|---|---|---|---|---|---|
| A | 10/06 14:16 | Baseline A | runs/baseA/common/{gt430,gt710,gt730,wx2100} | 0.370 / 0.358 / 0.394 / 0.332（mean 0.363, sd 0.026） | 0.220 / 0.234 / 0.234 / 0.212 | docs/BASELINE_A.md |
| S06 | 10/06 15:33 | + 候補意味表現蒸留 λ=0.5、Qwen3-Embedding-0.6B（PCA128, cosine）、文脈なし候補 pass、判断経路は不変 | runs/sem06/common/{gt710,gt730} | 0.388 / 0.336（mean 0.362） | 0.210 / 0.226 | 改善なし |
| S17 | 10/06 15:33 | S06 の provider を Qwen3-1.7B（mean pooling）に | runs/sem17/common/{wx2100,gt430} | 0.310 / (GT430 除外) | 0.202 / - | WX2100 は best step 1250。GT430 は 10/06 17:50 にユーザー指示で除外（step 約5400/5715 で SIGTERM 停止、評価なし） |
| D1 | 10/06 17:20〜 | 診断（学習なし）: Teacher/Student 候補埋め込み cosine、Recall@k、seen/unseen、intent retrieval | runs/diag_sem/ | - | - | **全 run「近くない」**（docs/DIAG_SEM.md）。sem06 head: specificity seen 0.43→unseen 0.09、unseen intent acc 0.04〜0.05（Teacher 0.6B 空間 0.858）。近傍は定型ラッパー語で決まる。Qwen3-1.7B mean pooling は target 不適（intent acc 0.229、batch 依存汚染）→ 以降 0.6B のみ。分岐: 判断経路変更は保留、埋め込み蒸留を修正（独立ミニバッチ、mse / 類似度行列蒸留 / InfoNCE を比較） |
| W | 10/06 17:20〜 | 独立 branch: Wikipedia 事前学習（LM）→ Baseline A 同条件で蒸留（sem なし） | runs/wikiA/ (WX2100) | 0.348 | 0.228 | **改善なし**。LM 事前学習 jawiki 86k 記事・約 46M token・1 epoch（5652 step、112 分、val ppl 8205→50.6、GPU 22.3 W / 0.207 J/sample）→ Baseline A 同条件で蒸留（best step 2000）。JCQA gold 0.228（Baseline A mean 0.225、random 0.20）、synth agree 0.647（A: 0.75前後）、cand_para 0.377 と一部悪化。Wikipedia は CC BY-SA 4.0、公開版採否は未判断 |
| SF0 | 10/06 20:54 | スクリーニング基準（sem-only: 判断 KD/CE なし、Common-S init、3000 step、M=128、λ=1、判断バッチ内候補・cos）= 従来相当 | runs/semfix/screen/cos_jb/gt730 | -（学習は意味表現のみ） | - | MASSIVE **val** unseen intent acc 0.142（seen 0.789、Teacher 0.858、init 0.063）。specificity unseen 0.161 |
| SF1 | 10/06 20:54 | SF0 の候補を独立ミニバッチ（source 等量 M=128）に（cos） | runs/semfix/screen/cos_ind/{gt710,gt730s1} | - | - | unseen acc 0.188 / 0.163（seed 0/1）。specificity 0.18 |
| SF2 | 10/06 20:57 | SF1 の損失を mse（L2 正規化後。= 2(1-cos) で cos と同値）に | runs/semfix/screen/mse/gt730 | - | - | unseen acc 0.184（cos と差なし = ノイズ） |
| SF3 | 10/06 21:00 | SF1 の損失を rkd（Teacher 1024 次元を中心化+L2、τ 0.1/0.1）に | runs/semfix/screen/rkd/{gt710,gt730s1} | - | - | unseen acc 0.176 / 0.205。cos/specificity は絶対位置を合わせないので 0 付近（評価は intent acc・R@k） |
| SF4 | 10/06 20:54 | SF3 の rkd 空間を PCA128（中心化）に | runs/semfix/screen/rkd_pca/{wx2100,wx2100s1} | - | - | **unseen acc 0.209 / 0.205（最良）**、seen 0.80、R@10 0.23 |
| SF5 | 10/06 21:01 | SF1 の損失を infonce（対称、τ 0.07、t = PCA128）に | runs/semfix/screen/infonce/gt730 | - | - | unseen acc 0.117、seen 0.338（同 intent の兄弟文字列を負例にするため seen 内でも悪い） |
| SF6 | 10/06 21:08 | SF5 に rkd を加算（infonce+rkd、重み 1:1） | runs/semfix/screen/infonce_rkd/{gt710,gt710s1} | - | - | unseen acc 0.197 / 0.197（2 番手）、specificity 0.15、seen 0.70 |
| SF7 | 10/06 20:58〜 | SF5 の変種: τ=0.1 / τ=0.03 / target 中心化（pca_c） | runs/semfix/screen/infonce_{t10,t03,pcac}/* | - | - | unseen acc 0.184 / 0.084 / 0.130（τ 小は悪化） |
| SF8 | 10/06 21:08〜 | SF1・SF6 の source 配分を massive 0.5 / synth 0.25 / jcqa 0.25 に | runs/semfix/screen/{cos_mass,infonce_rkd_mass}/wx2100 | - | - | unseen acc 0.167 / 0.155（改善なし） |
| SP1 | 10/06 21:31〜 | 本番比較: Baseline A + sem（λ=0.5、notrunc）+ 独立ミニバッチ M=128 + rkd（PCA128） | runs/semfix/prod/rkd_pca/{gt730,gt710} | 中断 | 中断 | 10/06 21:38 ユーザー指示「一旦終了」で SIGTERM 停止（gt730 step 1281/5715、last ckpt 保存）。gt710 側は未開始。21:48 に誤って `--resume` で再開したが 21:49 に再停止（gt730 step 1417、gt710 step 750 で last ckpt 保存）。再開は `RESUME=1 scripts/launch_semfix_prod_resume.sh`（ユーザー指示後） |
| SP2 | 10/06 21:31〜 | SP1 の損失を infonce+rkd に | runs/semfix/prod/infonce_rkd/{gt710,gt730} | 中断 | 中断 | 同時刻に停止（gt710 step 680/5715、last ckpt 保存）。gt730 側は未開始 |

保留: When2Call の Lc 超過候補を sem loss から除外する cand_idx（`pubA_notrunc`）は、判断経路接続後の sem run から適用。λ=1.0/2.0 は判断経路接続後に実施。

## 運用メモ
- 2026-10-06 17:50 GT430 を実験から除外（ユーザー指示。連続学習で 85〜90℃・温度ガード頻発、推論 p50 100ms）。以降の run は GT710 / GT730 / WX2100 の 3 枚で行う。
- バックグラウンド起動（`&`）した run は SIGINT が無視される。停止は SIGTERM になる。train.py / sem_only.py は 10/06 に SIGTERM を KeyboardInterrupt 扱いにして last ckpt を保存して終了するよう修正済み（`tests/test_semfix.py::test_sigterm_*`）。
- プロセス操作は PID 指定かスクリプト経由（`pkill -f` / `pgrep -f` は ssh 自身のコマンドラインに一致して自滅した前例あり）。
- 2026-10-06 21:38 ユーザー指示「一旦終了」で tb250 の全実験プロセスを停止し GPU を解放（データ・ckpt・ログは保持）。
