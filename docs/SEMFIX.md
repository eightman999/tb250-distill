# SEMFIX: 候補意味表現蒸留の修正と損失比較

## 変更点（前段 = S06 の候補意味表現蒸留からの差分）
- **独立ミニバッチ** `--sem-batch M`（既定 0 = 従来）: 判断バッチとは別に、train の unique 候補文字列（`pubA_notrunc` で有効なもの。実測 pool 15,289 件 = jcqa 12,786 / massive 1,200 / synth 1,175 / when2call 128）から
  毎 step M 個を取り、文脈なし pass -> head -> 損失。サンプルは (seed, step) だけで決まる（状態なし。resume で厳密再現）。既定 `--sem-batch-mix equal` = source 等量（M=128 なら 32 ずつ。MASSIVE は 37 step で 1 周）。
- **損失** `--sem-loss`: `cos` / `mse`（L2 正規化後の二乗誤差の次元和 = 2(1-cos)） / `rkd`（行ごとの softmax(S/τ) の KL。Teacher はプール平均で中心化+L2 正規化、既定 元 1024 次元、`--sem-rkd-space pca` で PCA128、τ_t=τ_s=0.1、対角と同一文字列を除外）
  / `infonce`（対称 InfoNCE、τ=0.07、t = PCA128（`--sem-nce-target pca|pca_c|rproj`）、同一文字列は負例から除外）。`+` で組合せ（`--sem-loss-weights`）。cos 以外は `--sem-batch>0` 必須。
- 損失と勾配は **host の numpy float64**（M×M 行列は小さい）。Student は z を download、dL/dz を upload して head -> GRU を backward。np/cl の差は forward のみ。`--sem-batch 0`・新フラグ未指定はビット一致（テストあり）。
- **sem-only 学習** `python -m tb250distill.student.sem_only`（判断の KD/CE なし。Common-S の init、決定的）。診断 `diag_sem run --unseen-split val|test` を追加。
- **SIGTERM**: train.py / sem_only.py は SIGTERM を KeyboardInterrupt と同様に扱い last checkpoint を保存して終了（`train.install_sigterm_as_interrupt`。subprocess テストで検証）。
- ファイル: `tb250distill/student/semfix.py`（新規）、`student/sem_only.py`（新規）、`semfix_report.py`（新規）、`student/model.py`（Student に `max_sem`・`sem_ext`・候補側 buffer の `Nc`: 既定では従来と同一）、`student/train.py`、`diag_sem.py`、`tests/test_semfix.py`、`scripts/semfix_{queue,prod}.sh`、`scripts/launch_semfix_{s1,prod}.sh`。

## スクリーニング（意味表現だけの学習、MASSIVE **val** で評価）
条件: Common-S（init `runs/common/init.npz`、head は seed 決定的）、sem-only 3000 step、M=128、λ=1、AdamW lr 2e-3 / clip 1.0、seed 0（`*s1` は seed 1）、判断の KD/CE なし。
評価: `diag_sem run --unseen-split val`（unseen = MASSIVE val の候補文字列 239 件、seen = train 候補 3,000 件）、空間 = head（z）。`cos_jb` は従来相当（train.py と同じ順序の 32 item の判断バッチ内候補をそのまま使う。重複あり・約 100〜160 件）。
GPU は GT730 / GT710 / WX2100 に割り振り（`gpu` 列）。同一 GPU の条件間比較ではなく、**seed 違いのばらつき（±0.02〜0.03）が条件間の差と同程度**なので、差の読み方は下記「解釈」に従う。

### 最終 weight（step 3000）
| 条件 | GPU | ckpt | unseen intent acc | (同言語) | AUC | specificity unseen | cos unseen | R@1 | R@5 | R@10 | seen intent acc | specificity seen | cos seen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rkd_pca | wx2100 | best | **0.209** | 0.188 | 0.731 | -0.002 | -0.018 | 0.469 | 0.251 | 0.231 | 0.803 | 0.005 | -0.013 |
| rkd | gt730s1 | best | **0.205** | 0.205 | 0.720 | 0.015 | 0.015 | 0.435 | 0.246 | 0.229 | 0.753 | -0.008 | -0.015 |
| rkd_pca | wx2100s1 | best | **0.205** | 0.213 | 0.734 | 0.002 | 0.013 | 0.402 | 0.250 | 0.232 | 0.819 | -0.008 | 0.001 |
| infonce_rkd | gt710 | best | **0.197** | 0.176 | 0.710 | 0.153 | 0.209 | 0.464 | 0.237 | 0.202 | 0.696 | 0.685 | 0.772 |
| infonce_rkd | gt710s1 | best | **0.197** | 0.176 | 0.731 | 0.158 | 0.223 | 0.531 | 0.251 | 0.210 | 0.723 | 0.687 | 0.774 |
| cos_ind | gt710 | best | **0.188** | 0.188 | 0.721 | 0.180 | 0.336 | 0.523 | 0.235 | 0.208 | 0.788 | 0.804 | 0.985 |
| infonce_t10 | wx2100 | best | **0.184** | 0.176 | 0.700 | 0.166 | 0.234 | 0.485 | 0.218 | 0.178 | 0.570 | 0.777 | 0.874 |
| mse | gt730 | best | **0.184** | 0.188 | 0.721 | 0.180 | 0.336 | 0.523 | 0.235 | 0.209 | 0.787 | 0.804 | 0.985 |
| rkd | gt710 | best | **0.176** | 0.176 | 0.717 | 0.003 | -0.017 | 0.435 | 0.223 | 0.206 | 0.767 | 0.007 | -0.003 |
| cos_mass | wx2100 | best | **0.167** | 0.163 | 0.741 | 0.181 | 0.372 | 0.477 | 0.243 | 0.216 | 0.788 | 0.812 | 0.991 |
| cos_ind | gt730s1 | best | **0.163** | 0.163 | 0.751 | 0.188 | 0.362 | 0.485 | 0.237 | 0.215 | 0.797 | 0.804 | 0.985 |
| infonce_rkd_mass | wx2100 | best | **0.155** | 0.155 | 0.710 | 0.157 | 0.214 | 0.469 | 0.243 | 0.210 | 0.698 | 0.724 | 0.785 |
| cos_jb | gt730 | best | **0.142** | 0.142 | 0.721 | 0.161 | 0.335 | 0.477 | 0.203 | 0.178 | 0.789 | 0.801 | 0.981 |
| infonce_pcac | gt730 | best | **0.130** | 0.134 | 0.679 | 0.146 | 0.216 | 0.460 | 0.204 | 0.163 | 0.367 | 0.706 | 0.790 |
| infonce | gt730 | best | **0.117** | 0.126 | 0.669 | 0.151 | 0.211 | 0.460 | 0.194 | 0.151 | 0.338 | 0.693 | 0.774 |
| infonce_t03 | gt710 | best | **0.084** | 0.088 | 0.627 | 0.116 | 0.172 | 0.410 | 0.193 | 0.153 | 0.104 | 0.468 | 0.518 |
| ref_init(p000) | - | p000 | **0.063** | 0.084 | 0.607 | -0.007 | 0.009 | 0.485 | 0.179 | 0.143 | 0.050 | 0.001 | -0.003 |

Teacher 空間の unseen intent acc = 0.858（上限の目安）。空間 = head（z。学習時に損失を取った空間）。unseen = MASSIVE val の候補文字列。

### 途中（step 1500 = p050）
| 条件 | GPU | ckpt | unseen intent acc | (同言語) | AUC | specificity unseen | cos unseen | R@1 | R@5 | R@10 | seen intent acc | specificity seen | cos seen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rkd_pca | wx2100s1 | p050 | **0.209** | 0.205 | 0.728 | -0.003 | 0.011 | 0.402 | 0.257 | 0.238 | 0.808 | -0.008 | -0.003 |
| cos_ind | gt730s1 | p050 | **0.188** | 0.172 | 0.754 | 0.192 | 0.381 | 0.444 | 0.223 | 0.211 | 0.785 | 0.796 | 0.979 |
| rkd_pca | wx2100 | p050 | **0.188** | 0.176 | 0.717 | -0.003 | -0.011 | 0.494 | 0.246 | 0.226 | 0.793 | 0.006 | -0.007 |
| rkd | gt730s1 | p050 | **0.184** | 0.172 | 0.711 | 0.008 | 0.003 | 0.414 | 0.239 | 0.223 | 0.752 | -0.008 | -0.015 |
| mse | gt730 | p050 | **0.180** | 0.188 | 0.712 | 0.178 | 0.345 | 0.489 | 0.238 | 0.212 | 0.793 | 0.799 | 0.979 |
| cos_ind | gt710 | p050 | **0.176** | 0.180 | 0.712 | 0.177 | 0.345 | 0.489 | 0.238 | 0.213 | 0.794 | 0.799 | 0.979 |
| infonce_rkd | gt710 | p050 | **0.163** | 0.155 | 0.703 | 0.155 | 0.198 | 0.473 | 0.222 | 0.201 | 0.713 | 0.655 | 0.729 |
| infonce_rkd_mass | wx2100 | p050 | **0.163** | 0.155 | 0.696 | 0.157 | 0.178 | 0.502 | 0.243 | 0.207 | 0.703 | 0.699 | 0.740 |
| cos_mass | wx2100 | p050 | **0.159** | 0.163 | 0.718 | 0.182 | 0.374 | 0.489 | 0.227 | 0.211 | 0.790 | 0.806 | 0.988 |
| infonce_rkd | gt710s1 | p050 | **0.155** | 0.155 | 0.712 | 0.155 | 0.197 | 0.531 | 0.249 | 0.204 | 0.707 | 0.652 | 0.727 |
| rkd | gt710 | p050 | **0.142** | 0.142 | 0.695 | -0.001 | -0.012 | 0.406 | 0.210 | 0.200 | 0.741 | 0.006 | 0.004 |
| cos_jb | gt730 | p050 | **0.134** | 0.146 | 0.712 | 0.158 | 0.334 | 0.435 | 0.195 | 0.184 | 0.777 | 0.792 | 0.973 |
| infonce_pcac | gt730 | p050 | **0.130** | 0.121 | 0.674 | 0.144 | 0.219 | 0.489 | 0.213 | 0.170 | 0.349 | 0.673 | 0.758 |
| infonce_t10 | wx2100 | p050 | **0.126** | 0.126 | 0.683 | 0.165 | 0.241 | 0.515 | 0.225 | 0.189 | 0.551 | 0.751 | 0.848 |
| infonce | gt730 | p050 | **0.113** | 0.105 | 0.652 | 0.149 | 0.217 | 0.469 | 0.205 | 0.170 | 0.341 | 0.659 | 0.746 |
| infonce_t03 | gt710 | p050 | **0.105** | 0.100 | 0.624 | 0.110 | 0.172 | 0.368 | 0.196 | 0.155 | 0.100 | 0.437 | 0.499 |
| ref_init(p000) | - | p000 | **0.063** | 0.084 | 0.607 | -0.007 | 0.009 | 0.485 | 0.179 | 0.143 | 0.050 | 0.001 | -0.003 |

Teacher 空間の unseen intent acc = 0.858（上限の目安）。空間 = head（z。学習時に損失を取った空間）。unseen = MASSIVE val の候補文字列。

### 条件別 unseen intent acc（seed 反復の平均）
rkd_pca 0.207（0.209/0.205）、infonce_rkd 0.197（0.197/0.197）、rkd 0.191（0.205/0.176）、mse 0.184（1 本）、cos_ind 0.175（0.188/0.163）、cos_jb 0.142（1 本）、infonce 0.117、infonce τ0.03 0.084、init 0.063。Teacher 空間 0.858。
n=239 の比率なので 1 本の標準誤差は約 0.025。

## 解釈（実測と推測を分ける）
実測:
- **どの損失でも unseen intent acc は 0.08〜0.21（Teacher 0.858 の 25% 以下）** で、DIAG_SEM の「近い」基準（Teacher の 70%）に遠い。一方 seen（学習した文字列）内の intent acc は 0.7〜0.8（cos/rkd）で、**seen→unseen の落差（0.8→0.2）が主問題**。train の MASSIVE は 60 intent × 2 言語 × 2 言い回し = 240 個の言い回し（× 5 ラッパー）しか無く、val/test は別の言い回しなので、損失の種類より言い回しの多様性が律速している可能性が高い（推測）。
- 判断バッチ内 cos（cos_jb 0.142）→ 独立ミニバッチ cos（0.175）→ rkd 系（0.19〜0.21）の順に増えるが、**差は seed ばらつきと同程度**で、有意とは言えない（rkd_pca vs cos_ind は +0.03 程度）。従来の joint 学習（unseen 0.04〜0.05、S06）よりは大きく改善しているが、これは sem-only（判断 loss なし、λ=1）の効果と独立ミニバッチの効果が混ざっている（切り分けは本番比較で）。
- **mse は cos と数学的に同値**（正規化後の二乗誤差 = 2(1-cos)。方向は同じでスケールが 2 倍）で、cos_ind と mse の差（0.188 vs 0.184、別 GPU）はノイズ。
- **rkd の cos/specificity（Teacher PCA との cosine）は意味を持たない**（絶対位置を合わせていないので 0 付近。表の specificity -0.00〜0.02）。rkd の評価は intent acc・AUC・R@k で見る。infonce は絶対位置も合うので cos/specificity も出る（infonce_rkd: specificity 0.15）。
- **infonce 単独は悪い**（unseen 0.12、seen 0.34）: バッチ内に同 intent の文字列（同じ言い回しのラッパー違い）が複数ありそれを負例にするため seen 内でも intent が揃わない。τ を小さく（0.03）すると更に悪化、大きく（0.1）で 0.18。rkd と組み合わせると 0.197 に回復。
- source 配分を MASSIVE 寄せ（50%）にしても改善しない（cos_mass 0.167、infonce_rkd_mass 0.155）。
- WX2100 の run も GT730/GT710 と同水準で、GPU 種別による系統差は見えない。
- AdamW の weight decay が判断 head（head.*）にも掛かるため、sem-only では head.* は勾配 0 のまま (1-lr*wd)^steps で縮む（無害、テストで確認）。
注意（val/test の独立性）: MASSIVE の **val と test は候補文字列集合がほぼ同一**（239/240 件共通。tier=1 の言い回し）。val でのモデル選択は test の unseen 文字列を見て選んだことになる。選択肢は少数の離散条件で、選択は丸めた差で行ったが、**表現指標についての val/test は独立な確認にならない**（本番の判断指標＝test の item 単位 Teacher 一致率は、選択に使っていない）。

## 選んだ損失と根拠
- 最良: **rkd（PCA128 空間、τ=0.1/0.1）** — val unseen intent acc 平均 0.207（2 seed）、seen 0.80。2 番手: **infonce+rkd** — 0.197（2 seed）。cos は 0.175。
- 差は小さく（≈1 標準誤差）、「rkd 系が cos より改善する」は弱い証拠に留まる。本番比較は rkd_pca と infonce_rkd を GT730/GT710 の両方で実施。

## 本番比較（判断と同時学習、Baseline A と同条件 + `--sem-cand-weight 0.5 --sem-emb data/emb/qwen3_emb_0p6b/pubA_notrunc --sem-batch 128 --sem-loss <条件>`）
run: `runs/semfix/prod/<条件>/<gpu>/`（tb250）。起動: `scripts/launch_semfix_prod.sh`。完了後の評価は `scripts/semfix_prod.sh` が続けて実行（evaluate: val/test/robust・--dump-preds・replay-db、diag_sem: MASSIVE test unseen）。
結果表は完了後にここへ追記する（`python -m tb250distill.semfix_report prod` で集計）。
