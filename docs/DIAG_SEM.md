# Candidate semantic 診断（DIAG_SEM）

> 対象外: GT 430 は実験から除外（sem17/gt430 は途中停止・評価なし）。表の baseA_gt430 は除外前に完了していた Baseline A の既存 run の診断結果で、参考として残している。

## 要点（人手記述。数値は下の表）

- **判定: すべての run が「近くない（表現自体が未見の言い回しへ汎化できていない）」**。MASSIVE test の unseen 240 文字列の intent retrieval accuracy は、Teacher(0.6B)空間 0.858 に対し Student は 0.02〜0.07（chance 0.017）。
- sem06（Teacher=Qwen3-Embedding-0.6B）: head は seen（学習した文字列）では Teacher と整合（cos 0.68、specificity 0.43）するが unseen では cos 0.26〜0.28、specificity 0.08〜0.10（seen の 18〜22%）。
  probe（線形読み出し）の unseen specificity は Baseline A（0.065〜0.076）や**未学習の乱数初期化 GRU（ref_init 0.114）と同程度**で、sem 蒸留で増えていない。seen 内でも intent 検索 acc は 0.15〜0.25（Teacher 0.78）。
- sem17（Teacher=Qwen3-1.7B mean pooling）: cos は高い（unseen 0.85）が、同言語・他 intent との差（specificity）は 0.02 で Baseline A の probe と同じ。cos の大半は共通成分。
  さらに Teacher 空間自体の intent acc が 0.229 しかなく（0.6B は 0.858）、この target は paraphrase を区別する情報が薄い。
- 例: unseen「asks the assistant to make the room darker」の Student 最近傍は「ask the assistant to calculate」（qa_maths）。ラッパー等の表層（共通語）で近傍が決まり、内容語の意味では決まっていない。
- 判断経路: 表現だけで Teacher top-1 を選ぶ rep_acc は sem06 で 0.40〜0.44、実判断の judge_acc は 0.34〜0.39（random 0.26）。rep が当たる item でも judge が当たるのは 3〜5 割で、phi は 0.03〜0.2 と弱い。表現の質が低い上に、判断経路もそれを強く使っていない。
- **取得上の注意**: 1.7B（mean pooling）の llama-server embedding は、同一リクエスト内の他文字列に影響される（batch=1 は文字列だけで決まる。batch 32 と batch 1 の cos 平均 0.96、最小 0.44）。
  学習用 target（batch 32 で取得）はこの汚染を含む。評価専用 emb は batch=1 版（`qwen3_1p7b`）を主、batch=32 版（`qwen3_1p7b__batch32`）を併記。0.6B（last pooling）は再取得との cos 0.9999 で問題なし。

生成: 2026-10-06 17:57:29+0900。値はすべて実測（np backend の CPU 計算、各 run の best checkpoint）。

## 読み方

- seen = train 候補に出た文字列（MASSIVE train 1200 件 + 他 source の seed 固定抽出）、unseen = MASSIVE test 候補の文字列（train に 0% 出現、240 件）。
- 空間: **head** = 学習時に cosine loss を取った projection head 出力（sem run のみ）、**probe** = seen だけで学習した ridge 線形プローブ（h_c -> Teacher PCA。seen 値は phrase 単位 group 5-fold の cross-fit、unseen は全 seen で fit。head の無い Baseline A と sem run を同じ尺度で比べる）、**h** = h_c そのもの（seen 平均で中心化）。
- cos = 空間の z と評価専用 Teacher PCA 埋め込み（128 次元）の cosine 平均。chance = 同集合内の別文字列との平均 cosine。
- R@k = unseen 各文字列の Teacher 空間 top-k 近傍集合と Student 空間 top-k 近傍集合の重なり（pool = seen_massive 1200 + unseen 240、自分自身は除外）。chance = k/(pool-1)。
- intent acc = unseen の最近傍 seen_massive が同 intent の割合（60 intent、chance ≈ 1/60）。Teacher = Teacher 空間での同じ値（上限の目安）。margin/AUC は同言語の (unseen, seen_massive) ペアで同 intent と異 intent の cosine を比べた差と、同 intent の cosine が高い確率。
- 判定の目安: 「十分近い」= unseen の intent acc >= Teacher 空間の値の 70% かつ chance の 5 倍以上、かつ unseen の cos >= seen(massive) の cos の 80%。満たさなければ「近くない（表現自体が学べていない）」。

## Teacher = qwen3_1p7b

### A. Teacher 埋め込みとの cosine と近傍の再現（seen / unseen）

| run | 空間 | cos seen(massive) | cos seen(synth) | cos seen(jcqa) | cos seen(w2c) | cos **unseen** | chance(unseen) | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ref_init | probe | 0.776 | 0.829 | 0.702 | 0.724 | **0.672** | 0.074 | 0.046 | 0.041 | 0.043 |
| ref_init | h | - | - | - | - | - | - | 0.058 | 0.076 | 0.083 |
| baseA_gt430 | probe | 0.809 | 0.716 | 0.742 | 0.827 | **0.534** | 0.099 | 0.075 | 0.064 | 0.087 |
| baseA_gt430 | h | - | - | - | - | - | - | 0.133 | 0.090 | 0.109 |
| baseA_gt710 | probe | 0.830 | 0.718 | 0.735 | 0.821 | **0.209** | 0.100 | 0.058 | 0.066 | 0.082 |
| baseA_gt710 | h | - | - | - | - | - | - | 0.058 | 0.091 | 0.111 |
| baseA_gt730 | probe | 0.825 | 0.751 | 0.759 | 0.822 | **0.621** | 0.141 | 0.046 | 0.071 | 0.096 |
| baseA_gt730 | h | - | - | - | - | - | - | 0.113 | 0.118 | 0.134 |
| baseA_wx2100 | probe | 0.810 | 0.727 | 0.755 | 0.852 | **0.447** | 0.118 | 0.083 | 0.060 | 0.086 |
| baseA_wx2100 | h | - | - | - | - | - | - | 0.113 | 0.110 | 0.118 |
| sem17_wx2100 | head | 0.936 | 0.938 | 0.789 | 0.820 | **0.847** | 0.087 | 0.058 | 0.078 | 0.102 |
| sem17_wx2100 | probe | 0.953 | 0.946 | 0.820 | 0.903 | **0.886** | 0.084 | 0.050 | 0.087 | 0.110 |
| sem17_wx2100 | h | - | - | - | - | - | - | 0.125 | 0.117 | 0.145 |
| (chance) | | | | | | | | 0.0007 | 0.0035 | 0.0069 |

### A2. cosine の中身（共通成分で説明できる分を除く）

PCA 空間は言語・文体などの共通成分が大きく、cos だけでは intent を当てているか分からない。「他 intent」= 同言語で intent が違う別文字列の Teacher 埋め込みとの cosine 平均。specificity = cos(自分の Teacher 埋め込み) - cos(他 intent)。
| run | 空間 | cos seen | 他intent(seen) | specificity seen | cos unseen | 他intent(unseen) | specificity unseen | spec. unseen/seen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ref_init | probe | 0.776 | 0.757 | 0.019 | 0.672 | 0.655 | 0.017 | 0.90 |
| baseA_gt430 | probe | 0.809 | 0.785 | 0.024 | 0.534 | 0.511 | 0.023 | 0.95 |
| baseA_gt710 | probe | 0.830 | 0.804 | 0.026 | 0.209 | 0.186 | 0.023 | 0.88 |
| baseA_gt730 | probe | 0.825 | 0.802 | 0.024 | 0.621 | 0.590 | 0.031 | 1.31 |
| baseA_wx2100 | probe | 0.810 | 0.786 | 0.024 | 0.447 | 0.417 | 0.030 | 1.23 |
| sem17_wx2100 | head | 0.936 | 0.917 | 0.019 | 0.847 | 0.826 | 0.021 | 1.08 |
| sem17_wx2100 | probe | 0.953 | 0.932 | 0.021 | 0.886 | 0.866 | 0.020 | 0.99 |

### B. 言い換えの近さ（unseen -> seen_massive の intent 検索）

seen 内 = 学習した文字列どうし（同じ言い回しのラッパー違いの兄弟は除く）の intent 検索。unseen の値が seen 内より大きく下がれば未見の言い回しへの汎化の問題、seen 内でも低ければ表現が intent を整理できていない。
| run | 空間 | intent acc (unseen) | (同言語のみ) | seen 内 intent acc | cos 同intent | cos 異intent | margin(同言語) | AUC(同言語) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Teacher 空間（上限の目安）** | teacher | **0.229** | 0.229 | 0.087 | 0.117 | 0.103 | 0.016 | 0.646 |
| (chance) | | 0.0167 | 0.0167 | 0.0126 | | | 0 | 0.5 |
| ref_init | probe | 0.017 | 0.017 | 0.009 | 0.115 | 0.113 | 0.028 | 0.516 |
| ref_init | h | 0.083 | 0.087 | 0.045 | 0.053 | 0.013 | 0.086 | 0.615 |
| baseA_gt430 | probe | 0.017 | 0.017 | 0.030 | 0.166 | 0.166 | 0.009 | 0.505 |
| baseA_gt430 | h | 0.029 | 0.037 | 0.036 | 0.091 | 0.016 | 0.089 | 0.582 |
| baseA_gt710 | probe | 0.021 | 0.017 | 0.020 | 0.121 | 0.121 | 0.003 | 0.502 |
| baseA_gt710 | h | 0.029 | 0.033 | 0.046 | 0.104 | 0.010 | 0.113 | 0.605 |
| baseA_gt730 | probe | 0.021 | 0.017 | 0.035 | 0.193 | 0.186 | 0.012 | 0.507 |
| baseA_gt730 | h | 0.050 | 0.062 | 0.046 | 0.120 | 0.018 | 0.109 | 0.601 |
| baseA_wx2100 | probe | 0.017 | 0.013 | 0.028 | 0.152 | 0.154 | 0.008 | 0.502 |
| baseA_wx2100 | h | 0.037 | 0.042 | 0.038 | 0.117 | 0.011 | 0.119 | 0.611 |
| sem17_wx2100 | head | 0.046 | 0.050 | 0.050 | 0.115 | 0.113 | 0.005 | 0.509 |
| sem17_wx2100 | probe | 0.029 | 0.029 | 0.036 | 0.097 | 0.095 | 0.004 | 0.507 |
| sem17_wx2100 | h | 0.037 | 0.037 | 0.072 | 0.054 | 0.015 | 0.044 | 0.552 |

### C. 判定（head があれば head、無ければ probe。probe も併記）

| run | 空間 | intent acc | / Teacher | x chance | cos unseen/seen | 条件1 (>=70% Teacher) | 条件2 (>=5x chance) | 条件3 (cos>=80% seen) | 判定 |
|---|---|---:|---:|---:|---:|:-:|:-:|:-:|---|
| ref_init | probe | 0.017 | 0.07 | 1.0 | 0.672/0.776 = 0.87 | NG | NG | OK | 近くない |
| baseA_gt430 | probe | 0.017 | 0.07 | 1.0 | 0.534/0.809 = 0.66 | NG | NG | NG | 近くない |
| baseA_gt710 | probe | 0.021 | 0.09 | 1.2 | 0.209/0.830 = 0.25 | NG | NG | NG | 近くない |
| baseA_gt730 | probe | 0.021 | 0.09 | 1.2 | 0.621/0.825 = 0.75 | NG | NG | NG | 近くない |
| baseA_wx2100 | probe | 0.017 | 0.07 | 1.0 | 0.447/0.810 = 0.55 | NG | NG | NG | 近くない |
| sem17_wx2100 | head | 0.046 | 0.20 | 2.7 | 0.847/0.936 = 0.90 | NG | NG | OK | 近くない |
| sem17_wx2100 | probe | 0.029 | 0.13 | 1.7 | 0.886/0.953 = 0.93 | NG | NG | OK | 近くない |

### D. 判断経路との関係（MASSIVE test item。表現 z_c と Teacher 正解候補の類似 vs 評価時の判断）

target = Teacher top-1 候補（gold 版は JSON）。rep_acc = argmax_j cos(z_j, Teacher 埋め込み(target)) が target の割合（表現だけで選んだ場合）、judge_acc = 実際の判断（prefix からの継続 pass）の argmax が target の割合（= Teacher 一致率）。
| run | 空間 | n | random | rep_acc | judge_acc | judge_acc / rep ok | judge_acc / rep ng | phi(rep,judge) | Spearman(log p, sim) | sim(target)-他 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseA_gt430 | probe | 500 | 0.262 | 0.252 | 0.370 | 0.516 | 0.321 | 0.175 | 0.153 | -0.023 |
| baseA_gt710 | probe | 500 | 0.262 | 0.248 | 0.358 | 0.532 | 0.300 | 0.209 | 0.133 | 0.024 |
| baseA_gt730 | probe | 500 | 0.262 | 0.288 | 0.394 | 0.424 | 0.382 | 0.038 | 0.009 | 0.003 |
| baseA_wx2100 | probe | 500 | 0.262 | 0.226 | 0.332 | 0.443 | 0.300 | 0.127 | 0.074 | 0.013 |
| sem17_wx2100 | head | 500 | 0.262 | 0.340 | 0.310 | 0.482 | 0.221 | 0.268 | 0.276 | 0.022 |
| sem17_wx2100 | probe | 500 | 0.262 | 0.328 | 0.310 | 0.293 | 0.319 | -0.026 | 0.085 | 0.015 |

### E. 例（unseen 文字列、同言語の seen_massive への cosine。paraphrase = 同 intent の train 言い回し 10 件、other = 異 intent）

**ref_init**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.892/0.936 | 0.933/0.971 | 0.879/0.969 | 0.839/0.968 | ミュートにするための操作 (audio_volume_mute, 0.969) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.948/0.980 | 0.942/0.985 | 0.528/0.767 | 0.413/0.879 | 声をかけて挨拶すること (general_greet, 0.879) |
| asks the assistant to light up the room | iot_hue_lighton | 0.915/0.963 | 0.915/0.970 | 0.964/0.977 | 0.668/0.986 | the user wants to find a film to watch (recommendation_movies, 0.986) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.899/0.957 | 0.914/0.980 | 0.864/0.979 | 0.658/0.984 | the goal is to play a game (play_game, 0.984) |

**baseA_gt430**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.892/0.936 | 0.933/0.971 | 0.901/0.974 | 0.861/0.982 | 要するに、音を小さくする (audio_volume_down, 0.982) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.948/0.980 | 0.942/0.985 | 0.937/0.971 | 0.856/0.989 | 連絡先を追加する依頼 (email_addcontact, 0.989) |
| asks the assistant to light up the room | iot_hue_lighton | 0.915/0.963 | 0.915/0.970 | 0.683/0.974 | 0.728/0.985 | the goal is to check social media updates (social_query, 0.985) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.899/0.957 | 0.914/0.980 | 0.810/0.980 | 0.740/0.989 | the goal is to order takeaway food (takeaway_order, 0.989) |

**baseA_gt710**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.892/0.936 | 0.933/0.971 | -0.407/-0.098 | -0.190/0.679 | 要するに、設定済みのアラームを確認する (alarm_query, 0.679) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.948/0.980 | 0.942/0.985 | 0.806/0.854 | 0.788/0.928 | タクシーを呼ぶ依頼 (transport_taxi, 0.928) |
| asks the assistant to light up the room | iot_hue_lighton | 0.915/0.963 | 0.915/0.970 | 0.651/0.976 | 0.744/0.987 | the user wants to check the date or time (datetime_query, 0.987) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.899/0.957 | 0.914/0.980 | 0.634/0.943 | 0.694/0.964 | the user wants to ask the assistant to calculate (qa_maths, 0.964) |

**baseA_gt730**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.892/0.936 | 0.933/0.971 | -0.227/-0.032 | -0.067/0.642 | 為替を計算すること (qa_currency, 0.642) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.948/0.980 | 0.942/0.985 | 0.874/0.905 | 0.840/0.952 | あいさつをすること (general_greet, 0.952) |
| asks the assistant to light up the room | iot_hue_lighton | 0.915/0.963 | 0.915/0.970 | 0.278/0.546 | 0.391/0.872 | ask for movie recommendations (recommendation_movies, 0.872) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.899/0.957 | 0.914/0.980 | 0.667/0.874 | 0.641/0.976 | ask the assistant to calculate (qa_maths, 0.976) |

**baseA_wx2100**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.892/0.936 | 0.933/0.971 | -0.658/-0.522 | -0.513/0.791 | 相手のメールアドレスを尋ねる依頼 (email_querycontact, 0.791) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.948/0.980 | 0.942/0.985 | 0.894/0.958 | 0.815/0.976 | 笑える話をせがむための操作 (general_joke, 0.976) |
| asks the assistant to light up the room | iot_hue_lighton | 0.915/0.963 | 0.915/0.970 | 0.615/0.966 | 0.757/0.982 | the goal is to tell the assistant I dislike this music (music_dislikeness, 0.982) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.899/0.957 | 0.914/0.980 | 0.624/0.959 | 0.745/0.977 | the goal is to tell the assistant I dislike this music (music_dislikeness, 0.977) |

**sem17_wx2100**（student 空間 = head）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.892/0.936 | 0.933/0.971 | 0.871/0.956 | 0.920/0.976 | コーヒーメーカーでコーヒーを淹れる (iot_coffee, 0.976) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.948/0.980 | 0.942/0.985 | 0.836/0.861 | 0.830/0.891 | 目覚ましをセットする (alarm_set, 0.891) |
| asks the assistant to light up the room | iot_hue_lighton | 0.915/0.963 | 0.915/0.970 | 0.963/0.983 | 0.945/0.987 | the goal is to turn the volume down (audio_volume_down, 0.987) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.899/0.957 | 0.914/0.980 | 0.927/0.955 | 0.899/0.966 | the user wants to brew coffee with the machine (iot_coffee, 0.966) |

## Teacher = qwen3_1p7b__batch32

### A. Teacher 埋め込みとの cosine と近傍の再現（seen / unseen）

| run | 空間 | cos seen(massive) | cos seen(synth) | cos seen(jcqa) | cos seen(w2c) | cos **unseen** | chance(unseen) | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ref_init | probe | 0.746 | 0.796 | 0.615 | 0.574 | **0.641** | 0.098 | 0.292 | 0.109 | 0.095 |
| ref_init | h | - | - | - | - | - | - | 0.312 | 0.152 | 0.140 |
| baseA_gt430 | probe | 0.779 | 0.676 | 0.660 | 0.675 | **0.510** | 0.136 | 0.025 | 0.043 | 0.064 |
| baseA_gt430 | h | - | - | - | - | - | - | 0.029 | 0.055 | 0.079 |
| baseA_gt710 | probe | 0.785 | 0.683 | 0.660 | 0.674 | **0.262** | 0.143 | 0.021 | 0.047 | 0.078 |
| baseA_gt710 | h | - | - | - | - | - | - | 0.037 | 0.049 | 0.075 |
| baseA_gt730 | probe | 0.784 | 0.720 | 0.676 | 0.676 | **0.573** | 0.158 | 0.017 | 0.045 | 0.059 |
| baseA_gt730 | h | - | - | - | - | - | - | 0.042 | 0.054 | 0.083 |
| baseA_wx2100 | probe | 0.778 | 0.676 | 0.675 | 0.703 | **0.482** | 0.129 | 0.042 | 0.040 | 0.058 |
| baseA_wx2100 | h | - | - | - | - | - | - | 0.071 | 0.063 | 0.086 |
| sem17_wx2100 | head | 0.894 | 0.918 | 0.735 | 0.720 | **0.806** | 0.100 | 0.025 | 0.059 | 0.077 |
| sem17_wx2100 | probe | 0.903 | 0.913 | 0.741 | 0.750 | **0.832** | 0.103 | 0.025 | 0.061 | 0.090 |
| sem17_wx2100 | h | - | - | - | - | - | - | 0.042 | 0.074 | 0.104 |
| (chance) | | | | | | | | 0.0007 | 0.0035 | 0.0069 |

### A2. cosine の中身（共通成分で説明できる分を除く）

PCA 空間は言語・文体などの共通成分が大きく、cos だけでは intent を当てているか分からない。「他 intent」= 同言語で intent が違う別文字列の Teacher 埋め込みとの cosine 平均。specificity = cos(自分の Teacher 埋め込み) - cos(他 intent)。
| run | 空間 | cos seen | 他intent(seen) | specificity seen | cos unseen | 他intent(unseen) | specificity unseen | spec. unseen/seen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ref_init | probe | 0.746 | 0.692 | 0.054 | 0.641 | 0.620 | 0.022 | 0.40 |
| baseA_gt430 | probe | 0.779 | 0.751 | 0.028 | 0.510 | 0.492 | 0.018 | 0.64 |
| baseA_gt710 | probe | 0.785 | 0.754 | 0.031 | 0.262 | 0.234 | 0.027 | 0.88 |
| baseA_gt730 | probe | 0.784 | 0.756 | 0.029 | 0.573 | 0.550 | 0.023 | 0.81 |
| baseA_wx2100 | probe | 0.778 | 0.748 | 0.030 | 0.482 | 0.459 | 0.024 | 0.80 |
| sem17_wx2100 | head | 0.894 | 0.870 | 0.023 | 0.806 | 0.782 | 0.023 | 1.00 |
| sem17_wx2100 | probe | 0.903 | 0.875 | 0.029 | 0.832 | 0.814 | 0.018 | 0.61 |

### B. 言い換えの近さ（unseen -> seen_massive の intent 検索）

seen 内 = 学習した文字列どうし（同じ言い回しのラッパー違いの兄弟は除く）の intent 検索。unseen の値が seen 内より大きく下がれば未見の言い回しへの汎化の問題、seen 内でも低ければ表現が intent を整理できていない。
| run | 空間 | intent acc (unseen) | (同言語のみ) | seen 内 intent acc | cos 同intent | cos 異intent | margin(同言語) | AUC(同言語) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Teacher 空間（上限の目安）** | teacher | **0.171** | 0.171 | 0.104 | 0.133 | 0.111 | 0.028 | 0.595 |
| (chance) | | 0.0167 | 0.0167 | 0.0126 | | | 0 | 0.5 |
| ref_init | probe | 0.037 | 0.037 | 0.014 | 0.109 | 0.104 | 0.028 | 0.520 |
| ref_init | h | 0.083 | 0.087 | 0.045 | 0.053 | 0.013 | 0.086 | 0.615 |
| baseA_gt430 | probe | 0.033 | 0.033 | 0.052 | 0.191 | 0.191 | 0.009 | 0.506 |
| baseA_gt430 | h | 0.029 | 0.037 | 0.036 | 0.091 | 0.016 | 0.089 | 0.582 |
| baseA_gt710 | probe | 0.025 | 0.017 | 0.018 | 0.151 | 0.152 | 0.003 | 0.501 |
| baseA_gt710 | h | 0.029 | 0.033 | 0.046 | 0.104 | 0.010 | 0.113 | 0.605 |
| baseA_gt730 | probe | 0.008 | 0.017 | 0.037 | 0.197 | 0.189 | 0.012 | 0.508 |
| baseA_gt730 | h | 0.050 | 0.062 | 0.046 | 0.120 | 0.018 | 0.109 | 0.601 |
| baseA_wx2100 | probe | 0.013 | 0.017 | 0.024 | 0.166 | 0.163 | 0.010 | 0.506 |
| baseA_wx2100 | h | 0.037 | 0.042 | 0.038 | 0.117 | 0.011 | 0.119 | 0.611 |
| sem17_wx2100 | head | 0.046 | 0.050 | 0.050 | 0.115 | 0.113 | 0.005 | 0.509 |
| sem17_wx2100 | probe | 0.033 | 0.033 | 0.034 | 0.109 | 0.106 | 0.004 | 0.509 |
| sem17_wx2100 | h | 0.037 | 0.037 | 0.072 | 0.054 | 0.015 | 0.044 | 0.552 |

### C. 判定（head があれば head、無ければ probe。probe も併記）

| run | 空間 | intent acc | / Teacher | x chance | cos unseen/seen | 条件1 (>=70% Teacher) | 条件2 (>=5x chance) | 条件3 (cos>=80% seen) | 判定 |
|---|---|---:|---:|---:|---:|:-:|:-:|:-:|---|
| ref_init | probe | 0.037 | 0.22 | 2.2 | 0.641/0.746 = 0.86 | NG | NG | OK | 近くない |
| baseA_gt430 | probe | 0.033 | 0.20 | 2.0 | 0.510/0.779 = 0.65 | NG | NG | NG | 近くない |
| baseA_gt710 | probe | 0.025 | 0.15 | 1.5 | 0.262/0.785 = 0.33 | NG | NG | NG | 近くない |
| baseA_gt730 | probe | 0.008 | 0.05 | 0.5 | 0.573/0.784 = 0.73 | NG | NG | NG | 近くない |
| baseA_wx2100 | probe | 0.013 | 0.07 | 0.7 | 0.482/0.778 = 0.62 | NG | NG | NG | 近くない |
| sem17_wx2100 | head | 0.046 | 0.27 | 2.7 | 0.806/0.894 = 0.90 | NG | NG | OK | 近くない |
| sem17_wx2100 | probe | 0.033 | 0.20 | 2.0 | 0.832/0.903 = 0.92 | NG | NG | OK | 近くない |

### D. 判断経路との関係（MASSIVE test item。表現 z_c と Teacher 正解候補の類似 vs 評価時の判断）

target = Teacher top-1 候補（gold 版は JSON）。rep_acc = argmax_j cos(z_j, Teacher 埋め込み(target)) が target の割合（表現だけで選んだ場合）、judge_acc = 実際の判断（prefix からの継続 pass）の argmax が target の割合（= Teacher 一致率）。
| run | 空間 | n | random | rep_acc | judge_acc | judge_acc / rep ok | judge_acc / rep ng | phi(rep,judge) | Spearman(log p, sim) | sim(target)-他 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseA_gt430 | probe | 500 | 0.262 | 0.238 | 0.370 | 0.429 | 0.352 | 0.068 | 0.081 | -0.040 |
| baseA_gt710 | probe | 500 | 0.262 | 0.266 | 0.358 | 0.549 | 0.289 | 0.240 | 0.139 | 0.026 |
| baseA_gt730 | probe | 500 | 0.262 | 0.270 | 0.394 | 0.430 | 0.381 | 0.044 | -0.015 | 0.003 |
| baseA_wx2100 | probe | 500 | 0.262 | 0.242 | 0.332 | 0.455 | 0.293 | 0.147 | 0.128 | 0.025 |
| sem17_wx2100 | head | 500 | 0.262 | 0.308 | 0.310 | 0.416 | 0.263 | 0.152 | 0.203 | 0.018 |
| sem17_wx2100 | probe | 500 | 0.262 | 0.314 | 0.310 | 0.389 | 0.274 | 0.115 | 0.080 | 0.015 |

### E. 例（unseen 文字列、同言語の seen_massive への cosine。paraphrase = 同 intent の train 言い回し 10 件、other = 異 intent）

**ref_init**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.811/0.897 | 0.847/0.963 | 0.816/0.899 | 0.760/0.946 | 要するに、銘柄の値段を確認する (qa_stock, 0.946) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.863/0.950 | 0.840/0.966 | 0.382/0.536 | 0.318/0.742 | 曲が気に入ったと伝える (music_likeness, 0.742) |
| asks the assistant to light up the room | iot_hue_lighton | 0.916/0.964 | 0.854/0.970 | 0.921/0.951 | 0.645/0.974 | a request to switch the outlet on (iot_wemo_on, 0.974) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.836/0.958 | 0.852/0.975 | 0.834/0.976 | 0.656/0.981 | the goal is to play a game (play_game, 0.981) |

**baseA_gt430**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.811/0.897 | 0.847/0.963 | 0.896/0.959 | 0.874/0.968 | 要するに、持ち帰り注文の状態を調べる (takeaway_query, 0.968) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.863/0.950 | 0.840/0.966 | 0.924/0.957 | 0.870/0.986 | 今流れている曲について尋ねるための操作 (music_query, 0.986) |
| asks the assistant to light up the room | iot_hue_lighton | 0.916/0.964 | 0.854/0.970 | 0.641/0.954 | 0.737/0.980 | a request to switch the lamp to another color (iot_hue_lightchange, 0.980) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.836/0.958 | 0.852/0.975 | 0.826/0.968 | 0.746/0.978 | the goal is to tell the assistant I love this music (music_likeness, 0.978) |

**baseA_gt710**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.811/0.897 | 0.847/0.963 | -0.291/0.014 | -0.092/0.706 | 要するに、設定済みのアラームを確認する (alarm_query, 0.706) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.863/0.950 | 0.840/0.966 | 0.827/0.889 | 0.832/0.932 | リストの中身を確認する依頼 (lists_query, 0.932) |
| asks the assistant to light up the room | iot_hue_lighton | 0.916/0.964 | 0.854/0.970 | 0.615/0.950 | 0.695/0.979 | a request to tell the assistant I love this music (music_likeness, 0.979) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.836/0.958 | 0.852/0.975 | 0.668/0.890 | 0.651/0.955 | ask for someone's email address (email_querycontact, 0.955) |

**baseA_gt730**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.811/0.897 | 0.847/0.963 | -0.060/0.157 | 0.082/0.653 | SNSの投稿や通知を確認すること (social_query, 0.653) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.863/0.950 | 0.840/0.966 | 0.771/0.802 | 0.735/0.883 | あいさつをすること (general_greet, 0.883) |
| asks the assistant to light up the room | iot_hue_lighton | 0.916/0.964 | 0.854/0.970 | 0.314/0.505 | 0.399/0.846 | ask about transport or directions (transport_query, 0.846) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.836/0.958 | 0.852/0.975 | 0.657/0.833 | 0.628/0.928 | ask the assistant to calculate (qa_maths, 0.928) |

**baseA_wx2100**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.811/0.897 | 0.847/0.963 | -0.402/-0.204 | -0.276/0.742 | 相手のメールアドレスを尋ねる依頼 (email_querycontact, 0.742) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.863/0.950 | 0.840/0.966 | 0.904/0.923 | 0.801/0.964 | 音楽の情報を調べる (music_query, 0.964) |
| asks the assistant to light up the room | iot_hue_lighton | 0.916/0.964 | 0.854/0.970 | 0.574/0.953 | 0.728/0.986 | a request to tell the assistant I dislike this music (music_dislikeness, 0.986) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.836/0.958 | 0.852/0.975 | 0.594/0.930 | 0.713/0.966 | the goal is to tell the assistant I dislike this music (music_dislikeness, 0.966) |

**sem17_wx2100**（student 空間 = head）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.811/0.897 | 0.847/0.963 | 0.871/0.956 | 0.920/0.976 | コーヒーメーカーでコーヒーを淹れる (iot_coffee, 0.976) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.863/0.950 | 0.840/0.966 | 0.836/0.861 | 0.830/0.891 | 目覚ましをセットする (alarm_set, 0.891) |
| asks the assistant to light up the room | iot_hue_lighton | 0.916/0.964 | 0.854/0.970 | 0.963/0.983 | 0.945/0.987 | the goal is to turn the volume down (audio_volume_down, 0.987) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.836/0.958 | 0.852/0.975 | 0.927/0.955 | 0.899/0.966 | the user wants to brew coffee with the machine (iot_coffee, 0.966) |

## Teacher = qwen3_emb_0p6b

### A. Teacher 埋め込みとの cosine と近傍の再現（seen / unseen）

| run | 空間 | cos seen(massive) | cos seen(synth) | cos seen(jcqa) | cos seen(w2c) | cos **unseen** | chance(unseen) | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ref_init | probe | 0.382 | 0.630 | 0.221 | 0.210 | **0.308** | 0.166 | 0.479 | 0.162 | 0.136 |
| ref_init | h | - | - | - | - | - | - | 0.500 | 0.181 | 0.153 |
| baseA_gt430 | probe | 0.411 | 0.613 | 0.298 | 0.275 | **0.240** | 0.134 | 0.050 | 0.054 | 0.050 |
| baseA_gt430 | h | - | - | - | - | - | - | 0.071 | 0.076 | 0.070 |
| baseA_gt710 | probe | 0.410 | 0.611 | 0.295 | 0.272 | **0.221** | 0.122 | 0.046 | 0.056 | 0.060 |
| baseA_gt710 | h | - | - | - | - | - | - | 0.058 | 0.065 | 0.066 |
| baseA_gt730 | probe | 0.412 | 0.620 | 0.298 | 0.276 | **0.209** | 0.134 | 0.058 | 0.060 | 0.065 |
| baseA_gt730 | h | - | - | - | - | - | - | 0.054 | 0.061 | 0.070 |
| baseA_wx2100 | probe | 0.408 | 0.620 | 0.293 | 0.281 | **0.217** | 0.122 | 0.062 | 0.072 | 0.072 |
| baseA_wx2100 | h | - | - | - | - | - | - | 0.108 | 0.104 | 0.085 |
| sem06_gt710 | head | 0.682 | 0.810 | 0.423 | 0.315 | **0.261** | 0.124 | 0.037 | 0.060 | 0.058 |
| sem06_gt710 | probe | 0.602 | 0.762 | 0.383 | 0.310 | **0.295** | 0.138 | 0.046 | 0.073 | 0.065 |
| sem06_gt710 | h | - | - | - | - | - | - | 0.062 | 0.096 | 0.085 |
| sem06_gt730 | head | 0.686 | 0.811 | 0.420 | 0.316 | **0.276** | 0.163 | 0.037 | 0.069 | 0.067 |
| sem06_gt730 | probe | 0.601 | 0.765 | 0.383 | 0.303 | **0.297** | 0.166 | 0.033 | 0.066 | 0.069 |
| sem06_gt730 | h | - | - | - | - | - | - | 0.079 | 0.083 | 0.078 |
| (chance) | | | | | | | | 0.0007 | 0.0035 | 0.0069 |

### A2. cosine の中身（共通成分で説明できる分を除く）

PCA 空間は言語・文体などの共通成分が大きく、cos だけでは intent を当てているか分からない。「他 intent」= 同言語で intent が違う別文字列の Teacher 埋め込みとの cosine 平均。specificity = cos(自分の Teacher 埋め込み) - cos(他 intent)。
| run | 空間 | cos seen | 他intent(seen) | specificity seen | cos unseen | 他intent(unseen) | specificity unseen | spec. unseen/seen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ref_init | probe | 0.382 | 0.253 | 0.129 | 0.308 | 0.194 | 0.114 | 0.88 |
| baseA_gt430 | probe | 0.411 | 0.286 | 0.125 | 0.240 | 0.166 | 0.074 | 0.59 |
| baseA_gt710 | probe | 0.410 | 0.292 | 0.118 | 0.221 | 0.145 | 0.076 | 0.64 |
| baseA_gt730 | probe | 0.412 | 0.288 | 0.124 | 0.209 | 0.144 | 0.065 | 0.53 |
| baseA_wx2100 | probe | 0.408 | 0.290 | 0.118 | 0.217 | 0.149 | 0.068 | 0.57 |
| sem06_gt710 | head | 0.682 | 0.249 | 0.433 | 0.261 | 0.166 | 0.095 | 0.22 |
| sem06_gt710 | probe | 0.602 | 0.272 | 0.330 | 0.295 | 0.182 | 0.113 | 0.34 |
| sem06_gt730 | head | 0.686 | 0.252 | 0.433 | 0.276 | 0.197 | 0.080 | 0.18 |
| sem06_gt730 | probe | 0.601 | 0.279 | 0.322 | 0.297 | 0.208 | 0.088 | 0.27 |

### B. 言い換えの近さ（unseen -> seen_massive の intent 検索）

seen 内 = 学習した文字列どうし（同じ言い回しのラッパー違いの兄弟は除く）の intent 検索。unseen の値が seen 内より大きく下がれば未見の言い回しへの汎化の問題、seen 内でも低ければ表現が intent を整理できていない。
| run | 空間 | intent acc (unseen) | (同言語のみ) | seen 内 intent acc | cos 同intent | cos 異intent | margin(同言語) | AUC(同言語) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Teacher 空間（上限の目安）** | teacher | **0.858** | 0.825 | 0.783 | 0.596 | 0.148 | 0.459 | 0.978 |
| (chance) | | 0.0167 | 0.0167 | 0.0126 | | | 0 | 0.5 |
| ref_init | probe | 0.067 | 0.075 | 0.029 | 0.248 | 0.227 | 0.047 | 0.553 |
| ref_init | h | 0.083 | 0.087 | 0.045 | 0.053 | 0.013 | 0.086 | 0.615 |
| baseA_gt430 | probe | 0.033 | 0.025 | 0.059 | 0.265 | 0.238 | 0.039 | 0.545 |
| baseA_gt430 | h | 0.029 | 0.037 | 0.036 | 0.091 | 0.016 | 0.089 | 0.582 |
| baseA_gt710 | probe | 0.029 | 0.025 | 0.038 | 0.223 | 0.202 | 0.029 | 0.533 |
| baseA_gt710 | h | 0.029 | 0.033 | 0.046 | 0.104 | 0.010 | 0.113 | 0.605 |
| baseA_gt730 | probe | 0.021 | 0.025 | 0.037 | 0.238 | 0.202 | 0.041 | 0.552 |
| baseA_gt730 | h | 0.050 | 0.062 | 0.046 | 0.120 | 0.018 | 0.109 | 0.601 |
| baseA_wx2100 | probe | 0.029 | 0.025 | 0.040 | 0.228 | 0.202 | 0.026 | 0.537 |
| baseA_wx2100 | h | 0.037 | 0.042 | 0.038 | 0.117 | 0.011 | 0.119 | 0.611 |
| sem06_gt710 | head | 0.054 | 0.058 | 0.231 | 0.227 | 0.172 | 0.064 | 0.576 |
| sem06_gt710 | probe | 0.033 | 0.029 | 0.152 | 0.259 | 0.194 | 0.079 | 0.599 |
| sem06_gt710 | h | 0.050 | 0.054 | 0.158 | 0.081 | 0.012 | 0.090 | 0.628 |
| sem06_gt730 | head | 0.042 | 0.037 | 0.249 | 0.272 | 0.226 | 0.052 | 0.566 |
| sem06_gt730 | probe | 0.046 | 0.054 | 0.158 | 0.281 | 0.233 | 0.057 | 0.572 |
| sem06_gt730 | h | 0.071 | 0.075 | 0.150 | 0.089 | 0.025 | 0.083 | 0.616 |

### C. 判定（head があれば head、無ければ probe。probe も併記）

| run | 空間 | intent acc | / Teacher | x chance | cos unseen/seen | 条件1 (>=70% Teacher) | 条件2 (>=5x chance) | 条件3 (cos>=80% seen) | 判定 |
|---|---|---:|---:|---:|---:|:-:|:-:|:-:|---|
| ref_init | probe | 0.067 | 0.08 | 4.0 | 0.308/0.382 = 0.81 | NG | NG | OK | 近くない |
| baseA_gt430 | probe | 0.033 | 0.04 | 2.0 | 0.240/0.411 = 0.58 | NG | NG | NG | 近くない |
| baseA_gt710 | probe | 0.029 | 0.03 | 1.7 | 0.221/0.410 = 0.54 | NG | NG | NG | 近くない |
| baseA_gt730 | probe | 0.021 | 0.02 | 1.2 | 0.209/0.412 = 0.51 | NG | NG | NG | 近くない |
| baseA_wx2100 | probe | 0.029 | 0.03 | 1.7 | 0.217/0.408 = 0.53 | NG | NG | NG | 近くない |
| sem06_gt710 | head | 0.054 | 0.06 | 3.2 | 0.261/0.682 = 0.38 | NG | NG | NG | 近くない |
| sem06_gt710 | probe | 0.033 | 0.04 | 2.0 | 0.295/0.602 = 0.49 | NG | NG | NG | 近くない |
| sem06_gt730 | head | 0.042 | 0.05 | 2.5 | 0.276/0.686 = 0.40 | NG | NG | NG | 近くない |
| sem06_gt730 | probe | 0.046 | 0.05 | 2.7 | 0.297/0.601 = 0.49 | NG | NG | NG | 近くない |

### D. 判断経路との関係（MASSIVE test item。表現 z_c と Teacher 正解候補の類似 vs 評価時の判断）

target = Teacher top-1 候補（gold 版は JSON）。rep_acc = argmax_j cos(z_j, Teacher 埋め込み(target)) が target の割合（表現だけで選んだ場合）、judge_acc = 実際の判断（prefix からの継続 pass）の argmax が target の割合（= Teacher 一致率）。
| run | 空間 | n | random | rep_acc | judge_acc | judge_acc / rep ok | judge_acc / rep ng | phi(rep,judge) | Spearman(log p, sim) | sim(target)-他 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseA_gt430 | probe | 500 | 0.262 | 0.364 | 0.370 | 0.516 | 0.286 | 0.230 | 0.204 | 0.047 |
| baseA_gt710 | probe | 500 | 0.262 | 0.318 | 0.358 | 0.421 | 0.328 | 0.090 | 0.045 | 0.022 |
| baseA_gt730 | probe | 500 | 0.262 | 0.358 | 0.394 | 0.469 | 0.352 | 0.115 | -0.001 | 0.038 |
| baseA_wx2100 | probe | 500 | 0.262 | 0.342 | 0.332 | 0.503 | 0.243 | 0.262 | 0.112 | 0.040 |
| sem06_gt710 | head | 500 | 0.262 | 0.418 | 0.388 | 0.493 | 0.313 | 0.182 | 0.103 | 0.066 |
| sem06_gt710 | probe | 500 | 0.262 | 0.438 | 0.388 | 0.466 | 0.327 | 0.141 | 0.173 | 0.088 |
| sem06_gt730 | head | 500 | 0.262 | 0.420 | 0.336 | 0.352 | 0.324 | 0.029 | -0.003 | 0.054 |
| sem06_gt730 | probe | 500 | 0.262 | 0.398 | 0.336 | 0.367 | 0.316 | 0.053 | 0.041 | 0.067 |

### E. 例（unseen 文字列、同言語の seen_massive への cosine。paraphrase = 同 intent の train 言い回し 10 件、other = 異 intent）

**ref_init**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.529/0.697 | 0.225/0.648 | 0.322/0.491 | 0.265/0.556 | 要するに、予定を削除する (calendar_remove, 0.556) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.684/0.801 | 0.161/0.653 | 0.308/0.600 | 0.226/0.674 | ライトをオンにする依頼 (iot_hue_lighton, 0.674) |
| asks the assistant to light up the room | iot_hue_lighton | 0.799/0.862 | 0.223/0.844 | 0.609/0.719 | 0.474/0.809 | the user wants to dim the lights (iot_hue_lightdim, 0.809) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.796/0.825 | 0.195/0.728 | 0.644/0.761 | 0.401/0.705 | the goal is to dim the lights (iot_hue_lightdim, 0.761) |

**baseA_gt430**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.529/0.697 | 0.225/0.648 | 0.537/0.735 | 0.431/0.763 | 要するに、アラームを設定する (alarm_set, 0.763) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.684/0.801 | 0.161/0.653 | 0.370/0.412 | 0.289/0.732 | 今流れている曲について尋ねるための操作 (music_query, 0.732) |
| asks the assistant to light up the room | iot_hue_lighton | 0.799/0.862 | 0.223/0.844 | 0.360/0.492 | 0.448/0.800 | ask the assistant to calculate (qa_maths, 0.800) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.796/0.825 | 0.195/0.728 | 0.627/0.710 | 0.504/0.748 | convert a time between time zones (datetime_convert, 0.748) |

**baseA_gt710**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.529/0.697 | 0.225/0.648 | -0.057/0.467 | 0.176/0.613 | 要するに、天気予報を確認する (weather_query, 0.613) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.684/0.801 | 0.161/0.653 | 0.385/0.493 | 0.358/0.746 | リストの中身を確認する依頼 (lists_query, 0.746) |
| asks the assistant to light up the room | iot_hue_lighton | 0.799/0.862 | 0.223/0.844 | 0.391/0.535 | 0.408/0.698 | the user wants to tell the assistant I love this music (music_likeness, 0.698) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.796/0.825 | 0.195/0.728 | 0.571/0.674 | 0.419/0.750 | ask the assistant to calculate (qa_maths, 0.750) |

**baseA_gt730**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.529/0.697 | 0.225/0.648 | 0.153/0.529 | 0.229/0.642 | 要するに、SNSの投稿や通知を確認する (social_query, 0.642) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.684/0.801 | 0.161/0.653 | 0.084/0.199 | 0.054/0.652 | 最新の出来事を調べる依頼 (news_query, 0.652) |
| asks the assistant to light up the room | iot_hue_lighton | 0.799/0.862 | 0.223/0.844 | 0.216/0.378 | 0.259/0.822 | ask the assistant to calculate (qa_maths, 0.822) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.796/0.825 | 0.195/0.728 | 0.437/0.527 | 0.305/0.806 | ask the assistant to calculate (qa_maths, 0.806) |

**baseA_wx2100**（student 空間 = probe）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.529/0.697 | 0.225/0.648 | 0.014/0.295 | 0.094/0.470 | 要するに、ソーシャルメディアに書き込む (social_post, 0.470) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.684/0.801 | 0.161/0.653 | 0.176/0.324 | 0.127/0.667 | カレンダーに用事を入れる依頼 (calendar_set, 0.667) |
| asks the assistant to light up the room | iot_hue_lighton | 0.799/0.862 | 0.223/0.844 | 0.376/0.549 | 0.368/0.748 | ask the assistant to calculate (qa_maths, 0.748) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.796/0.825 | 0.195/0.728 | 0.467/0.515 | 0.327/0.680 | ask the assistant to calculate (qa_maths, 0.680) |

**sem06_gt710**（student 空間 = head）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.529/0.697 | 0.225/0.648 | -0.010/0.253 | 0.204/0.575 | 行事やイベントの情報を求める依頼 (recommendation_events, 0.575) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.684/0.801 | 0.161/0.653 | 0.315/0.498 | 0.207/0.550 | 放送局を流す依頼 (play_radio, 0.550) |
| asks the assistant to light up the room | iot_hue_lighton | 0.799/0.862 | 0.223/0.844 | 0.358/0.457 | 0.434/0.800 | ask the assistant to calculate (qa_maths, 0.800) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.796/0.825 | 0.195/0.728 | 0.433/0.556 | 0.359/0.804 | ask the assistant to calculate (qa_maths, 0.804) |

**sem06_gt730**（student 空間 = head）

| unseen 文字列 | intent | Teacher paraphrase 平均/最大 | Teacher other 平均/最大 | Student paraphrase 平均/最大 | Student other 平均/最大 | Student 最近傍 top1 (intent, cos) |
|---|---|---|---|---|---|---|
| ユーザーの意図は、音声を無音にすること | audio_volume_mute | 0.529/0.697 | 0.225/0.648 | 0.240/0.463 | 0.306/0.667 | 料理のレシピを尋ねること (cooking_recipe, 0.667) |
| 相手に電子メールを書いて出す要求 | email_sendemail | 0.684/0.801 | 0.161/0.653 | 0.470/0.553 | 0.314/0.802 | 笑える話をせがむ (general_joke, 0.802) |
| asks the assistant to light up the room | iot_hue_lighton | 0.799/0.862 | 0.223/0.844 | 0.331/0.472 | 0.460/0.816 | ask the assistant to calculate (qa_maths, 0.816) |
| asks the assistant to make the room darker | iot_hue_lightdim | 0.796/0.825 | 0.195/0.728 | 0.366/0.468 | 0.433/0.755 | the user wants to greet the assistant (general_greet, 0.755) |

