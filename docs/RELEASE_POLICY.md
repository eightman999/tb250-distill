# 公開版モデルの学習データ方針（2026-10-06 決定）

ユーザー判断: JGLUE / JCommonsenseQA は公開版の学習に採用してよい。

| source | ライセンス | 公開版 | 内部版 | 備考 |
|---|---|---|---|---|
| synth | 自作 | 使う | 使う | |
| jcqa（JGLUE JCommonsenseQA） | CC BY-SA 4.0 | **使う（ユーザー承認）** | 使う | モデルカードに帰属表示・ライセンス明記 |
| massive | CC BY 4.0 | 使う（帰属表示） | 使う | |
| when2call | CC BY 4.0 | 使う（帰属表示） | 使う | |
| jnli（JGLUE JNLI） | CC BY-SA 4.0 | 未決（使わない） | 使う | ユーザー未承認。Teacher(Qwen3-1.7B) が高確信で誤答（test gold 一致 0.32 < random 0.42） |
| wrime | CC BY-NC-ND 4.0 | 使わない | 使う | 非商用・改変禁止 |
| routellm | Apache-2.0（prompt は LMSYS-Chat-1M 由来、gold は GPT-4 判定） | 使わない | 使う | 元データ規約が及ぶ可能性 |

- モデル重み以外（replay DB、tokenized shard、teacher 出力、data/external）は公開しない。
- 公開版の shard は `tokenize_data --sources synth,jcqa,massive,when2call` で作る。
- Teacher: Qwen3-1.7B（Apache-2.0）。
