# Baseline A（固定・2026-10-06）

- データ: `data/tok/pubA`（公開版。train 60,938 = synth 30,000 上限 + jcqa 8,938 + massive 12,000 + when2call 10,000、val/test 各 2,500、robust 1,800、spm vocab 8192、Lp256/Lc16）
- Teacher: Qwen3-1.7B Q4_K_M（RX6400 Vulkan、label logprob、perm2 平均）
- Student: Common-S 1,493,633 params（emb128, GRU192×2）、同一 init `runs/common/init.npz`、seed 0、batch 32、AdamW lr 2e-3、T=2、KD0.8/CE0.2、3 epoch、val 250 step 毎、**best val KL checkpoint で評価**
- run: `tb250:~/tb250-distill/runs/baseA/common/<gpu>/`（eval.json は best ckpt、eval_train_final.json は最終 weight）

| GPU | Arch | Backend | wall | samples/s | J/sample | VRAM | max temp | best val KL (step) | test agree | test KL | ECE | MASSIVE agree | JCQA gold | MASSIVE gold | W2C gold | synth gold | cand_para | unseen_cand | p50 latency |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GT 430 | Fermi | OpenCL1.1+CLBlast | 94.0 min（温度停止 323 s） | 38.6 | n/a | 290 MB | 90 ℃ | 0.909 (3000) | 0.604 | 0.898 | 0.194 | 0.370 | 0.220 | 0.348 | 0.492 | 0.741 | 0.433 | 0.470 | 100.5 ms |
| GT 710 | Kepler | OpenCL1.2+CLBlast | 49.8 min | 69.4 | n/a | 227 MB | 51 ℃ | 0.941 (2250) | 0.591 | 0.897 | 0.192 | 0.358 | 0.234 | 0.346 | 0.510 | 0.716 | 0.470 | 0.460 | 25.9 ms |
| GT 730 | Kepler | OpenCL1.2+CLBlast | 32.4 min | 109.3 | n/a | 452 MB | 55 ℃ | 0.913 (2000) | 0.593 | 0.894 | 0.191 | 0.394 | 0.234 | 0.366 | 0.492 | 0.685 | 0.453 | 0.453 | 28.1 ms |
| WX 2100 | Polaris | rusticl(OpenCL3.0)+CLBlast | 43.4 min | 83.0 | **0.2205（実測 15.5 W）** | 466 MB | 66 ℃ | 0.915 (2500) | 0.584 | 0.880 | 0.201 | 0.332 | 0.212 | 0.320 | 0.504 | 0.748 | 0.493 | 0.503 | 41.7 ms |

- **MASSIVE test Teacher 一致率（最重要指標）: mean 0.363, sd 0.026（n=4）**。random 0.262。
- random: JCQA 0.200、synth 0.349。Teacher gold: JCQA 0.770, MASSIVE 0.794, W2C 0.512, synth 0.736。
- GPU 間の品質差は OpenCL float atomic の非決定性による run 間ばらつき（±0.03 程度）の範囲。samples/s は Celeron G3930 2 コアの CPU 律速（WX2100 の GPU busy ≈ 1%）で、Teacher 埋め込み抽出と同時実行の影響を含む。
- J/sample は amdgpu sysfs の GPU 単体電力。GeForce(390) は電力取得不可。
