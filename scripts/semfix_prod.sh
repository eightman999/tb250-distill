#!/usr/bin/env bash
# SEMFIX 本番比較: Baseline A と同条件（pubA, lp256, batch32, lr2e-3, 3 epoch, eval-every 250, seed0, init runs/common/init.npz）で
# 判断と同時に学習し、best ckpt を evaluate -> diag_sem（MASSIVE test unseen）。
# 使い方: scripts/semfix_prod.sh <cond> <gpu_tag> "<OpenCL デバイス名>" <追加 train 引数...>
# 例: nohup scripts/semfix_prod.sh infonce_rkd gt730 "GT 730" --sem-loss infonce+rkd > /dev/null 2>&1 &
# run dir = runs/semfix/prod/<cond>/<gpu_tag>/（既存なら中止）。プロセス操作は PID 指定で行うこと（pkill -f 禁止）。
set -u
cd "$(dirname "$0")/.."
COND="$1"; TAG="$2"; DEV="$3"; shift 3
RD=runs/semfix/prod/$COND/$TAG
# RESUME=1: 中断した run（ckpt/last.npz あり）を --resume で再開（ログは追記）。無ければ既存 dir は中止
RES=(); OUT=">"
if [ -n "${RESUME:-}" ] && [ -e "$RD/ckpt/last.npz" ]; then RES=(--resume); else [ -e "$RD" ] && { echo "$RD exists"; exit 2; }; fi
mkdir -p "$RD"
ENVP=(); [ "$DEV" = "WX 2100" ] && ENVP=(RUSTICL_ENABLE=radeonsi)
P=.venv/bin/python
env "${ENVP[@]}" $P -u -m tb250distill.student.train --backend cl --device "$DEV" --data data/tok/pubA --lp 256 --run-dir "$RD" \
  --init runs/common/init.npz --epochs 3 --eval-every 250 --batch-size 32 --seed 0 --replay-db data/replay.sqlite --no-final-eval \
  --sem-cand-weight 0.5 --sem-emb data/emb/qwen3_emb_0p6b/pubA_notrunc --sem-batch 128 "${RES[@]}" "$@" >> "$RD/stdout.log" 2>> "$RD/stderr.log"
echo $? > "$RD/exit_code"
env "${ENVP[@]}" $P -m tb250distill.student.evaluate --backend cl --device "$DEV" --data data/tok/pubA --lp 256 --ckpt "$RD/ckpt/best.npz" \
  --splits val test robust --replay-db data/replay.sqlite --dump-preds "$RD/preds.npz" --dump-latency-n 200 --out "$RD/eval.json" > "$RD/eval.log" 2>&1
echo $? > "$RD/eval_exit_code"
$P -m tb250distill.diag_sem run --run-dir "$RD" --name "prod_${COND}_${TAG}" --providers qwen3_emb_0p6b --unseen-split test --out runs/semfix/prod/diag > "$RD/diag.log" 2>&1
cp runs/semfix/prod/diag/prod_${COND}_${TAG}.json "$RD/diag_test.json" 2>/dev/null
echo "finished $(date -Is)" >> "$RD/eval_exit_code"
