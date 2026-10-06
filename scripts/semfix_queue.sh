#!/usr/bin/env bash
# SEMFIX スクリーニング（sem-only）。使い方: scripts/semfix_queue.sh <gpu_tag> "<OpenCL デバイス名>" <条件名>...
# 例: nohup scripts/semfix_queue.sh gt730 "GT 730" cos_jb mse > runs/semfix/screen/q_gt730.log 2>&1 &
# SEED=1 を環境変数で渡すと seed を変える（TAG には s1 等を付けて run dir を分ける: 例 gt730s1）。
# 既存の run dir は上書きしない（あればスキップ）。各 run の終了後に diag_sem（MASSIVE val の unseen、np CPU）を best(=最終) と p050 で実行する。
set -u
cd "$(dirname "$0")/.."
TAG="$1"; DEV="$2"; shift 2
EMB=data/emb/qwen3_emb_0p6b/pubA_notrunc
COMMON=(--backend cl --device "$DEV" --data data/tok/pubA --lp 256 --init runs/common/init.npz --sem-emb "$EMB" \
        --sem-cand-weight 1.0 --sem-batch 128 --max-steps 3000 --seed "${SEED:-0}" --log-every 100)
cond_args() {
  case "$1" in
    cos_jb)      echo "--sem-batch-mix items:32 --sem-loss cos" ;;
    cos_ind)     echo "--sem-loss cos" ;;
    mse)         echo "--sem-loss mse" ;;
    rkd)         echo "--sem-loss rkd" ;;
    infonce)     echo "--sem-loss infonce" ;;
    infonce_rkd) echo "--sem-loss infonce+rkd" ;;
    infonce_pcac) echo "--sem-loss infonce --sem-nce-target pca_c" ;;
    infonce_t03) echo "--sem-loss infonce --sem-tau-nce 0.03" ;;
    infonce_t10) echo "--sem-loss infonce --sem-tau-nce 0.1" ;;
    rkd_pca)     echo "--sem-loss rkd --sem-rkd-space pca" ;;
    cos_mass)    echo "--sem-loss cos --sem-batch-mix massive:0.5,synth:0.25,jcqa:0.25" ;;
    infonce_rkd_mass) echo "--sem-loss infonce+rkd --sem-batch-mix massive:0.5,synth:0.25,jcqa:0.25" ;;
    *) echo "unknown" ;;
  esac
}
for C in "$@"; do
  A=$(cond_args "$C"); [ "$A" = unknown ] && { echo "unknown cond $C"; continue; }
  RD=runs/semfix/screen/$C/$TAG
  if [ -e "$RD/ckpt/last.npz" ]; then echo "skip $C ($RD exists)"; continue; fi
  mkdir -p "$RD"
  echo "== $(date +%T) start $C on $DEV"
  # shellcheck disable=SC2086
  .venv/bin/python -u -m tb250distill.student.sem_only "${COMMON[@]}" $A --run-dir "$RD" > "$RD/stdout.log" 2>&1
  echo "== $(date +%T) done $C rc=$?"
  for CK in best p050; do
    .venv/bin/python -m tb250distill.diag_sem run --ckpt "$RD/ckpt/$CK.npz" --name "${C}_${TAG}_${CK}" --providers qwen3_emb_0p6b \
      --unseen-split val --out runs/semfix/screen/diag 2>&1 | tail -1
  done
done
# 参照: 初期 weight（p000 = init + seed 決定的な head）。最初のキューの最後に 1 回だけ
if [ "$TAG" = gt730 ] && [ ! -e runs/semfix/screen/diag/ref_init_p000.json ]; then
  FIRST=$(ls -d runs/semfix/screen/*/gt730 2>/dev/null | head -1)
  [ -n "$FIRST" ] && .venv/bin/python -m tb250distill.diag_sem run --ckpt "$FIRST/ckpt/p000.npz" --name ref_init_p000 --providers qwen3_emb_0p6b \
      --unseen-split val --out runs/semfix/screen/diag 2>&1 | tail -1
fi
echo "== queue $TAG finished $(date +%T)"
