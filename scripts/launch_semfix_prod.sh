#!/usr/bin/env bash
# SEMFIX 本番比較の起動（GT 730 / GT 710 に各 2 条件を直列。1 GPU 1 プロセス）。tb250 上で:
#   setsid nohup scripts/launch_semfix_prod.sh > runs/semfix/prod_launch.log 2>&1 < /dev/null &
# 条件: rkd_pca = --sem-loss rkd --sem-rkd-space pca（val 最良）、infonce_rkd = --sem-loss infonce+rkd（val 2 番手）
cd "$(dirname "$0")/.."
mkdir -p runs/semfix/prod
( scripts/semfix_prod.sh rkd_pca gt730 "GT 730" --sem-loss rkd --sem-rkd-space pca ; \
  scripts/semfix_prod.sh infonce_rkd gt730 "GT 730" --sem-loss infonce+rkd ) &
( scripts/semfix_prod.sh infonce_rkd gt710 "GT 710" --sem-loss infonce+rkd ; \
  scripts/semfix_prod.sh rkd_pca gt710 "GT 710" --sem-loss rkd --sem-rkd-space pca ) &
wait
