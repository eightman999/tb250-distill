#!/usr/bin/env bash
# SEMFIX 本番比較の再開（10/06 21:38 に外部から SIGTERM で 2 本が中断 -> last ckpt から --resume。残り 2 本は新規）。
cd "$(dirname "$0")/.."
( RESUME=1 scripts/semfix_prod.sh rkd_pca gt730 "GT 730" --sem-loss rkd --sem-rkd-space pca ; \
  scripts/semfix_prod.sh infonce_rkd gt730 "GT 730" --sem-loss infonce+rkd ) &
( RESUME=1 scripts/semfix_prod.sh infonce_rkd gt710 "GT 710" --sem-loss infonce+rkd ; \
  scripts/semfix_prod.sh rkd_pca gt710 "GT 710" --sem-loss rkd --sem-rkd-space pca ) &
wait
