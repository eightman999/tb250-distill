#!/usr/bin/env bash
# SEMFIX スクリーニングの seed=1 反復（ノイズ見積り）。tb250 上で: setsid nohup scripts/launch_semfix_s1.sh > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
SEED=1 setsid nohup scripts/semfix_queue.sh gt730s1 "GT 730" cos_ind rkd > runs/semfix/screen/q_gt730s1.log 2>&1 < /dev/null &
# GT 710 は旧キューの終了後に別途: SEED=1 scripts/semfix_queue.sh gt710s1 "GT 710" infonce_rkd
SEED=1 RUSTICL_ENABLE=radeonsi setsid nohup scripts/semfix_queue.sh wx2100s1 "WX 2100" rkd_pca > runs/semfix/screen/q_wx2100s1.log 2>&1 < /dev/null &
wait
