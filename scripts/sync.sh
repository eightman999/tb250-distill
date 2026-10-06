#!/usr/bin/env bash
# Mac -> tb250 の一方向同期（DESIGN.md 準拠）。--delete は使わない・渡せない。
# 使い方: scripts/sync.sh [--dry-run]
#   tb250 に rsync が無い場合（2026-10-06 時点で無い。apt 変更禁止）は、
#   tar | ssh tar x に自動フォールバックする（上書きのみ・削除なし。--dry-run は一覧表示のみ）。
set -euo pipefail

DRY=0
RSYNC_ARGS=()
for a in "$@"; do
  case "$a" in
    --delete*|--del|--remove-source-files)
      echo "sync.sh: $a は禁止（一方向・非破壊同期のみ）" >&2; exit 2 ;;
    -n|--dry-run) DRY=1; RSYNC_ARGS+=("$a") ;;
    *) RSYNC_ARGS+=("$a") ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# 接続先 IP は git 管理外の scripts/host.local（例: TB250_HOSTNAME=100.x.y.z）か環境変数で与える。
# 未設定なら ~/.ssh/config の Host tb250 の設定に従う。
[ -f "$ROOT/scripts/host.local" ] && . "$ROOT/scripts/host.local"
SSH_OPTS=(-o BatchMode=yes)
[ -n "${TB250_HOSTNAME:-}" ] && SSH_OPTS+=(-o "HostName=$TB250_HOSTNAME")

if ssh "${SSH_OPTS[@]}" tb250 'command -v rsync >/dev/null 2>&1'; then
  rsync -a \
    --exclude .git --exclude .venv --exclude runs --exclude data --exclude __pycache__ \
    -e "ssh ${SSH_OPTS[*]}" \
    ${RSYNC_ARGS[@]+"${RSYNC_ARGS[@]}"} ./ tb250:~/tb250-distill/
  echo "sync.sh: done via rsync ($ROOT -> tb250:~/tb250-distill/)"
else
  # runs/data はトップのみ除外（tb250distill/data パッケージは送る）。
  # bsdtar の --exclude は階層に関係なく名前一致するため、トップ階層の送信対象を列挙して渡す。
  TOP=()
  while IFS= read -r f; do TOP+=("$f"); done < <(find . -mindepth 1 -maxdepth 1 \
      ! -name .git ! -name .venv ! -name runs ! -name data ! -name .DS_Store | sort)
  TAR_EX=(--exclude .git --exclude .venv --exclude __pycache__ --exclude .DS_Store)
  if [ "$DRY" = 1 ]; then
    COPYFILE_DISABLE=1 tar -c "${TAR_EX[@]}" -f - "${TOP[@]}" | tar -tf - | sed 's/^/would send: /'
    echo "sync.sh: dry-run (tar fallback)"
  else
    COPYFILE_DISABLE=1 tar -c "${TAR_EX[@]}" -f - "${TOP[@]}" \
      | ssh "${SSH_OPTS[@]}" tb250 'mkdir -p ~/tb250-distill && tar -x --warning=no-unknown-keyword -C ~/tb250-distill -f -'
    echo "sync.sh: done via tar fallback ($ROOT -> tb250:~/tb250-distill/)"
  fi
fi
