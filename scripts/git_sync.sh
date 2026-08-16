#!/usr/bin/env bash
# 通过 ssh git 把本地代码推送到服务器（只推代码，数据/断点/日志全被 .gitignore 排除）
# 用法:
#   AGENTOPSD_GIT_REMOTE=ssh://user@server/~/git/AgentOPSD.git bash scripts/git_sync.sh
#   或先在本地配置: git remote add origin ssh://user@server/~/git/AgentOPSD.git
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

cd "${AGENTOPSD_REPO_ROOT}"

if [[ -z "$(git remote get-url origin 2>/dev/null || true)" ]]; then
  if [[ -z "${AGENTOPSD_GIT_REMOTE:-}" ]]; then
    die "未配置 remote。请先执行:\n  git remote add origin ssh://<user>@<server>/~/git/AgentOPSD.git\n或设置 AGENTOPSD_GIT_REMOTE"
  fi
  git remote add origin "${AGENTOPSD_GIT_REMOTE}"
fi

git add -A
if git diff --cached --quiet; then
  log "本地没有待提交的代码改动，直接 push"
else
  git commit -m "sync: $(date '+%F %T')"
fi
git push -u origin HEAD
log "==> 已推送到服务器。在服务器端: git clone <remote> && bash scripts/setup_server.sh"
