#!/bin/bash
# Obsidian Vault 自动备份脚本

VAULT_PATH="/home/matrix/Documents/Obsidian Vault"
LOG_FILE="$VAULT_PATH/.scripts/backup.log"
LAST_PUSH_FILE="$VAULT_PATH/.scripts/.last_push"

cd "$VAULT_PATH" || exit 1

# 记录日志
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 检查是否有变更
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    log "没有变更，跳过备份"
    exit 0
fi

# 生成 commit 标题
DATE=$(date '+%Y-%m-%d %H:%M')
CHANGED_FILES=$(git status --porcelain | wc -l)
COMMIT_MSG="Auto backup: $DATE | $CHANGED_FILES file(s) changed"

# 执行 git 操作
git add -A
git commit -m "$COMMIT_MSG"

if git push origin main; then
    log "备份成功: $COMMIT_MSG"
    date +%s > "$LAST_PUSH_FILE"
else
    log "备份失败: push 出错"
    exit 1
fi
