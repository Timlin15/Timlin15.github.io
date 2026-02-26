#!/bin/bash
# 开机后检查是否错过了备份，如果错过则执行备份

VAULT_PATH="/home/matrix/Documents/Obsidian Vault"
LAST_PUSH_FILE="$VAULT_PATH/.scripts/.last_push"
BACKUP_SCRIPT="$VAULT_PATH/.scripts/obsidian-backup.sh"
LOG_FILE="$VAULT_PATH/.scripts/backup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 检查上次备份时间
if [ -f "$LAST_PUSH_FILE" ]; then
    LAST_PUSH=$(cat "$LAST_PUSH_FILE")
    NOW=$(date +%s)
    DIFF=$((NOW - LAST_PUSH))

    # 如果超过25小时没有备份（允许1小时误差）
    if [ $DIFF -gt 90000 ]; then
        log "检测到错过备份，上次备份距今 $((DIFF / 3600)) 小时，执行补充备份"
        "$BACKUP_SCRIPT"
    else
        log "开机检查：上次备份在 $((DIFF / 3600)) 小时前，无需补充备份"
    fi
else
    log "首次运行，执行备份"
    "$BACKUP_SCRIPT"
fi
