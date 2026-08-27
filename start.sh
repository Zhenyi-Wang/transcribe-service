#!/usr/bin/env bash
# 统一启动入口：幂等地在 tmux 会话中运行 ./run.sh
# 与 livetrans / asr-engine 的 start.sh 同一模板，差异仅在头部变量
set -euo pipefail

SESSION="transcribe"
HEALTH_URL=""   # 就绪探测端点；留空跳过等待（server.py 无 /health 路由）
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"

# 幂等：会话已存在则不重复创建
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux 会话 '$SESSION' 已在运行，未重复启动（tmux a -t $SESSION 查看）"
    exit 0
fi

tmux new-session -d -s "$SESSION" -c "$DIR" "./run.sh"
echo "✓ 已创建 tmux 会话 '$SESSION'，后台启动中..."

# 可选：等待健康端点就绪
if [[ -n "$HEALTH_URL" ]]; then
    echo "等待服务就绪（最多 25s）..."
    for _ in $(seq 1 25); do
        if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
            echo "✓ 服务就绪: $HEALTH_URL"
            exit 0
        fi
        sleep 1
    done
    echo "⚠ 服务未在 25s 内就绪，查看输出: tmux a -t $SESSION"
    exit 1
fi
