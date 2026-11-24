#!/usr/bin/env bash
# 无人值守 B 站 Cookie 刷新：claude -p 执行 /update-bilibili-cookie 技能，
# 从 Windows 侧 Edge 登录态提取完整 Cookie（含 HttpOnly SESSDATA）推到生产。
# 依然失败才 tellme。由 transcribe-service 触发（防抖在服务侧），本脚本自带
# flock 防并发 + 超时兜底。日志: logs/bili_refresh.log
set -u

DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$DIR/logs/bili_refresh.log"
export HOME=/home/zhenyi
export PATH="/home/zhenyi/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

echo "=== 无人值守刷新开始: $(date '+%F %T') ===" >> "$LOG"

timeout 1500 /home/zhenyi/.local/bin/claude -p "/update-bilibili-cookie 执行要求：无人值守模式，禁止交互等待。按 SKILL.md 流程执行：起 Edge 调试口 → Playwright 检查 B 站登录态 → 用 context.cookies() 提取完整 Cookie（含 HttpOnly）→ 推到生产 NoteFlowConfig。若浏览器登录态已失效（isLogin=false），不要等待用户登录，立即输出「浏览器登录态过期，需人工登录」并直接退出（非零退出码）。" --dangerously-skip-permissions >> "$LOG" 2>&1
rc=$?

if [ $rc -eq 0 ]; then
  echo "=== 技能执行成功: $(date '+%F %T') ===" >> "$LOG"
  # 从生产同步到本地副本并验证登录态
  cd "$DIR"
  /home/zhenyi/miniconda3/envs/funasr/bin/python -c "
import bili_cookie_manager as m
ok = m.sync_from_production()
valid = m.probe_login(m.load_cookie())
print('同步:', ok, '登录态:', valid)
exit(0 if (ok and valid) else 1)
" >> "$LOG" 2>&1
  sync_rc=$?
  if [ $sync_rc -eq 0 ]; then
    echo "=== 刷新完成，本地副本已更新且登录有效: $(date '+%F %T') ===" >> "$LOG"
    exit 0
  fi
  echo "=== 技能成功但本地副本验证失败: $(date '+%F %T') ===" >> "$LOG"
  /usr/local/bin/tellme "【告警】B站Cookie自动刷新部分成功但本地验证失败，请检查 logs/bili_refresh.log 与生产配置" || true
  exit 1
fi

echo "=== 刷新失败（退出码 $rc，可能浏览器登录态过期）: $(date '+%F %T') ===" >> "$LOG"
/usr/local/bin/tellme "【告警】B站Cookie无人值守刷新失败（浏览器登录态可能过期），请在 Edge 里重新登录 B 站后手动跑 /update-bilibili-cookie，详见 church 与 transcribe 的 bili_refresh.log" || true
exit 1
