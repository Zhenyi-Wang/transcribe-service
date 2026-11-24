# 2026-11-24 B 站 Cookie 本地持有与自同步

## 背景

2026-11-23 启用队列模式后发现：`/jobs` 载荷携带的 bilibili.cookie 在服务端任务对象创建时被剥离（回显安全），执行器却从同一对象取参——**异步下载实际全程无登录态**。充电专属视频（未登录仅试看）与风控窗口直接挂（BV1zGyZzxEFv 事故）。同时证实：带 Cookie 的下载路径完全正常（同视频手动带 Cookie 一次成功，全片 2481.88s）。

## 方案（用户拍板：本地持有 + 需要时自同步，不做逐次传参）

真值源仍是生产 NoteFlowConfig（update-bilibili-cookie 技能更新它），transcribe-service 本地持有副本：

```
Edge 登录态 → update-bilibili-cookie 技能 → 生产 NoteFlowConfig
    →（服务自同步）→ bili_cookie.local（本地副本，chmod 600，不入 git）
```

### 组件

- **bili_cookie_manager.py**：`load/save_cookie`（文件+chmod 600）、`sync_from_production`（`ssh mryk curl 127.0.0.1:10002/api/noteflow/config/get` 解析 `data.bilibili.cookie`；Cookie 值永不写日志）、`probe_login`（nav 接口，干净 JSON）、`ensure_fresh_cookie`（下载失败自愈入口）、`bili_cookie_probe_loop`（每日探测，失效当天自愈）
- **执行器接线**（server.py `_exec_bilibili`）：载荷无 Cookie 时读本地副本；下载失败 → 自愈（探测区分登录类/风控类）→ 有效即带 Cookie 重试一次；自愈被禁用（返回空）时保持原有失败语义（测试桩依赖）
- **启动钩子**：本地副本缺失即同步一次（失败不阻断启动）
- **scripts/refresh_bili_cookie.sh**：无人值守刷新——`claude -p /update-bilibili-cookie`（提示词要求浏览器未登录直接非零退出，不等待），成功后重同步+验证；依然失败 tellme。服务侧 6h 防抖触发（detached 子进程，日志 logs/bili_refresh.log）

### 自愈决策表

| 现有 Cookie 探测 | 同步后探测 | 动作 |
|---|---|---|
| 有效 | — | 原样重试（失败与登录态无关，疑似风控） |
| 失效 | 有效 | 带新 Cookie 重试 |
| 失效 | 失效 | 触发无人值守刷新（6h 防抖），本次任务报错，脚本失败时 tellme |

## 测试与部署

- 443 项全绿（新增 bili_cookie_manager 单测 16 项全离线 + /jobs 集成回归 3 项：本地加载断言/自愈重试/禁用语义）
- `/jobs` BilibiliSubmit 模型移除 cookie 必填字段（载荷停传，pydantic 默认忽略多余字段）
- mryk24 侧同步提交：BilibiliJobParams 删 cookie、buildJobBody 不下发（f58dbf7，已部署）
- 生产活体验证：无 Cookie 载荷任务 → 下载失败 → `cookie 有效，重试一次`（本地副本被正确加载并通过 nav 探测）
- 单测日志噪音说明：pytest 会向 repo logs/bili_cookie.log 写测试桩日志（无 chdir 的用例），与 server.log 既有现象一致，属既有模式未处理

## 已知边界

- 每日探测与自愈都依赖 nav 接口可达；网络异常按未登录处理（会触发一次无谓同步，幂等无害）
- 刷新脚本依赖 Windows 侧 Edge（C:\ChromeMCP\9222 配置文件）登录态有效；其过期时技能按规程退出 → tellme 人工兜底
- 触发刷新的阈值是"下载失败且登录探测失效"，对风控窗口（Cookie 有效）不会误触发

## 关联

- 事故与根因链：mryk24 `docs/2026-11-23_noteflow失败任务排查与归档套娃修复.md`
- Fable 评审（Cookie 丢失发现 + 重试分类建议）：本轮实施其第 1 步；错误分类穿两机与 playurl 回退按真实失败率再议
