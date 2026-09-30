@echo off
chcp 65001 >nul
title 恢复转录服务
echo 正在恢复转录服务...
echo.
curl.exe -s -X POST http://localhost:31080/resume -H "Authorization: Bearer <TOKEN>"
echo.
echo 当前状态：
curl.exe -s http://localhost:31080/status -H "Authorization: Bearer <TOKEN>"
echo.
echo.
echo （paused=false 即已恢复；模型将在下一个转录请求时自动加载）
pause
