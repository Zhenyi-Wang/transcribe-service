@echo off
chcp 65001 >nul
title 暂停转录服务
echo ==========================================
echo   转录服务暂停（到时自动恢复）
echo ==========================================
echo.
set /p HOURS=请输入暂停小时数，支持小数 [直接回车=2]：

if "%HOURS%"=="" set HOURS=2

echo %HOURS%|findstr /r "^[0-9][0-9.]*$" >nul
if errorlevel 1 (
    echo 输入无效：%HOURS%
    pause
    exit /b 1
)

echo.
echo 正在暂停服务...
curl.exe -s -X POST http://localhost:31080/pause -H "Authorization: Bearer <TOKEN>" -H "Content-Type: application/json" -d "{\"hours\": %HOURS%}"
echo.
echo.
echo （paused=true 即成功；本地释放按 local_release 字段说明在后台完成，最终状态可查 /status）
pause
