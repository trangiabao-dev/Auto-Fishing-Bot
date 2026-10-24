@echo off
title Cai dat thu vien - Auto Fishing Tool
cd /d "%~dp0"

echo ========================================================
echo   DANG CAI DAT THU VIEN CAN THIET CHO AUTO FISHING TOOL
echo ========================================================
echo.

:: 1. Uu tien Python rieng neu co
if exist "D:\APPCODE\Python\python.exe" (
    "D:\APPCODE\Python\python.exe" -m pip install -r requirements.txt
    goto finish
)

:: 2. Tim Python he thong
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    python -m pip install -r requirements.txt
    goto finish
)

echo [LOI] Khong tim thay Python tren may tinh cua ban!
echo Vui long cai Python va tick vao "Add Python to PATH".
pause
exit

:finish
echo.
echo ========================================================
echo   CAI DAT HOAN TAT! BAN CO THE MO "ChayTool.bat" DE CHAY.
echo ========================================================
pause
exit
