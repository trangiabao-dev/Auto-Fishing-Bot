@echo off
title Auto Fishing Tool
cd /d "%~dp0"

:: 1. Uu tien duong dan Python tren may hien tai
if exist "D:\APPCODE\Python\pythonw.exe" (
    start "" "D:\APPCODE\Python\pythonw.exe" app_gui.py
    exit
)

:: 2. Tu dong tim pythonw trong PATH he thong (cho may ban be)
where pythonw >nul 2>nul
if %ERRORLEVEL% equ 0 (
    start "" pythonw app_gui.py
    exit
)

:: 3. Fallback sang python thuong
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    start "" python app_gui.py
    exit
)

:: 4. Thong bao neu may ban be chua co Python
echo ========================================================
echo   [LOI] May tinh chua cai dat Python!
echo ========================================================
echo Vui long tai va cai dat Python tu: https://www.python.org/
echo (Chu y: Nho tick chon vao o 'Add Python to PATH' khi cai).
echo ========================================================
pause
exit
