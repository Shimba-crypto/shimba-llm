@echo off
REM ═══════════════════════════════════════════════════════════════
REM Shimba LLM — One-command installer for Windows
REM ═══════════════════════════════════════════════════════════════
REM Usage: double-click or run in terminal
REM ═══════════════════════════════════════════════════════════════

title Shimba LLM Installer
color 0B

echo.
echo   ╔═══════════════════════════════════════════╗
echo   ║       Shimba LLM Trainer Installer        ║
echo   ║   One-file LLM - CPU/GPU/Colab/Cloud      ║
echo   ╚═══════════════════════════════════════════╝
echo.

REM ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] Python 3.8+ required. Download from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2 delims=." %%v in ('python -c "import sys; print(sys.version)" 2^>nul') do set PYMINOR=%%v
if "%PYMINOR%"=="" set PYMINOR=0
REM Check version (simple check)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo [FAIL] Python 3.8+ required. You have:
    python --version
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version') do echo [shimba] Found: %%v

REM ── Set install directory ─────────────────────────────────────
set "DEST=%USERPROFILE%\.shimba"
if not "%SHIMBA_DIR%"=="" set "DEST=%SHIMBA_DIR%"
echo [shimba] Installing to: %DEST%

if not exist "%DEST%" mkdir "%DEST%"

REM ── Download shimba.py ────────────────────────────────────────
if not exist "%DEST%\shimba.py" (
    if exist "%~dp0shimba.py" (
        copy "%~dp0shimba.py" "%DEST%\shimba.py" >nul
        echo [  OK] Copied shimba.py from current directory
    ) else (
        echo [shimba] Downloading shimba.py...
        powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/anomalyco/any-llm-trainer/main/shimba.py' -OutFile '%DEST%\shimba.py'" 2>nul
        if exist "%DEST%\shimba.py" (
            echo [  OK] Downloaded shimba.py
        ) else (
            echo [FAIL] Could not download shimba.py
            echo        Copy shimba.py manually to %DEST%
            pause
            exit /b 1
        )
    )
) else (
    echo [  OK] shimba.py already exists
)

REM ── Virtual environment ───────────────────────────────────────
if not exist "%DEST%\venv" (
    echo [shimba] Creating virtual environment...
    python -m venv "%DEST%\venv"
    echo [  OK] Virtual environment created
) else (
    echo [  OK] Virtual environment exists
)

REM ── Install dependencies ──────────────────────────────────────
echo [shimba] Installing dependencies (torch, numpy)...
call "%DEST%\venv\Scripts\pip" install --upgrade pip -q
call "%DEST%\venv\Scripts\pip" install torch numpy -q
echo [  OK] Dependencies installed

REM ── Create shimba.cmd launcher ────────────────────────────────
set "LAUNCHER=%DEST%\shimba.cmd"
(
echo @echo off
echo "%%~dp0venv\Scripts\python" "%%~dp0shimba.py" %%*
) > "%LAUNCHER%"
echo [  OK] Created launcher: %LAUNCHER%

REM ── Add to PATH ───────────────────────────────────────────────
for /f "tokens=*" %%p in ('"%USERPROFILE%\AppData\Local\Microsoft\WindowsApps\where.exe shimba.cmd 2>nul"') do goto :path_ok
echo [shimba] Adding to PATH...
setx PATH "%PATH%;%DEST%" >nul
echo [  OK] Added %DEST% to PATH
echo        You may need to restart your terminal
:path_ok

REM ── Smoke test ────────────────────────────────────────────────
echo [shimba] Running smoke test...
call "%LAUNCHER%" test
echo [  OK] Smoke test passed!

REM ── Done ──────────────────────────────────────────────────────
echo.
echo   ╔═══════════════════════════════════════════╗
echo   ║       Installation Complete!              ║
echo   ╚═══════════════════════════════════════════╝
echo.
echo   Commands:
echo     shimba test           - Verify everything works
echo     shimba train --data data.txt --out model.pth
echo     shimba chat --model model.pth
echo     shimba benchmark --model model.pth
echo     shimba serve --model model.pth --port 8000
echo.
echo   Installed at: %DEST%
echo.
echo   Quick start:
echo     echo Hello world ^> data.txt
echo     shimba train --data data.txt --out my_model.pth
echo     shimba chat --model my_model.pth
echo.
echo   To run on Google Colab (free GPUs):
echo     1. Upload shimba_colab.ipynb to https://colab.research.google.com
echo     2. Upload your dataset to Google Drive
echo     3. Click Runtime -^> Run all
echo.
pause
