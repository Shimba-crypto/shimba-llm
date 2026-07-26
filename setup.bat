@echo off
REM Shimba LLM Trainer — Universal Setup (Windows)

echo === Shimba LLM Trainer Setup ===
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.8+ from python.org
    pause
    exit /b 1
)

REM Create venv
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM Install
echo Installing dependencies (torch, numpy)...
pip install --upgrade pip -q
pip install torch numpy -q

echo.
echo === Setup complete! ===
echo.
echo Quick test:  python shimba.py test
echo Train:       python shimba.py train --data data.txt --out model.pth
echo Chat:        python shimba.py chat --model model.pth
echo Benchmark:   python shimba.py benchmark --model model.pth
echo Serve API:   python shimba.py serve --model model.pth --port 8000
pause
