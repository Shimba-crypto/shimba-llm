#!/bin/bash
# Shimba LLM Trainer — Universal Setup (Linux/macOS)
set -e
echo "=== Shimba LLM Trainer Setup ==="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.8+ first."
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install
echo "Installing dependencies (torch, numpy)..."
pip install --upgrade pip -q
pip install torch numpy -q

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Quick test:  python shimba.py test"
echo "Train:       python shimba.py train --data data.txt --out model.pth"
echo "Chat:        python shimba.py chat --model model.pth"
echo "Benchmark:   python shimba.py benchmark --model model.pth"
echo "Serve API:   python shimba.py serve --model model.pth --port 8000"
