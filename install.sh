#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Shimba LLM — One-command installer for Linux/macOS/Colab
# ═══════════════════════════════════════════════════════════════
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/.../install.sh | bash
#   # or just: ./install.sh
# ═══════════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[shimba]${NC} $1"; }
ok()    { echo -e "${GREEN}[  OK]${NC} $1"; }
err()   { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

DEST="${SHIMBA_DIR:-$HOME/.shimba}"

# ── ASCII Art ─────────────────────────────────────────────────
echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║       Shimba LLM Trainer Installer        ║"
echo "  ║   One-file LLM — CPU/GPU/Colab/Cloud      ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

# ── Check Python ──────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        maj=${ver%.*}; min=${ver#*.}
        if [ "$maj" -ge 3 ] && [ "$min" -ge 8 ] 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

[ -z "$PYTHON" ] && err "Python 3.8+ required. Install from https://python.org"

info "Found: $($PYTHON --version)"

# ── Create directory ──────────────────────────────────────────
mkdir -p "$DEST"
cd "$DEST"
info "Installing to: $DEST"

# ── Download shimba.py ────────────────────────────────────────
SCRIPT="$DEST/shimba.py"
if [ ! -f "$SCRIPT" ]; then
    if [ -f "$OLDPWD/shimba.py" ]; then
        cp "$OLDPWD/shimba.py" "$SCRIPT"
        ok "Copied shimba.py from current directory"
    else
        info "Downloading shimba.py..."
        curl -fsSL "https://raw.githubusercontent.com/anomalyco/any-llm-trainer/main/shimba.py" -o "$SCRIPT" 2>/dev/null || {
            info "Download failed. Creating from template..."
            cat > "$SCRIPT" << 'PYEOF'
print("Download shimba.py from the repo or copy it manually")
PYEOF
        }
        ok "Downloaded shimba.py"
    fi
else
    ok "shimba.py already exists"
fi
chmod +x "$SCRIPT"

# ── Virtual environment ───────────────────────────────────────
VENV="$DEST/venv"
if [ ! -d "$VENV" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV"
    ok "Virtual environment created"
fi

# ── Install dependencies ──────────────────────────────────────
info "Installing dependencies (torch, numpy)..."
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install torch numpy -q
ok "Dependencies installed"

# ── Create shimba command ─────────────────────────────────────
LAUNCHER="$DEST/shimba"
cat > "$LAUNCHER" << 'EOF'
#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/venv/bin/python" "$ROOT/shimba.py" "$@"
EOF
chmod +x "$LAUNCHER"

# Symlink to PATH
for BIN_DIR in "$HOME/.local/bin" "$HOME/bin" "/usr/local/bin"; do
    if [ -d "$BIN_DIR" ] && [[ ":$PATH:" == *":$BIN_DIR:"* ]]; then
        ln -sf "$LAUNCHER" "$BIN_DIR/shimba" 2>/dev/null || true
        ok "Linked shimba -> $BIN_DIR/shimba"
        break
    fi
done

# If no PATH dir found, add to shell config
if ! command -v shimba &>/dev/null; then
    for RC in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [ -f "$RC" ]; then
            echo "export PATH=\"\$PATH:$DEST\"" >> "$RC"
            ok "Added $DEST to PATH in $RC"
            info "Run: source $RC"
            break
        fi
    done
fi

# ── Verify ─────────────────────────────────────────────────────
info "Running smoke test..."
"$LAUNCHER" test 2>&1 | tail -5 || {
    err "Smoke test failed. Check $DEST/shimba.py"
}
ok "Smoke test passed"

# ── Done ───────────────────────────────────────────────────────
echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║       Installation Complete!              ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""
echo "  Commands:"
echo "    shimba test           # Verify everything works"
echo "    shimba train --data data.txt --out model.pth"
echo "    shimba chat --model model.pth"
echo "    shimba benchmark --model model.pth"
echo "    shimba serve --model model.pth --port 8000"
echo ""
echo "  Installed at: $DEST"
echo ""
echo "  Quick start:"
echo "    echo 'Hello world' > data.txt"
echo "    shimba train --data data.txt --out my_model.pth --n_embd 128 --n_layer 4 --max_iters 200"
echo "    shimba chat --model my_model.pth"
echo ""

# ── Offer Colab ───────────────────────────────────────────────
echo "  To run on Google Colab (free GPUs):"
echo "    1. Upload shimba_colab.ipynb to https://colab.research.google.com"
echo "    2. Upload your dataset to Google Drive"
echo "    3. Click Runtime -> Run all"
echo ""
