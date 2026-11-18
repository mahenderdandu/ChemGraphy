#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -n "${HELIOS_VENV:-}" ]; then
    VENV_DIR="$ROOT_DIR/${HELIOS_VENV}"
elif [ -d "$ROOT_DIR/venv" ]; then
    VENV_DIR="$ROOT_DIR/venv"
elif [ -d "$ROOT_DIR/.venv" ]; then
    VENV_DIR="$ROOT_DIR/.venv"
else
    VENV_DIR="$ROOT_DIR/venv"
fi

echo "======================================"
echo "  🔬 Helios Chemistry AI Chatbot"
echo "======================================"
echo ""

if [ ! -d "$VENV_DIR" ]; then
    echo "[Helios] venv not found. Running bootstrap first..."
    "$ROOT_DIR/scripts/bootstrap.sh"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Check dependencies
python -c "import gradio" 2>/dev/null || {
    echo "Installing gradio..."
    pip install gradio
}

# Launch chatbot
echo "Starting Helios Chatbot..."
echo "Loading models (this may take 1-2 minutes)..."
echo ""

python -m helios.ui.chatbot_app \
    --rag-dir data/rag \
    --llm TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --device cpu \
    --port 7860

echo ""
echo "Chatbot stopped."
