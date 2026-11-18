#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${HELIOS_VENV:-}" ]; then
  VENV_DIR="$ROOT_DIR/${HELIOS_VENV}"
elif [ -d "$ROOT_DIR/venv" ]; then
  VENV_DIR="$ROOT_DIR/venv"
elif [ -d "$ROOT_DIR/.venv" ]; then
  VENV_DIR="$ROOT_DIR/.venv"
else
  VENV_DIR="$ROOT_DIR/venv"
fi

cd "$ROOT_DIR"

echo "[Helios] Bootstrapping environment..."

if ! command -v python3 >/dev/null 2>&1; then
  echo "[Helios] python3 is required but not found."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[Helios] Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

if [ -f "$ROOT_DIR/frontend/package.json" ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[Helios] Installing frontend dependencies..."
    (cd "$ROOT_DIR/frontend" && npm install)
  else
    echo "[Helios] npm not found. Frontend dependencies were not installed."
    echo "[Helios] Install Node.js 18+ and run: cd frontend && npm install"
  fi
fi

echo "[Helios] Bootstrap completed successfully."
