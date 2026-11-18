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
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

cd "$ROOT_DIR"

if [ ! -d "$VENV_DIR" ]; then
  echo "[Helios] venv not found. Running bootstrap first..."
  "$ROOT_DIR/scripts/bootstrap.sh"
fi

source "$VENV_DIR/bin/activate"

if [ -f "$ROOT_DIR/frontend/package.json" ] && [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[Helios] Installing missing frontend dependencies..."
    (cd "$ROOT_DIR/frontend" && npm install)
  else
    echo "[Helios] npm is required to run the frontend."
    exit 1
  fi
fi

if [ "${HELIOS_PREPARE_ASSETS:-1}" = "1" ]; then
  echo "[Helios] Preparing datasets, models, RAG assets, and GNN checkpoint..."
  python "$ROOT_DIR/scripts/prepare_startup_assets.py"
else
  echo "[Helios] Skipping asset preparation (HELIOS_PREPARE_ASSETS=${HELIOS_PREPARE_ASSETS})"
fi

echo "[Helios] Starting API on http://localhost:${BACKEND_PORT}"
python -m uvicorn helios.ui.api_server_v2:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

cleanup() {
  echo ""
  echo "[Helios] Stopping services..."
  kill "$BACKEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[Helios] Starting frontend on http://localhost:${FRONTEND_PORT}"
cd "$ROOT_DIR/frontend"
npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT"
