#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Helios] Running setup via scripts/bootstrap.sh"
"$ROOT_DIR/scripts/bootstrap.sh"

echo "[Helios] Setup complete."
echo "[Helios] Start full stack with: ./scripts/start.sh"
