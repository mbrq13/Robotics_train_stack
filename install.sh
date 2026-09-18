#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: ./install.sh {station|policy|ui|dev|all} [...]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if ! .venv/bin/python -c "import sys" >/dev/null 2>&1; then
  echo "Rebuilding an unusable .venv with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv --clear .venv
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade
fi

.venv/bin/python -m pip install --upgrade pip
if [[ "$1" == "all" ]]; then
  .venv/bin/python -m pip install -e ".[station,policy-lerobot,ui,dev]"
else
  extras=$(IFS=,; echo "$*")
  .venv/bin/python -m pip install -e ".[${extras}]"
fi

echo "Installed. Activate with: source .venv/bin/activate"
