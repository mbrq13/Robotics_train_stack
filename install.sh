#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: ./install.sh {station|so100|policy|policy-lerobot|dream|dream-replay|ui|dev|all} [...]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

create_venv() {
  if command -v uv >/dev/null 2>&1; then
    uv venv --clear --seed --python "$PYTHON_BIN" .venv
  else
    resolved_python=$(readlink -f "$(command -v "$PYTHON_BIN")")
    env -u PYTHONPATH "$resolved_python" -m venv --clear .venv
  fi
}

if [[ ! -d .venv ]]; then
  create_venv
fi

if ! env -u PYTHONPATH .venv/bin/python -c "import sys" >/dev/null 2>&1; then
  echo "Rebuilding an unusable .venv with $PYTHON_BIN"
  create_venv
fi

if ! env -u PYTHONPATH .venv/bin/python -m pip --version >/dev/null 2>&1; then
  env -u PYTHONPATH .venv/bin/python -m ensurepip --upgrade
fi

env -u PYTHONPATH .venv/bin/python -m pip install --upgrade pip
if [[ "$1" == "all" ]]; then
  env -u PYTHONPATH .venv/bin/python -m pip install -e ".[station,so100,policy-lerobot,dream,ui,dev]"
else
  extras=$(IFS=,; echo "$*")
  env -u PYTHONPATH .venv/bin/python -m pip install -e ".[${extras}]"
fi

echo "Installed. Activate with: source .venv/bin/activate"
