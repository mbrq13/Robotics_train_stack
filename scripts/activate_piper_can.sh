#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || "${1:-}" == "--help" ]]; then
  echo "Usage: scripts/activate_piper_can.sh LEFT_USB_BUS_INFO RIGHT_USB_BUS_INFO" >&2
  echo "Example: scripts/activate_piper_can.sh 1-5:1.0 1-6:1.0" >&2
  exit 2
fi

left_bus=$1
right_bus=$2
if [[ "$left_bus" == "$right_bus" ]]; then
  echo "The left and right CAN adapters need different USB bus-info values." >&2
  exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Install the station environment first: ./install.sh station dream ui dev" >&2
  exit 1
fi

sdk_script=$("$python_bin" -c 'from pathlib import Path; import piper_sdk; print(Path(piper_sdk.__file__).with_name("can_activate.sh"))')
if [[ ! -f "$sdk_script" ]]; then
  echo "piper_sdk/can_activate.sh is missing; reinstall the station extra." >&2
  exit 1
fi

# This configures kernel CAN links; the SDK script does not create a Python env.
sudo bash "$sdk_script" can0 1000000 "$left_bus"
sudo bash "$sdk_script" can1 1000000 "$right_bus"
ip -details link show can0
ip -details link show can1
