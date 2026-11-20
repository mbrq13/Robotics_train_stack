#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/activate_piper_can.sh LEFT_USB_BUS_INFO RIGHT_USB_BUS_INFO
       bash scripts/activate_piper_can.sh --list
Example (confirm physical arm sides first):
  bash scripts/activate_piper_can.sh 1-3.1.1:1.0 1-3.1.2:1.0
HELP
}
if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 2 && ! ( $# -eq 1 && "$1" == "--list" ) ]]; then
  usage >&2
  exit 2
fi
for tool in ip ethtool; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing $tool. Install: sudo apt install iproute2 ethtool can-utils" >&2
    exit 1
  fi
done

# Inspect both adapters before invoking the SDK or changing any link.
mapfile -t interfaces < <(ip -br link show type can | awk '{print $1}')
if [[ ${#interfaces[@]} -eq 0 ]]; then
  echo "No CAN interfaces found. Check USB adapters and the gs_usb driver." >&2
  exit 1
fi
declare -A by_bus=()
for iface in "${interfaces[@]}"; do
  bus=$(ethtool -i "$iface" | awk '$1 == "bus-info:" {print $2}')
  printf '%s bus-info: %s\n' "$iface" "${bus:-unknown}"
  if [[ -n "$bus" ]]; then
    by_bus["$bus"]=$iface
  fi
done
if [[ "${1:-}" == "--list" ]]; then
  for iface in "${interfaces[@]}"; do
    ip -details link show "$iface"
  done
  exit 0
fi
left_bus=$1
right_bus=$2
if [[ "$left_bus" == "$right_bus" ]]; then
  echo "The left and right CAN adapters need different USB bus-info values." >&2
  exit 2
fi
for bus in "$left_bus" "$right_bus"; do
  if [[ -z "${by_bus[$bus]:-}" ]]; then
    echo "USB bus-info $bus not found. Use the detected values above, after checking arm sides." >&2
    exit 1
  fi
done
# The SDK cannot rename onto an occupied interface. Do not leave a partial mapping.
for target in can0 can1; do
  expected=$left_bus
  [[ "$target" == can1 ]] && expected=$right_bus
  if ip link show "$target" >/dev/null 2>&1 && [[ "${by_bus[$expected]}" != "$target" ]]; then
    echo "$target is occupied by a different adapter. Resolve the interface names before activation." >&2
    exit 1
  fi
done

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Find the installed shell script directly; CAN setup needs no Python imports.
shopt -s nullglob
sdk_scripts=("$repo_root"/.venv/lib/python*/site-packages/piper_sdk/can_activate.sh)
if [[ ${#sdk_scripts[@]} -ne 1 ]]; then
  echo "Expected one piper_sdk/can_activate.sh in .venv. Run: ./install.sh station dream ui dev" >&2
  exit 1
fi
sudo bash "${sdk_scripts[0]}" can0 1000000 "$left_bus"
sudo bash "${sdk_scripts[0]}" can1 1000000 "$right_bus"
# Do not trust the SDK exit status alone: its shell script can mask ip failures.
for iface in can0 can1; do
  details=$(ip -details link show "$iface")
  if [[ "$details" != *",UP,"* || "$details" != *"bitrate 1000000 "* ]]; then
    echo "$iface did not reach UP at 1000000 bit/s." >&2
    exit 1
  fi
  bus=$(ethtool -i "$iface" | awk '$1 == "bus-info:" {print $2}')
  expected=$left_bus
  [[ "$iface" == can1 ]] && expected=$right_bus
  if [[ "$bus" != "$expected" ]]; then
    echo "$iface has bus-info $bus, expected $expected." >&2
    exit 1
  fi
  printf '%s\n' "$details"
done
