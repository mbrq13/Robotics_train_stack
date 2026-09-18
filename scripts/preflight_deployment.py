"""Verify a station, worker and checkpoint contract before hardware is armed.

This script reads configuration and checkpoint metadata only. It does not open
CAN devices, cameras or model weights, and it never sends a robot action.
"""

from __future__ import annotations

import argparse
import json

from robotics_stack.evaluation.preflight import check_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deployment contracts without hardware")
    parser.add_argument("--station-config", default="configs/piper_station.yaml")
    parser.add_argument("--worker-config", default="configs/policy_worker.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    result = check_deployment(args.station_config, args.worker_config, args.checkpoint)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
