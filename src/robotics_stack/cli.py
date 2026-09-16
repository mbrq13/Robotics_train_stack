"""Stable command line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from robotics_stack.hardware.fake import FakeRobot
from robotics_stack.hardware.piper import BiPiper
from robotics_stack.learning.train import train_state_mlp
from robotics_stack.runtime.config import load_station_config
from robotics_stack.runtime.policy_agent import run_policy_agent
from robotics_stack.runtime.station import RobotStation


def _station(args: argparse.Namespace) -> None:
    schema, station_cfg, piper_cfg = load_station_config(args.config)
    robot = FakeRobot(schema) if args.fake else BiPiper(schema, piper_cfg)
    station = RobotStation(
        robot,
        schema,
        watchdog_ms=float(station_cfg.get("watchdog_ms", 400)),
        action_age_ms=float(station_cfg.get("action_age_ms", 250)),
    )
    station.connect()
    try:
        asyncio.run(_run_station_services(station, station_cfg, args.no_ui))
    finally:
        station.disconnect()


async def _run_station_services(station: RobotStation, config: dict, no_ui: bool) -> None:
    tasks = [asyncio.create_task(station.serve(config.get("host", "0.0.0.0"), int(config.get("port", 8765))))]
    if not no_ui:
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("install the ui extra or pass --no-ui") from exc
        from robotics_stack.ui.app import create_app

        server = uvicorn.Server(uvicorn.Config(create_app(station), host=config.get("ui_host", "0.0.0.0"), port=int(config.get("ui_port", 8080)), log_level="warning"))
        tasks.append(asyncio.create_task(server.serve()))
    await asyncio.gather(*tasks)


def _train(args: argparse.Namespace) -> None:
    schema, _, _ = load_station_config(args.schema)
    result = train_state_mlp(args.data, args.output, schema, epochs=args.epochs, batch_size=args.batch_size)
    print(json.dumps({"output": str(result.output), "samples": result.samples, "train_mse": result.loss}, indent=2))


def _inspect(args: argparse.Namespace) -> None:
    value = json.loads((Path(args.checkpoint) / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(value, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rstack")
    commands = parser.add_subparsers(dest="command", required=True)
    station = commands.add_parser("station", help="run the PC-side robot station")
    station.add_argument("--config", required=True)
    station.add_argument("--fake", action="store_true", help="run without CAN hardware")
    station.add_argument("--no-ui", action="store_true")
    station.set_defaults(func=_station)
    policy = commands.add_parser("policy", help="run a policy worker on the Thor")
    policy.add_argument("--config", required=True)
    policy.add_argument("--checkpoint", required=True)
    policy.set_defaults(func=lambda a: asyncio.run(run_policy_agent(_policy_url(a.config), a.checkpoint)))
    train = commands.add_parser("train", help="train the native state_mlp policy")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--schema", default="configs/piper_station.yaml")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.set_defaults(func=_train)
    inspect = commands.add_parser("inspect", help="print checkpoint manifest")
    inspect.add_argument("checkpoint")
    inspect.set_defaults(func=_inspect)
    args = parser.parse_args()
    args.func(args)


def _policy_url(path: str) -> str:
    from robotics_stack.runtime.config import load_yaml

    return str(load_yaml(path)["station_url"])
