"""Stable command line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json

from robotics_stack.evaluation.replay import evaluate_state_mlp
from robotics_stack.hardware.cameras import CameraHub
from robotics_stack.hardware.fake import FakeRobot
from robotics_stack.hardware.piper import BiPiper
from robotics_stack.learning.pi05 import run_pi05_training
from robotics_stack.learning.train import train_state_mlp
from robotics_stack.policy.checkpoints import inspect_checkpoint
from robotics_stack.runtime.config import load_camera_configs, load_station_config, load_yaml
from robotics_stack.runtime.policy_agent import run_policy_agent
from robotics_stack.runtime.station import RobotStation


def _station(args: argparse.Namespace) -> None:
    schema, station_cfg, piper_cfg = load_station_config(args.config)
    robot = FakeRobot(schema) if args.fake else BiPiper(schema, piper_cfg)
    cameras = None if args.fake else CameraHub(load_camera_configs(load_yaml(args.config)))
    station = RobotStation(
        robot,
        schema,
        watchdog_ms=float(station_cfg.get("watchdog_ms", 400)),
        first_action_timeout_ms=float(station_cfg.get("first_action_timeout_ms", 15_000)),
        action_age_ms=float(station_cfg.get("action_age_ms", 250)),
        scheduled_action_age_ms=float(station_cfg["scheduled_action_age_ms"])
        if "scheduled_action_age_ms" in station_cfg
        else None,
        motion_mode=str(station_cfg.get("motion_mode", "direct")),
        stream_rate_hz=float(station_cfg.get("stream_rate_hz", 100)),
        cameras=cameras,
        home_action=tuple(float(value) for value in station_cfg["home_action"])
        if "home_action" in station_cfg
        else None,
    )
    station.connect()
    try:
        asyncio.run(_run_station_services(station, station_cfg, args.no_ui))
    finally:
        station.disconnect()


async def _run_station_services(station: RobotStation, config: dict, no_ui: bool) -> None:
    tasks = [
        asyncio.create_task(
            station.serve(config.get("host", "0.0.0.0"), int(config.get("port", 8765)))
        )
    ]
    if not no_ui:
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("install the ui extra or pass --no-ui") from exc
        from robotics_stack.ui.app import create_app

        server = uvicorn.Server(
            uvicorn.Config(
                create_app(station),
                host=config.get("ui_host", "0.0.0.0"),
                port=int(config.get("ui_port", 8080)),
                log_level="warning",
            )
        )
        tasks.append(asyncio.create_task(server.serve()))
    await asyncio.gather(*tasks)


def _train(args: argparse.Namespace) -> None:
    schema, _, _ = load_station_config(args.schema)
    result = train_state_mlp(
        args.data, args.output, schema, epochs=args.epochs, batch_size=args.batch_size
    )
    print(
        json.dumps(
            {"output": str(result.output), "samples": result.samples, "train_mse": result.loss},
            indent=2,
        )
    )


def _inspect(args: argparse.Namespace) -> None:
    print(json.dumps(inspect_checkpoint(args.checkpoint).__dict__, indent=2))


def _train_pi05(args: argparse.Namespace) -> None:
    command = run_pi05_training(args.config, dry_run=args.dry_run)
    print(" ".join(command))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rstack")
    commands = parser.add_subparsers(dest="command", required=True)
    station = commands.add_parser("station", help="run the hardware station")
    station.add_argument("--config", required=True)
    station.add_argument("--fake", action="store_true", help="run without CAN hardware")
    station.add_argument("--no-ui", action="store_true")
    station.set_defaults(func=_station)
    policy = commands.add_parser("policy", help="run the policy worker")
    policy.add_argument("--config", required=True)
    policy.add_argument("--checkpoint", required=True)
    policy.set_defaults(func=_policy)
    train = commands.add_parser("train", help="train the native state_mlp policy")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--schema", default="configs/piper_station.yaml")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.set_defaults(func=_train)
    train_pi05 = commands.add_parser("train-pi05", help="launch validated Pi0.5 RTC training")
    train_pi05.add_argument("--config", required=True)
    train_pi05.add_argument("--dry-run", action="store_true")
    train_pi05.set_defaults(func=_train_pi05)
    inspect = commands.add_parser("inspect", help="print checkpoint manifest")
    inspect.add_argument("checkpoint")
    inspect.set_defaults(func=_inspect)
    evaluate = commands.add_parser("evaluate", help="validate a checkpoint against replay data")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.set_defaults(func=_evaluate)
    args = parser.parse_args()
    args.func(args)


def _evaluate(args: argparse.Namespace) -> None:
    report = evaluate_state_mlp(args.checkpoint, args.data)
    print(json.dumps(report.__dict__, indent=2))


def _policy(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    asyncio.run(
        run_policy_agent(
            str(config["station_url"]),
            args.checkpoint,
            task=str(config.get("task", "")),
            device=str(config.get("device", "cuda")),
            schema_version=int(config.get("schema_version", 1)),
            state_names=tuple(str(name) for name in config.get("state_names", [])),
            execution_mode=str(config.get("execution_mode", "standard")),
            rtc_execution_horizon=int(config["rtc_execution_horizon"])
            if "rtc_execution_horizon" in config
            else None,
            rtc_refill_threshold=int(config["rtc_refill_threshold"])
            if "rtc_refill_threshold" in config
            else None,
        )
    )


if __name__ == "__main__":
    main()
