"""Stable command line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from robotics_stack.config.loaders import load_camera_configs, load_station_config, load_yaml
from robotics_stack.evaluation.replay import evaluate_state_mlp
from robotics_stack.policies.checkpoints import inspect_checkpoint
from robotics_stack.robots.cameras import CameraHub
from robotics_stack.robots.fake import FakeRobot
from robotics_stack.robots.registry import create_robot_driver
from robotics_stack.services.policy_worker import run_policy_agent
from robotics_stack.services.station import RobotStation
from robotics_stack.training.pi05 import run_pi05_training
from robotics_stack.training.train import train_state_mlp


def _station(args: argparse.Namespace) -> None:
    schema, station_cfg, profile = load_station_config(args.config)
    robot = FakeRobot(schema) if args.fake else create_robot_driver(schema, profile)
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


def _simulate(args: argparse.Namespace) -> None:
    """Run an offline, visual guided-collection dry run from replayed frames."""
    from robotics_stack.policies.checkpoints import load_deployed_policy
    from robotics_stack.simulation.controls import SimulationControls
    from robotics_stack.simulation.dagger import DatasetReplay, GuidedSimulation
    from robotics_stack.simulation.operator_input import OperatorTargetReceiver
    from robotics_stack.simulation.runner import run_guided_simulation, start_terminal_controls

    schema, _, _ = load_station_config(args.schema)
    worker = load_yaml(args.config)
    replay = DatasetReplay.from_npz(args.dataset, camera_names=schema.camera_names)
    policy = load_deployed_policy(
        args.checkpoint,
        task=str(worker.get("task", "")),
        device=str(worker.get("device", "cuda")),
        state_names=tuple(str(name) for name in worker.get("state_names", [])),
        robot_type=str(worker.get("robot_type", "")),
    )
    policy.descriptor.validate_station(schema)
    simulation = GuidedSimulation(schema, replay, policy, task=args.task)
    controls = SimulationControls()
    receiver = None
    if args.operator_udp_port is not None:
        receiver = OperatorTargetReceiver(
            controls, action_size=schema.action_size, port=args.operator_udp_port
        )
        receiver.start()
        print(f"Local operator-target UDP input: 127.0.0.1:{args.operator_udp_port}")
    panel = None
    if not args.no_viser:
        from robotics_stack.simulation.viser_panel import ViserGuidancePanel

        panel = ViserGuidancePanel(schema, controls, port=args.viser_port)
        print(f"Viser controls: http://127.0.0.1:{args.viser_port}")
    if not args.no_terminal_controls:
        print("Terminal controls: p pause, t recovery, c correction, r resume, s save, q stop.")
        print("Press Enter after each terminal control.")
        start_terminal_controls(controls)
    try:
        output = run_guided_simulation(
            simulation,
            controls,
            output=args.output,
            max_steps=args.max_steps,
            panel=panel,
        )
    finally:
        if receiver is not None:
            receiver.stop()
    print(json.dumps({"output": str(output), **simulation.episode.summary()}, indent=2))


def _teleop(args: argparse.Namespace) -> None:
    """Run a manually confirmed VR correction session against a local station."""
    from robotics_stack.teleoperators.guidance.records import GuidedEpisode, GuidedSample
    from robotics_stack.teleoperators.guidance.tempo import TargetBridgeSettings
    from robotics_stack.teleoperators.vr.guided import GuidedVrHandoff
    from robotics_stack.teleoperators.vr.session import VrTeleoperator, VrTeleoperatorSettings
    from robotics_stack.teleoperators.vr.tracking import PicoTrackingProvider, QuestTrackingClient

    config = load_yaml(args.config)
    schema_path = str(config["robot_config"])
    schema, station_cfg, profile = load_station_config(schema_path)
    headset = dict(config.get("headset", {}))
    motion = dict(config.get("motion", {}))
    bridge = dict(config.get("bridge", {}))
    if str(headset.get("kind", "")) == "quest":
        provider = QuestTrackingClient(str(headset["host"]), port=int(headset.get("port", 65432)))
        provider.start()
    elif str(headset.get("kind", "")) == "pico":
        if args.pico_usb:
            PicoTrackingProvider.prepare_usb()
        provider = PicoTrackingProvider()
    else:
        raise ValueError("headset.kind must be quest or pico")
    robot = FakeRobot(schema) if args.fake else create_robot_driver(schema, profile)
    cameras = None if args.fake else CameraHub(load_camera_configs(load_yaml(schema_path)))
    station = RobotStation(
        robot,
        schema,
        action_age_ms=float(station_cfg.get("action_age_ms", 250)),
        motion_mode=str(station_cfg.get("motion_mode", "direct")),
        stream_rate_hz=float(station_cfg.get("stream_rate_hz", 100)),
        cameras=cameras,
    )
    settings = VrTeleoperatorSettings(**motion)
    teleoperator = VrTeleoperator(schema, settings=settings)
    handoff = GuidedVrHandoff(station, teleoperator, bridge_settings=TargetBridgeSettings(**bridge))
    started = False
    episode = GuidedEpisode(task=args.task, schema_version=schema.version) if args.record else None
    observation_id = 0
    try:
        station.connect()
        station.arm()
        station.start()
        station.pause("VR teleoperation preparation")
        deadline = time.monotonic() + args.connect_timeout_s
        frame = provider.poll()
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.01)
            frame = provider.poll()
        if frame is None:
            raise TimeoutError("no valid VR controller frame arrived before the timeout")
        if not args.engage:
            print("VR tracking is connected. Re-run with --engage after clearing the workcell.")
            return
        handoff.begin_correction(frame)
        started = True
        print("VR control is active. Press Ctrl+C to hold and finish this correction session.")
        until = time.monotonic() + args.duration_s if args.duration_s > 0 else None
        last_timestamp = -1
        while until is None or time.monotonic() < until:
            frame = provider.poll()
            if frame is not None and frame.timestamp_ns > last_timestamp:
                accepted = handoff.submit_tracking(frame)
                if accepted and episode is not None:
                    observation_id += 1
                    target = teleoperator.target
                    assert target is not None
                    images = {name: station.camera_frame(name) for name in schema.camera_names}
                    episode.append(
                        GuidedSample(
                            observation_id=observation_id,
                            station_monotonic_ns=time.monotonic_ns(),
                            state=station.robot.observation(),
                            action=target,
                            phase=station.supervisor.authority.snapshot.phase,
                            control_generation=station.supervisor.control_generation,
                            images=images,
                        )
                    )
                last_timestamp = frame.timestamp_ns
            time.sleep(1.0 / settings.tracking_rate_hz)
    finally:
        try:
            if started:
                handoff.finish_correction()
        finally:
            if isinstance(provider, QuestTrackingClient):
                provider.stop()
            station.disconnect()
    if episode is not None and episode.samples:
        output = episode.export_npz(args.output)
        print(json.dumps({"output": str(output), **episode.summary()}, indent=2))


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
    train.add_argument("--schema", default="configs/robots/piper_bimanual.yaml")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.set_defaults(func=_train)
    train_pi05 = commands.add_parser("train-pi05", help="launch validated Pi0.5 RTC training")
    train_pi05.add_argument("--config", required=True)
    train_pi05.add_argument("--dry-run", action="store_true")
    train_pi05.set_defaults(func=_train_pi05)
    simulate = commands.add_parser(
        "simulate", help="replay dataset frames through a policy and guided controls"
    )
    simulate.add_argument("--config", default="configs/deploy/policy_worker.yaml")
    simulate.add_argument("--schema", default="configs/robots/piper_bimanual.yaml")
    simulate.add_argument("--checkpoint", required=True)
    simulate.add_argument("--dataset", required=True, help="NPZ containing images_<camera> arrays")
    simulate.add_argument("--output", default="outputs/guided_simulation.npz")
    simulate.add_argument("--task", default="")
    simulate.add_argument("--max-steps", type=int, default=0, help="0 runs until stopped")
    simulate.add_argument("--viser-port", type=int, default=8010)
    simulate.add_argument(
        "--operator-udp-port",
        type=int,
        default=None,
        help="accept local JSON targets from a VR retargeter",
    )
    simulate.add_argument("--no-viser", action="store_true")
    simulate.add_argument("--no-terminal-controls", action="store_true")
    simulate.set_defaults(func=_simulate)
    teleop = commands.add_parser("teleop", help="teleoperate a local robot from a VR headset")
    teleop.add_argument("--config", default="configs/teleop/piper_vr.example.yaml")
    teleop.add_argument(
        "--fake", action="store_true", help="use the deterministic simulation backend"
    )
    teleop.add_argument("--engage", action="store_true", help="explicitly enable robot motion")
    teleop.add_argument(
        "--pico-usb", action="store_true", help="configure the PICO ADB tracking tunnel"
    )
    teleop.add_argument("--connect-timeout-s", type=float, default=15.0)
    teleop.add_argument("--duration-s", type=float, default=0.0, help="0 runs until interrupted")
    teleop.add_argument("--record", action="store_true", help="write the manual correction episode")
    teleop.add_argument("--output", default="outputs/vr_episode.npz")
    teleop.add_argument("--task", default="")
    teleop.set_defaults(func=_teleop)
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
            robot_type=str(config.get("robot_type", "")),
            expected_action_space=str(config.get("action_space", "")),
            execution_mode=str(config.get("execution_mode", "sync")),
            rtc_execution_horizon=int(config["rtc_execution_horizon"])
            if "rtc_execution_horizon" in config
            else None,
            rtc_refill_threshold=int(config["rtc_refill_threshold"])
            if "rtc_refill_threshold" in config
            else None,
            action_smoothing_alpha=float(config.get("action_smoothing_alpha", 1.0)),
        )
    )


if __name__ == "__main__":
    main()
