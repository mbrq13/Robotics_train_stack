"""Measure deployed-policy inference from live station observations.

The script uses the station's read-only observation mode. It never sends a
motion command and therefore does not require arming the robot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from robotics_stack.evaluation.latency import summarize_latency
from robotics_stack.link.messages import parse_observation
from robotics_stack.policy.checkpoints import DeployedPolicy, load_deployed_policy
from robotics_stack.policy.rtc import RtcSettings
from robotics_stack.runtime.config import load_yaml


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure policy inference without sending actions")
    parser.add_argument("--config", default="configs/policy_worker.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--mode", choices=("standard", "rtc"), default=None)
    return parser.parse_args()


def _resolved_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("--warmup must be non-negative and --samples must be positive")
    config = load_yaml(args.config)
    return {
        "url": str(config["station_url"]),
        "task": str(config.get("task", "")),
        "device": str(config.get("device", "cuda")),
        "schema_version": int(config.get("schema_version", 1)),
        "state_names": tuple(str(name) for name in config.get("state_names", [])),
        "mode": args.mode or str(config.get("execution_mode", "standard")),
        "rtc_execution_horizon": config.get("rtc_execution_horizon"),
        "rtc_refill_threshold": config.get("rtc_refill_threshold"),
    }


def _sync_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _configure_mode(
    policy: DeployedPolicy,
    *,
    mode: str,
    control_hz: float,
    horizon: object,
    threshold: object,
) -> None:
    if mode == "standard":
        return
    delay = policy.descriptor.rtc_training_max_delay
    if delay <= 0:
        raise ValueError("RTC profiling requires a checkpoint trained with RTC support")
    execution_horizon = int(horizon) if horizon is not None else delay
    refill_threshold = int(threshold) if threshold is not None else max(delay, execution_horizon)
    policy.configure_rtc(
        RtcSettings(
            control_hz=control_hz,
            training_max_delay=delay,
            execution_horizon=execution_horizon,
            refill_threshold=refill_threshold,
        )
    )


def _predict(policy: DeployedPolicy, observation, mode: str) -> None:
    if mode == "rtc":
        policy.predict_rtc_chunk(
            observation,
            inference_delay_steps=0,
            previous_raw_actions=None,
        )
        return
    policy.predict(observation)


async def _measure(
    policy: DeployedPolicy,
    settings: dict[str, Any],
    *,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise RuntimeError("install the policy extra to measure a deployed policy") from exc

    async with connect(settings["url"], max_size=8 * 1024 * 1024) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "hello",
                    "schema_version": settings["schema_version"],
                    "mode": "observe",
                }
            )
        )
        hello = json.loads(await websocket.recv())
        if not hello.get("read_only", False):
            raise RuntimeError("station did not acknowledge a read-only observation session")

        from robotics_stack.contracts import RobotSchema

        schema = RobotSchema.from_dict(hello["schema"])
        policy.descriptor.validate_station(schema)
        _configure_mode(
            policy,
            mode=settings["mode"],
            control_hz=schema.control_hz,
            horizon=settings["rtc_execution_horizon"],
            threshold=settings["rtc_refill_threshold"],
        )

        inference_samples: list[float] = []
        observation_age_samples: list[float] = []
        observation_delivery_samples: list[float] = []
        end_to_end_samples: list[float] = []
        for index in range(warmup + samples):
            request_started_at = time.perf_counter()
            await websocket.send(json.dumps({"type": "observation_request"}))
            observation = parse_observation(json.loads(await websocket.recv()))
            delivery_s = time.perf_counter() - request_started_at
            age_s = max(
                0.0,
                (time.monotonic_ns() - observation.station_monotonic_ns) / 1_000_000_000,
            )
            _sync_device(settings["device"])
            started_at = time.perf_counter()
            await asyncio.to_thread(_predict, policy, observation, settings["mode"])
            _sync_device(settings["device"])
            elapsed_s = time.perf_counter() - started_at
            if index >= warmup:
                inference_samples.append(elapsed_s)
                observation_age_samples.append(age_s)
                observation_delivery_samples.append(delivery_s)
                end_to_end_samples.append(delivery_s + elapsed_s)

    return {
        "checkpoint": policy.descriptor.source,
        "mode": settings["mode"],
        "control_hz": schema.control_hz,
        "inference": summarize_latency(inference_samples, schema.control_hz).to_dict(),
        "observation_age": summarize_latency(observation_age_samples, schema.control_hz).to_dict(),
        "observation_delivery": summarize_latency(
            observation_delivery_samples, schema.control_hz
        ).to_dict(),
        "observation_and_inference": summarize_latency(
            end_to_end_samples, schema.control_hz
        ).to_dict(),
    }


def main() -> None:
    args = _arguments()
    settings = _resolved_settings(args)
    if settings["mode"] not in {"standard", "rtc"}:
        raise ValueError("execution_mode must be standard or rtc")
    policy = load_deployed_policy(
        args.checkpoint,
        task=settings["task"],
        device=settings["device"],
        state_names=settings["state_names"],
    )
    result = asyncio.run(_measure(policy, settings, warmup=args.warmup, samples=args.samples))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
