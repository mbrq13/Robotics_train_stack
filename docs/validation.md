# Validation

This document explains the runtime checks that protect an interface contract;
it does not replace commissioning of a physical workcell.

## Checkpoint and robot contract

Preflight compares state/action dimensions and ordering, camera names and
shapes, chunk metadata and action representation. Equal dimensions alone are
not sufficient: a checkpoint and station must agree on the semantic joint
order. The Piper adapter uses physical radians and a `0..1` gripper at the
policy boundary, and converts to device units only at the CAN boundary.

## Trained real-time chunking

RTC runs a chunk producer alongside a fixed-rate actor. It is enabled only if
the checkpoint carries a positive training delay budget. The worker verifies
that the execution horizon, refill threshold, observed inference delay and
queued action age remain within the trained range. A late chunk is discarded;
it is never treated as a valid prediction for an untrained prefix.

Queue depth, underruns, p95 and maximum recent inference time are diagnostics,
not control inputs. The maximum recent time matters for prefix conditioning;
p95 is useful to see trend and jitter but cannot make a tail delay safe. This
matches the [Pi0.5 documentation](https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx)
and its [reference action queue](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/action_queue.py).

## Evidence before a real rollout

Keep the output of static preflight, a read-only live latency profile, a camera
preview, unloaded axis/gripper checks, and a supervised low-risk motion test.
Automated tests can cover protocol ordering, fake devices, units and stale
actions. They cannot prove camera geometry, CAN wiring, physical clearance,
coordinate direction or emergency-stop behaviour.

The local test gate is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

The action interface of the Piper SDK is also documented by the
[vendor interface reference](https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD).
