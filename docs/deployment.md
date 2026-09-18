# Deployment guide

This guide keeps model inference separate from hardware control. The station is
the only process that opens CAN devices and cameras. The policy worker may run
on a separate compute machine and connects to the station over the configured
network address.

## 1. Install the two roles

On the machine attached to the Piper and cameras, install the station and UI:

```bash
./install.sh station ui dev
```

On the compute machine, install the policy dependencies:

```bash
./install.sh policy-lerobot dev
```

Use the same revision of this repository in both locations. The worker does
not need CAN access or camera device paths.

## 2. Review the rig contract

Before connecting hardware, edit `configs/robots/piper_bimanual.yaml` for the actual
CAN interface names and camera paths. Confirm these items together:

- The left arm occupies action/state indices 0–6 and the right arm 7–13.
- Camera names, output dimensions and crop match the checkpoint inputs.
- The gripper endpoint file, if used, contains independent measured endpoints
  for both arms.
- The selected action representation matches the checkpoint. The supplied
  profile is `radians`: joints use physical radians and grippers use `0..1`.
- Joint position limits are read from each actuator during connection. Confirm
  the resulting range and the gripper endpoints for this particular rig.

The station schema, hardware profile, and worker configuration each declare
the action representation. Preflight and the worker handshake reject a
disagreement, so changing a profile cannot silently reinterpret policy output.

`motion_mode: direct` is the default and preserves the policy's output cadence.
`bounded` runs a station-local trajectory streamer and is deliberately opt-in:
it changes the trajectory received by the arm. If it is selected, establish
limits in the same logical coordinates as the profile and qualify them on the
actual rig before any task rollout.

## 3. Check the checkpoint without hardware

Run the preflight on the compute machine or any machine with the policy extra:

```bash
python scripts/preflight_deployment.py \
  --station-config configs/robots/piper_bimanual.yaml \
  --worker-config configs/deploy/policy_worker.yaml \
  --checkpoint <checkpoint>
```

It reads configuration and checkpoint metadata only. It rejects mismatched
state dimensions/order, action dimensions, camera names/shapes and invalid RTC
settings. For checkpoints whose metadata omits state names, retain the explicit
`state_names` list in `configs/deploy/policy_worker.yaml`; dimensions alone do not
prove that joint ordering is correct.

## 4. Profile the real observation path

Connect the station without arming it:

```bash
rstack station --config configs/robots/piper_bimanual.yaml
```

Then, on the compute machine, measure the complete model path:

```bash
python scripts/measure_policy_latency.py \
  --config configs/deploy/policy_worker.yaml \
  --checkpoint <checkpoint>
```

The profiler opens a read-only connection, requests live observations one at a
time, and never sends actions. Its report separates observation delivery,
model inference and their combined duration. Stop any active policy worker
first because the station accepts one worker connection at a time.

For RTC, the runtime uses the maximum recent inference latency for action-prefix
conditioning; p95 is shown only as an operational diagnostic. Convert a latency
to controller steps with `ceil(seconds * control_hz)`. A result beyond the
checkpoint's trained delay is discarded rather than applied with an untrained
prefix. If the measured peak does not fit, reduce load or control frequency, or
train a compatible model.
`execution_mode`, `rtc_execution_horizon` and `rtc_refill_threshold` live in
the worker configuration. The trained delay limit belongs to checkpoint
metadata; runtime settings cannot add RTC compatibility to a model that was
not trained for it.

For queued RTC actions, `scheduled_action_age_ms` must cover at least
`(trained_delay + execution_horizon) / control_hz`. Preflight enforces this
lower bound. The supplied value includes commissioning margin; it is not a
model-latency target.

## 5. Start a controlled rollout

Keep the station process running on the hardware machine. Start the policy
worker on the compute machine:

```bash
rstack policy --config configs/deploy/policy_worker.yaml --checkpoint <checkpoint>
```

Open the station UI and follow this order:

1. Check the camera preview. It is the same encoded frame delivered to policy
   inference, not a separate viewer feed.
2. Run **Validate** and resolve every camera, state or calibration error.
3. Verify that the workcell is clear and the physical emergency stop is ready.
4. Select **Arm**, then **Start**.
5. Monitor the state, action rejections and RTC queue telemetry.

The station rejects an action when its session, sequence, age, schema, control
generation or mechanical range is invalid. A watchdog holds the robot if the
first inference or later action stream exceeds its configured deadline.

## 6. Pause, recover and resume

Use **Pause** for a controlled stop. It commands the measured pose before
changing authority, so queued policy actions cannot resume motion afterwards.

Guided recovery follows a strict sequence: physical hold, operator authority,
operator targets, physical hold, then a fresh policy session. Timestamped
operator targets can be played at a fixed hardware cadence with
`TargetBridgeSettings`; stale input triggers a hold. Its defaults accept a
72 Hz input and deliver targets at 100 Hz, with a short intentional playback
delay so that interpolation always has two real samples. The adapter must run
on the station machine and either use the station monotonic clock or omit a
timestamp so it is stamped locally. A concrete headset or teleoperator adapter
uses this interface; it is intentionally not part of the policy-worker network
protocol.

## 7. Validate changes before hardware

Run the automated suite after changing configuration or software:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

Then validate in this order: fake station, read-only profiler, camera preview,
unloaded low-risk motion, and finally the target task. A test suite cannot
verify CAN wiring, camera geometry, gripper endpoints or emergency-stop
behavior; those remain an on-rig commissioning responsibility.
