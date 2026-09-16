# Operations runbook

## Before connecting hardware

1. Confirm physical e-stop access and a clear workspace.
2. Confirm the configured CAN interfaces match the connected left/right arms.
3. Validate and replace every joint limit in `configs/piper_station.yaml` for this rig.
4. Run the full fake-robot test path first.

## Fake end-to-end check

```bash
# PC terminal
rstack station --config configs/piper_station.yaml --fake

# Thor terminal, once a checkpoint exists
rstack policy --config configs/thor_policy.yaml --checkpoint artifacts/my-policy
```

Open the console, select **Validate**, then **Arm**, then **Start**. The fake robot accepts actions only in `running` state.

## Real Piper commissioning

1. Start the station without `--fake` only after CAN activation and gripper calibration are verified.
2. Inspect the dashboard. It must report `ready` before arming.
3. Validate hardware. Any failure is a blocker, not a warning.
4. Arm only with the workspace clear.
5. Start with a checkpoint previously approved in dry-run.
6. Use **Hold / Pause** for controlled stopping. Use the physical emergency stop for immediate physical hazards.

Start the station without `--fake` only after CAN activation and gripper calibration are verified. Use the Piper SDK activation procedure appropriate for the installed hardware, and verify both configured interfaces are `UP` at the required bitrate first.

`Home` remains disabled until a reviewed `station.home_action` is configured.

The station enters pause when the policy link disappears or action traffic stops. It never resumes motion on reconnect; arm and start are operator actions.

## Pi0.5 / LeRobot checkpoint preflight

On the Thor, install `policy-lerobot`, set the task instruction in
`configs/thor_policy.yaml`, and inspect the artifact before connecting:

```bash
rstack inspect NONHUMAN-RESEARCH/hanoi-v1
```

The descriptor must show 14 state values, 14 action values and the `left`,
`top`, `right` cameras. The first real connection repeats that compatibility
check against the live station. A checkpoint that requests a missing camera,
uses a different feature order, or receives an image with unexpected shape is
rejected; it cannot silently drive a differently wired Piper.

Pi0.5's public config may omit state feature names. For that case,
`thor_policy.yaml` explicitly records the state order used by this Piper and
the worker rejects a station whose order differs. Review it against the
checkpoint's training data before a real run.
