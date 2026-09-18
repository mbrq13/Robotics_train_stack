# Operations

## Prepare

1. Review the station profile and device mappings.
2. Confirm joint limits, camera views and gripper calibration for the rig.
3. Run the test suite and a fake session before hardware use.

```bash
rstack station --config configs/robots/piper_bimanual.yaml --fake
rstack policy --config configs/deploy/policy_worker.yaml --checkpoint <checkpoint>
```

## Run

Start the station without `--fake`, open the operations UI, then validate,
arm and start. Use **Hold / Pause** for a controlled stop and the physical
emergency stop for physical hazards.

The camera preview shows the frames delivered to the policy. Check framing,
orientation and crop before starting a run.

## Configuration

- `watchdog_ms` controls the steady-state action deadline.
- `first_action_timeout_ms` covers initial model warmup.
- `home_action` is disabled until an approved pose is configured.
- `gripper_calibration_file` may point to per-arm endpoints based on
  `configs/robots/piper_grippers.example.yaml`.
- `execution_mode`, `rtc_execution_horizon` and `rtc_refill_threshold` are
  worker settings. Use RTC only when checkpoint metadata declares its trained
  delay budget.

## Checkpoints

Use `rstack inspect <checkpoint>` before deployment. State/action order and
camera requirements must match the station profile. RTC mode requires a
checkpoint trained with RTC support and a reviewed latency budget.

To measure a checkpoint on live camera and robot observations without arming
the station or sending actions, run:

```bash
python scripts/measure_policy_latency.py --checkpoint <checkpoint>
```

The report separates model inference, observation delivery and their combined
duration, with p95 converted to station control steps.

See [deployment](deployment.md) for the full commissioning sequence, including
the hardware-free checkpoint preflight and policy-to-operator handoff.
