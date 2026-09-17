# Operations

## Prepare

1. Review the station profile and device mappings.
2. Confirm joint limits, camera views and gripper calibration for the rig.
3. Run the test suite and a fake session before hardware use.

```bash
rstack station --config configs/piper_station.yaml --fake
rstack policy --config configs/policy_worker.yaml --checkpoint <checkpoint>
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
  `configs/piper_grippers.example.yaml`.

## Checkpoints

Use `rstack inspect <checkpoint>` before deployment. State/action order and
camera requirements must match the station profile. RTC mode requires a
checkpoint trained with RTC support and a reviewed latency budget.
