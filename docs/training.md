# Training

## Native baseline

The built-in baseline uses an NPZ with `states` and `actions` arrays. Both
arrays must be shaped `[N, 14]` for the supplied bimanual Piper profile.

```bash
rstack train --data dataset.npz --output artifacts/baseline
rstack inspect artifacts/baseline
rstack evaluate --checkpoint artifacts/baseline --data dataset.npz
```

This is a small state-to-action baseline and a fast way to validate state
order, action order and artifact handling. It is not a substitute for a
vision-language-action training run.

The native VR recorder writes LeRobot episodes rather than this compact
baseline format. Keep that dataset for Pi0.5 training; convert or explicitly
select state/action rows only when using this small baseline.

## Pi0.5

Use `configs/train/pi05_rtc.yaml` as a starting point. Set the dataset and
output location, then validate the configuration before launching training.

```bash
rstack train-pi05 --config configs/train/pi05_rtc.yaml --dry-run
rstack train-pi05 --config configs/train/pi05_rtc.yaml
```

`rtc_training_max_delay` is persisted with the checkpoint. Runtime settings
must remain compatible with that value and the model chunk size.

## Dataset boundary

Pi0.5 training expects the dataset representation supported by its training
library, including image/video and metadata layout. `rstack teleop --record`
writes that native LeRobot layout with state/action order, task, source,
control generation and station-clock capture provenance. Before fine-tuning,
review the operator source field and retain only the correction policy chosen
for the run; autonomous actions are not expert labels automatically.

Inspect a checkpoint before deployment regardless of its origin:

```bash
rstack inspect <checkpoint>
python scripts/preflight_deployment.py \
  --station-config configs/robots/piper_bimanual.yaml \
  --worker-config configs/deploy/policy_worker.yaml \
  --checkpoint <checkpoint>
```
