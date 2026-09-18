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

The current guided-episode archive can be used here after selecting the
recorded samples to retain. Its provenance fields (`operator_mask`, phase and
control generation) should be reviewed before corrections are mixed with prior
demonstrations.

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
library, including its image/video and metadata layout. The compact NPZ written
by the current manual teleoperation command is intentionally an inspection and
baseline format; it is not yet a native Pi0.5 training dataset. Converting
guided episodes into that native layout is the remaining data-pipeline step
before correction data can be used for Pi0.5 fine-tuning.

Inspect a checkpoint before deployment regardless of its origin:

```bash
rstack inspect <checkpoint>
python scripts/preflight_deployment.py \
  --station-config configs/robots/piper_bimanual.yaml \
  --worker-config configs/deploy/policy_worker.yaml \
  --checkpoint <checkpoint>
```
