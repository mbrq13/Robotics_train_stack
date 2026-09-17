# Training

## Native baseline

The built-in baseline uses an NPZ with `states` and `actions` arrays. Both
arrays must be shaped `[N, 14]` for the supplied bimanual Piper profile.

```bash
rstack train --data dataset.npz --output artifacts/baseline
rstack inspect artifacts/baseline
rstack evaluate --checkpoint artifacts/baseline --data dataset.npz
```

## Pi0.5

Use `configs/train/pi05_rtc.yaml` as a starting point. Set the dataset and
output location, then validate the configuration before launching training.

```bash
rstack train-pi05 --config configs/train/pi05_rtc.yaml --dry-run
rstack train-pi05 --config configs/train/pi05_rtc.yaml
```

`rtc_training_max_delay` is persisted with the checkpoint. Runtime settings
must remain compatible with that value and the model chunk size.
