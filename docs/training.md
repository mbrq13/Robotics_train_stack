# Training and validation

The initial native policy is `state_mlp`, a behaviour-cloning MLP. It exists to make the full train → artifact → replay → Thor → Piper path reproducible with code owned by this repository.

## Dataset format

Create an NPZ containing two numeric arrays:

```text
states:  [N, 14]
actions: [N, 14]
```

The order is exactly `configs/piper_station.yaml`: seven left-arm values, then seven right-arm values. Joint values are radians; grippers are normalized to `[0, 1]`.

## Commands

```bash
rstack train --data dataset.npz --output artifacts/first-run --epochs 100
rstack inspect artifacts/first-run
rstack evaluate --checkpoint artifacts/first-run --data dataset.npz
```

`evaluate` reports replay MSE and the number of policy outputs rejected by the same robot schema used at deployment. Any rejected output blocks hardware approval.

## Adding a policy

A new policy stays under `src/robotics_stack/policy/`. It must expose `predict(state)`, save a `CheckpointManifest`, and pass replay/schema tests. It must not import CAN, UI or network code.
