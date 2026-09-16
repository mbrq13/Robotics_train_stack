# Robotics Train Stack

Clean training and deployment software for a Piper robot connected to a PC and a policy running on a Jetson Thor.

The PC is always the safety authority. It owns CAN, cameras, motion limits, arming and watchdogs. The Thor only produces policy actions.

## Install

```bash
./install.sh station ui     # PC with Piper and cameras
./install.sh policy-lerobot # Jetson Thor, incl. Pi0.5 / LeRobot checkpoints
./install.sh dev            # development tools
```

## Train a first policy

The built-in `state_mlp` policy trains from an NPZ with `states` and `actions` arrays. It is deliberately small, deterministic and suitable for validating the full lifecycle before using a larger model.

```bash
rstack train --data demos.npz --output artifacts/my-policy --epochs 100
rstack inspect artifacts/my-policy
rstack evaluate --checkpoint artifacts/my-policy --data demos.npz
```

## Deploy

On the Piper PC, review `configs/piper_station.yaml`, connect the CAN interfaces and run:

```bash
rstack station --config configs/piper_station.yaml
```

On the Thor, deploy either a native artifact or a standard Pi0.5 repository.
The checkpoint is inspected before the station can accept its actions; its
state/action order and required cameras must match the Piper configuration.

```bash
rstack policy --config configs/thor_policy.yaml --checkpoint artifacts/my-policy
# Example: exact Pi0.5 checkpoint format used by hanoi-v1
rstack policy --config configs/thor_policy.yaml --checkpoint NONHUMAN-RESEARCH/hanoi-v1
```

Set `task` in `configs/thor_policy.yaml` to the instruction used for the Pi0.5
run. Its `state_names` is an explicit state-vector contract because Pi0.5
`config.json` can omit that semantic ordering. `rstack inspect
NONHUMAN-RESEARCH/hanoi-v1` reads its requirements without downloading its
model weights.

Open `http://PC_IP:8080` for the operations console. The robot remains paused until the operator validates and arms it. See [docs/operations.md](docs/operations.md) before using physical hardware.

`demos.npz` must contain float arrays named `states` and `actions`, each shaped `[samples, 14]` for the included bimanual Piper schema. See [docs/training.md](docs/training.md).

## Guarantees and scope

Every incoming action is checked against the current session, monotonic observation age, ordering, schema and joint limits on the robot PC. Those checks do not reshape a valid action. A lost policy connection triggers `hold()` locally. `station.motion_mode: direct` preserves accepted policy targets; `bounded` adds locally configured speed/acceleration trajectory limits after physical commissioning. Physical commissioning is still mandatory for each Piper, camera setup and workspace.
