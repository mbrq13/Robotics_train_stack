# Robotics Stack

Robotics Stack is a compact toolkit for training, validating and running
robot policies. It keeps hardware control, policy inference and operator
controls as separate components with explicit interfaces.

## Components

- Hardware station: Piper CAN, cameras, motion control and operator lifecycle.
- Policy worker: native artifacts and compatible LeRobot Pi0.5 checkpoints.
- Training tools: native baseline training and validated Pi0.5 launch configs.
- Operations UI: status, camera preview and run controls.

## Install

```bash
./install.sh station ui
./install.sh policy-lerobot
./install.sh dev
```

## Quick start

```bash
rstack station --config configs/robots/piper_bimanual.yaml
rstack policy --config configs/deploy/policy_worker.yaml --checkpoint <checkpoint>
```

Inspect a checkpoint before use:

```bash
rstack inspect <checkpoint>
```

The station remains idle until it is validated, armed and started from the
operations UI.

## Training

```bash
rstack train --data dataset.npz --output artifacts/baseline
rstack evaluate --checkpoint artifacts/baseline --data dataset.npz
rstack train-pi05 --config configs/train/pi05_rtc.yaml --dry-run
```

See [deployment](docs/deployment.md), [guided simulation](docs/simulation.md), [training](docs/training.md),
[operations](docs/operations.md) and [architecture](docs/architecture.md) for
the corresponding workflows. See [guided collection](docs/guided_collection.md)
for correction-data provenance and [technical validation](docs/validation_basis.md)
for the runtime assumptions and commissioning evidence. See [inference modes](docs/inference.md)
and [robot profiles](docs/robot_profiles.md) for deployment boundaries.
See [VR teleoperation](docs/teleoperation.md) for headset integration, DLS and
DAgger control handoff.
