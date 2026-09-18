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

## Documentation

The documentation is deliberately limited to four connected documents:

- [Architecture](docs/architecture.md): component boundaries, robot adapters
  and the policy/hardware contract.
- [Workflows](docs/workflows.md): commissioning, simulation, VR collection and
  the DAgger handoff.
- [Training](docs/training.md): dataset requirements, checkpoint inspection and
  training entry points.
- [Validation](docs/validation.md): the technical basis and evidence required
  before a real rollout.
