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

## Dream-1 zero-shot on bimanual Piper

The Dream worker follows the [Dream Labs physical robot loop](https://github.com/dreamlabs-co/dreamlabs/blob/9a57047222129a76afde497ff280c4ae771be1f2/dream_library/robots/src/robots/control.py):
observe continuously, request one action chunk at a time, anchor IK to the last
submitted joints, skip only rows already executed while the request was pending,
and replace the remaining queue. The station owns CAN and bounded motion.

Run these commands from this checkout's root with Python 3.12:

```bash
./install.sh station dream ui dev
source .venv/bin/activate
```

Create `~/.dreamlabs/credentials` with the following content and restrict its
permissions with `chmod 600 ~/.dreamlabs/credentials`:

```text
DREAM_GATEWAY_URL=https://api.dreamlabs.ai
DREAM_API_URL=https://platform.dreamlabs.ai
DREAM_DATA_URL=https://data.dreamlabs.ai
DREAM_API_KEY=dl_live_REPLACE_ME
```

The hardware profile maps the left arm to `can0` and the right arm to `can1`,
both at 1,000,000 bit/s. Verify the physical adapter mapping and firmware before
enabling the arms; use the [Piper deployment guide](docs/piper_dream1_deploy.md#can-cameras-and-preflight).

| Policy view | Capture | Model image | Processing |
| --- | --- | --- | --- |
| Left wrist | 640x480 MJPG | 640x360 | Centre crop; optional calibrated fisheye rectification |
| Overhead | 1344x376 YUYV | 640x360 | ZED Mini left eye only, then centre crop and resize |
| Right wrist | 640x480 MJPG | 640x360 | Centre crop; optional calibrated fisheye rectification |

Exact persistent USB paths are in [the hardware profile](configs/robots/piper_bimanual_dream1.yaml).
Fisheye rectification is disabled until per-camera `K`/`D` calibration is available;
cropping alone does not remove lens distortion. Inspect the views before motion.

The [Piper asset bundle](assets/robots/piper/README.md) includes a single-arm URDF,
ten STL meshes (arm, gripper body and both fingers), and the Dream arm file. All
mesh references are relative; no external asset directory or ROS package is needed.
Measure the configured TCP and verify the jaw axis, 0.68 m base separation and
support-column location against the real fixture. Only after that check:

```bash
rstack dream-confirm-tcp --arm-file assets/robots/piper/piper_arm.json \
  --i-measured-this-on-the-real-gripper
rstack dream-preflight --offline
rstack dream-preflight --check-devices
```

The preflight refuses missing meshes or an unconfirmed TCP. Its online check also
verifies checkpoint access. The default is `dreamlabs/dream-1-beta`, matching the
official reference; put your account's exact checkpoint ID in
[the Dream config](configs/deploy/dream1_piper.yaml) if it differs from the default.

Start the station:

```bash
rstack station --config configs/robots/piper_bimanual_dream1.yaml
```

Open `http://127.0.0.1:8080` and inspect the three camera previews. In another
terminal, activate the same environment and request inference without actions:

```bash
rstack dream --config configs/deploy/dream1_piper.yaml --dry-run \
  --task "pick up the red block and place it in the blue bowl"
```

Stop that worker before starting the motion worker. Once the physical checks
in the deployment guide pass, set `dry_run: false` in the Dream config:

```bash
rstack dream --config configs/deploy/dream1_piper.yaml \
  --task "pick up the red block and place it in the blue bowl"
```

Use `Validate`, `Arm` and `Start` in the UI, with the physical emergency stop
within reach. A first reply after more than 60 seconds ends the motion worker
without sending its first action; restart the worker and start again once the
model is warm. Authentication and non-retryable API errors also stop the worker.
Closing the station disables motor torque: support the arms before shutdown.
See the [step-by-step Piper guide](docs/piper_dream1_deploy.md) and
[Dream-1 deployment](docs/dream1.md) for the complete commissioning sequence
and the differences between this hardware command path and the reference loop.

To inspect Dream before connecting hardware, replay a recorded LeRobot episode:

```bash
./install.sh dream-replay dev
rstack dream-replay \
  --dataset NONHUMAN-RESEARCH/yellow-towel-101-piper-640-v2 \
  --episode 0 --queries 3 --fov-ablation \
  --task "Fold the yellow towel" \
  --output outputs/dream_replay_episode_000
```

This sends recorded state and camera windows to Dream without opening a station
connection. It compares each returned chunk with the recorded continuation and
repeats identical crops before comparing them with full-frame letterboxing, so
ordinary session variation is separated from FOV sensitivity. The HTML report,
camera contact sheet, angular heatmap and raw CSV are written under the output
directory. This is open-loop evidence; it cannot establish task success because
the recorded scene does not respond to Dream's proposed actions.

To advance a simulated joint state through the complete episode, using each
Dream prediction as the state for the next request, run:

```bash
rstack dream-shadow \
  --dataset NONHUMAN-RESEARCH/yellow-towel-101-piper-640-v2 \
  --episode 0 --task "Fold the yellow towel" \
  --output outputs/dream_shadow_episode_000
```

The output video overlays the recorded trajectory in gray and Dream's
kinematic shadow trajectory in color. No station or CAN connection is opened.
The camera stream still comes from the recording, so this evaluates proposed
joint motion and trajectory divergence, not cloth interaction or task success.

The station remains idle until it is validated, armed and started from the
operations UI.

## Dream-1 with one physical SO-100 arm

The single-arm profile exposes only five joints and one gripper to the hardware
station. Dream keeps a fixed virtual left-arm context internally, but only the
right-arm chunk can cross the station contract; there is no second serial port,
driver or action vector.

Install and follow the fail-closed commissioning guide:

```bash
./install.sh so100 ui dev
source .venv/bin/activate
rstack dream-preflight --offline \
  --robot-config configs/robots/so100_single_dream1.yaml \
  --dream-config configs/deploy/dream1_so100_single.yaml
```

The provided hardware file intentionally contains placeholders for the stable
serial path and the measured five-joint home pose. It cannot start the arm until
both are replaced. The Dream config also starts with `dry_run: true`.
See [Single-arm SO-100 deployment](docs/dream1_so100.md) for calibration,
camera verification, read-only inference and the controlled-motion checklist.

## Training

```bash
rstack train --data dataset.npz --output artifacts/baseline
rstack evaluate --checkpoint artifacts/baseline --data dataset.npz
rstack train-pi05 --config configs/train/pi05_rtc.yaml --dry-run
```

## Documentation

The documentation is organized into these connected documents:

- [Architecture](docs/architecture.md): component boundaries, robot adapters
  and the policy/hardware contract.
- [Workflows](docs/workflows.md): commissioning, simulation, VR collection and
  the DAgger handoff.
- [Training](docs/training.md): dataset requirements, checkpoint inspection and
  training entry points.
- [Validation](docs/validation.md): the technical basis and evidence required
  before a real rollout.
- [Dream-1 deployment](docs/dream1.md): zero-shot Piper setup, credentials,
  cameras, preflight and first rollout.
- [Piper Dream-1 deployment](docs/piper_dream1_deploy.md): installation, CAN
  activation, cameras, verification, and robot-computer deployment.
- [Single-arm SO-100 deployment](docs/dream1_so100.md): one physical arm,
  serial calibration, dry-run inference and guarded first motion.

See [IK configuration](docs/ik_configuration.md) to select the solver, model,
and environment overrides.
