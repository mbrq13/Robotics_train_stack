# Workflows

This stack separates three roles: the station owns the robot and cameras, the
worker owns policy inference, and a local operator input may temporarily own
motion. That boundary keeps a remote model process from becoming a hardware
driver and makes a correction auditable.

## Autonomous rollout

First review the robot profile: CAN and camera mappings, camera names and
crop, joint order, gripper calibration and the action representation must
match the checkpoint. Then inspect the artifact and run static preflight:

```bash
rstack inspect <checkpoint>
python scripts/preflight_deployment.py \
  --station-config configs/robots/piper_bimanual.yaml \
  --worker-config configs/deploy/policy_worker.yaml \
  --checkpoint <checkpoint>
```

Start the station on the machine attached to the robot, and the worker where
the model runs:

```bash
rstack station --config configs/robots/piper_bimanual.yaml
rstack policy --config configs/deploy/policy_worker.yaml --checkpoint <checkpoint>
```

The station remains idle until the UI validates, arms and starts it. Preview
the exact encoded camera frames supplied to the worker before arming. For a
new checkpoint, measure live observations without sending actions first:

```bash
python scripts/measure_policy_latency.py \
  --config configs/deploy/policy_worker.yaml --checkpoint <checkpoint>
```

`sync` is the starting mode. `rtc` is only valid for a checkpoint trained with
its declared delay budget; it is not a generic speed switch. Direct targets are
the default. Motion shaping is opt-in because it changes the trajectory seen by
the robot and therefore needs its own task-level qualification.

## Simulation before hardware

The replay runner exercises the policy, station lifecycle and intervention
state machine without CAN. It is kinematic replay, not a contact or collision
simulator.

```bash
./install.sh policy-lerobot sim dev
python scripts/prepare_sim_replay.py \
  --dataset <dataset-repository> --cameras left top right \
  --frames 300 --output artifacts/replay.npz
rstack simulate --checkpoint <checkpoint> --dataset artifacts/replay.npz \
  --output outputs/guided_simulation.npz
```

The panel and terminal support pause, recovery, correction and resume. Its
saved episode contains state, emitted target and control provenance. Test the
whole transition there before a real intervention. The optional localhost UDP
input (`--operator-udp-port 8766`) accepts only a 14-value simulation target;
it is an integration seam, not a headset driver.

## VR teleoperation and manual collection

VR runs beside the station, never through the policy-worker protocol. The
input path is: tracking frame, explicit source-frame transform, clutch anchor,
damped inverse kinematics, VR-only jitter filter, fixed-rate interpolation and
station validation. A lost or stale tracking stream produces a hold rather
than extrapolated movement.

```bash
rstack teleop --config configs/teleop/piper_vr.example.yaml
rstack teleop --config configs/teleop/piper_vr.example.yaml \
  --fake --engage --record --task "place object" --output outputs/vr_place.npz
```

The first command is observational. `--engage` is an explicit motion consent;
use `--fake` to validate frame conventions, clutching and limits before CAN.
Quest uses its configured TCP pose source. PICO is optional and requires its
local SDK. The source-to-robot transform, axes, gripper direction and scale
must be commissioned for the physical rig; the example identity transform is
only appropriate for a compatible simulated frame.

## DAgger: authority and recording

DAgger does **not** put the policy into a special collection mode. The policy
continues ordinary inference. The strategy around it changes station authority
and decides when a writer records samples:

```text
AUTONOMOUS --pause/hold--> HOLD --start correction--> COLLECTING
COLLECTING --finish/hold--> HOLD --resume--> AUTONOMOUS (new session)
```

During `COLLECTING`, operator targets are accepted only for that authority
generation. Resume creates a new policy session, so a policy result or RTC
chunk made before the handoff cannot move the robot afterwards. This is the
right control flow for a correction: hold first, transfer authority, record
the correction, hold again, then start policy inference afresh.

The integrated orchestrator must expose two intentional recording policies:

- **Corrections only** opens the writer at `COLLECTING` and is the default for
  initial DAgger data. It produces a compact, clearly labelled correction set.
- **Continuous** records autonomous and operator samples with an operator mask.
  Use it only when the desired retraining mix has been decided, because policy
  samples can dominate the dataset. Autonomous actions are rollout telemetry,
  not expert labels by themselves; select or relabel them before using them as
  supervised targets.

`GuidedVrHandoff` implements the authority transition and the station-local
interpolation bridge. `rstack teleop --record` currently implements the
manual, corrections-only archive used by the native baseline; the simulation
runner records both sources to make the provenance boundary observable. The
missing piece for a complete Pi0.5 DAgger loop is an integrated `rstack dagger`
orchestrator with an explicit recording policy and a native training-dataset
writer. It must run the policy, VR handoff and camera-synchronised writer in
one process group, then export the episode in the training library's native
format. Until that exists, the authority flow is valid but the manual NPZ
archive is not a Pi0.5 fine-tuning input.

## Further reading

The implementation choices follow the public specifications and reference
work below: [OpenXR reference spaces](https://registry.khronos.org/OpenXR/specs/1.0-khr/html/xrspec.html),
[damped least-squares inverse kinematics](https://cir.nii.ac.jp/crid/1361137046123304448),
[One Euro filtering](https://direction.bordeaux.inria.fr/~roussel/publications/2012-CHI-one-euro-filter.pdf),
and [LeRobot's DAgger strategy](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/strategies/dagger.py).
