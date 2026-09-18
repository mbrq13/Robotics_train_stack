# VR teleoperation

This module converts controller motion into local robot targets. It is used for
two distinct workflows: collecting demonstrations with the policy stopped, and
human correction during a DAgger rollout. It does not run on the policy worker
and it never gives a headset network connection direct access to CAN.

The local command is deliberately inert unless motion is explicitly confirmed:

```bash
rstack teleop --config configs/teleop/piper_vr.example.yaml
rstack teleop --config configs/teleop/piper_vr.example.yaml --engage --record \
  --task "place the object" --output outputs/vr_place.npz
```

The first command checks controller frames only. The second creates a manual
correction epoch and writes measured state, emitted target, available encoded
camera frames, phase and intervention provenance. `--fake --engage` exercises
the same teleoperation state path without opening CAN. PICO users may add
`--pico-usb` after installing the vendor SDK; Quest requires the companion app
to expose its configured TCP pose stream.

## Pipeline

```text
PICO or Quest tracking
  → normalized controller frame
  → source-frame calibration and clutch anchor
  → bounded incremental DLS
  → adaptive filter
  → timestamped interpolation bridge
  → station-local target validation
  → robot or simulation backend
```

`tracking.py` owns headset integration. Quest is a TCP client for
newline-delimited JSON pose frames. PICO is an optional adapter for the local
XRoboToolkit SDK; `PicoTrackingProvider.prepare_usb()` only configures its ADB
tracking tunnel and does not configure robot buses. Both yield the same
`TrackingFrame`: a timestamp, a pose and controls for each hand.

`transforms.py` owns coordinates. A tracked pose is meaningless without its
source reference space, so the rig configuration explicitly supplies the
source-to-robot rotation. An operator must validate that mapping in simulation
and then on a clear workcell. `session.py` creates a clutch anchor from the
current controller pose and the current measured tool pose every time an arm
engages; no pose received before that anchor can move the robot.

`dls.py` is generic and only requires forward kinematics plus a Jacobian.
`robots/piper_kinematics.py` provides the Piper-specific geometry. The solver
uses one warm-started, damped step, joint limits and a real elapsed-time speed
bound. This is appropriate for continuous human targets; it is not a policy
postprocessor.

`motion.py` filters only the VR-derived joint target. The adaptive cutoff
attenuates stationary tracking jitter while becoming less aggressive when the
operator moves quickly. The output then enters the existing fixed-rate bridge,
which interpolates known targets and holds if the source becomes stale. It
never extrapolates a missing controller frame.

## DAgger handoff

`GuidedVrHandoff` binds the VR session to the station state machine:

```text
policy running → pause and hold → synchronize measured joints
→ engage VR anchor → correction targets → hold → new policy session
```

The handoff calls `begin_operator_correction`, uses a station-local
`FixedRateTargetBridge`, calls `finish_operator_control`, and only then allows
`resume_policy`. This preserves source provenance and prevents queued policy
actions or stale headset frames from crossing the intervention boundary.

For a recording run, persist each camera frame, measured state, emitted target,
task, phase and `operator_mask`. The existing guided-episode format already
stores this provenance and is written by `rstack teleop --record`. A dataset
writer can translate completed episodes into the chosen training dataset format
without changing the live teleoperation path.

## Commissioning order

1. Use a simulated backend and a recorded or mock `TrackingFrame` stream.
2. Confirm controller axes, translation scale, clutch, gripper direction and
   both arm limits with no object in the workcell.
3. Validate tracking-loss hold by stopping the source stream.
4. Run a short manual recording before using DAgger with a policy.
5. Keep the physical emergency stop available for every real run.

## Technical basis

- [OpenXR spaces and controller pose actions](https://registry.khronos.org/OpenXR/specs/1.0-khr/html/xrspec.html)
  motivate treating poses as values in an explicit reference frame rather than
  device-neutral coordinates.
- [Nakamura and Hanafusa's singularity-robust inverse kinematics](https://cir.nii.ac.jp/crid/1361137046123304448)
  motivates damped IK near poorly conditioned configurations. The local speed
  bound and joint limits remain required engineering constraints.
- [The One Euro Filter paper](https://direction.bordeaux.inria.fr/~roussel/publications/2012-CHI-one-euro-filter.pdf)
  establishes the jitter/lag tradeoff used for the VR target filter; it is not
  evidence to filter learned policy actions.
- [LeRobot's DAgger rollout strategy](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/strategies/dagger.py)
  independently supports the same separation between autonomous, paused and
  human-correction phases. Our distributed station/worker topology retains
  that state machine while keeping physical I/O local.
