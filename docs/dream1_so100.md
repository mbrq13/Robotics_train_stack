# Dream-1 inference with one physical SO-100 arm

This profile controls exactly one physical arm. The station state and action are
six values: five joint angles in radians followed by a gripper value from 0
(closed) to 1 (open). No second arm driver is created.

Dream-1's public checkpoint still has two-hand visual and action context. The
adapter therefore gives Dream a fixed virtual left-arm pose, selects only the
right-arm result and discards the virtual arm result before station validation.
This is an integration bridge, not a native single-arm checkpoint. Start with
read-only inference and inspect the proposed right-arm chunks before motion.

The implementation follows the [Dream Labs robot deployment guide](https://github.com/dreamlabs-co/dreamlabs/blob/main/dream_library/robots/README.md)
for stable serial paths, per-arm calibration, 20 Hz policy execution, temporal
camera windows, end-effector IK and fail-closed hardware checks.

## 1. Install and credentials

Use Python 3.12:

```bash
./install.sh so100 ui dev
source .venv/bin/activate
```

Store the API key outside YAML and Git, in `~/.dreamlabs/credentials`:

```text
DREAM_GATEWAY_URL=https://api.dreamlabs.ai
DREAM_API_URL=https://platform.dreamlabs.ai
DREAM_DATA_URL=https://data.dreamlabs.ai
DREAM_API_KEY=dl_live_REPLACE_ME
```

Then restrict it:

```bash
chmod 600 ~/.dreamlabs/credentials
```

`DREAM_API_KEY` may instead be exported in the shell. Never add it to
`configs/deploy/dream1_so100_single.yaml`.

## 2. Calibrate the physical arm

The arm needs a calibration file containing all six servo entries:
`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` and
`gripper`. Use the calibration produced for this exact follower arm; a file from
another arm can map the same requested angle to a different physical pose.

Resolve the adapter to a persistent name:

```bash
ls -l /dev/serial/by-id/
ls -l /dev/serial/by-path/
```

Edit [the hardware profile](../configs/robots/so100_single_dream1.yaml):

- replace `port` with the persistent path;
- replace `calibration_file` with this arm's calibration;
- replace all five `REPLACE_RAD` home values with a measured, collision-free
  pose in radians.

Do not use `/dev/ttyUSB0` or `/dev/ttyACM0`. The driver refuses those names
because they can change after a reconnect. The driver also refuses a missing
calibration or placeholder home.

Opening the station energizes the servos at their measured position and holds
that position. It does not move to `home`. Home is used only when the operator
explicitly requests it.

## 3. Cameras

The public checkpoint requires all three visual slots even though only one arm
is physical:

| Dream role | Local camera | Capture | Policy frame |
| --- | --- | --- | --- |
| `wrist_l` | left fisheye | 640x480 MJPG | 640x360 |
| `overhead` | ZED Mini left eye | 1344x376 YUYV | 640x360 |
| `wrist_r` | right fisheye | 640x480 MJPG | 640x360 |

The ZED UVC frame is split and only its left 672x376 eye is used. Fisheye
preprocessing belongs to the camera profile, not the Dream inference YAML, so a
camera replacement can disable it independently. The current fisheye profile is
an advertised-FOV approximation; measured intrinsics and distortion are still
required before treating image geometry as calibrated.

## 4. Fail-closed preflight

First replace the hardware placeholders, then validate configuration without
touching devices or the API:

```bash
rstack dream-preflight --offline \
  --robot-config configs/robots/so100_single_dream1.yaml \
  --dream-config configs/deploy/dream1_so100_single.yaml
```

Run the full preflight on the robot computer:

```bash
rstack dream-preflight --check-devices \
  --robot-config configs/robots/so100_single_dream1.yaml \
  --dream-config configs/deploy/dream1_so100_single.yaml
```

This checks the 6D schema, the one-arm driver, stable serial path, calibration
file, camera paths and shapes, bounded motion, checkpoint access and camera
roles. It does not energize the arm.

## 5. Read-only inference

Clear the workspace and keep the emergency stop within reach. Start the station:

```bash
rstack station --config configs/robots/so100_single_dream1.yaml
```

At `http://127.0.0.1:8080`, inspect all three images and use `Validate`, but do
not use `Arm` or `Start`. In another terminal run:

```bash
rstack dream --config configs/deploy/dream1_so100_single.yaml \
  --task "pick up the red block and place it in the blue bowl"
```

The supplied Dream config has `dry_run: true`, so observations and predicted
chunks are processed without sending an action message. The virtual left arm
stays fixed; only the predicted right chunk is converted to the station's 6D
format. Stop with Ctrl+C after checking inference timing and logs.

## 6. First controlled motion

Before motion:

1. Confirm the measured home pose does not cross the table, camera supports or
   cable limits.
2. Confirm positive motion of each configured joint matches the previewed arm
   convention.
3. Keep the initial 0.5 rad/s schema caps and the workspace fence unchanged.
4. Place the arm near the demonstrated task start and keep a person at the
   emergency stop.
5. Change only `dry_run: false` in
   `configs/deploy/dream1_so100_single.yaml`.

Restart the worker, then use `Validate`, `Arm` and `Start` in the UI. Any stale
camera, expired action, missing policy worker, invalid dimension or out-of-limit
target causes a local hold. A first chunk that arrives after a long cold start is
not moved; restart the worker after the model is warm.

The configured IK uses Dream's `so101_faas` geometry because Dream Labs does
not provide a separate SO-100 physical embodiment in the public client. The
servo topology is compatible, but mechanical offsets can differ. Do not treat
zero-shot behaviour as geometrically validated until measured tool poses agree
with the real arm throughout the intended workspace. A native fine-tune and a
measured embodiment are the correct path if that check fails.
