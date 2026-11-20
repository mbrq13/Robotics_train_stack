# Dream-1 zero-shot Piper deployment

This deployment uses three independent processes: the hardware station owns
CAN, cameras, safety and arming; the Dream worker owns DreamAPI, temporal frame
windows and EEF IK; the browser UI owns the operator's explicit start and hold.

The inference lifecycle and chunk replacement rules track the official
[Dream Labs physical robot loop](https://github.com/dreamlabs-co/dreamlabs/blob/9a57047222129a76afde497ff280c4ae771be1f2/dream_library/robots/src/robots/control.py)
and the camera/policy contract follows its
[robot deployment guide](https://github.com/dreamlabs-co/dreamlabs/blob/9a57047222129a76afde497ff280c4ae771be1f2/dream_library/robots/README.md).

## 1. Install

Use Python 3.12 and install the hardware, Dream, UI and test dependencies:

```bash
./install.sh station dream ui dev
source .venv/bin/activate
```

If the shell has sourced ROS, run project commands as `env -u PYTHONPATH rstack ...`
or open a clean terminal. This prevents Python 3.10 ROS packages from entering
the Python 3.12 environment.

The `dream` extra installs the official `dreamlabs` distribution, including
`dreamlabs-client` and the IK runtime. Do not install the unrelated PyPI
package named `dream`.

## 2. DreamAPI credentials

Create an API key in the Dream Labs platform. Put it in
`~/.dreamlabs/credentials`, never in a YAML file or in Git:

```text
DREAM_GATEWAY_URL=https://api.dreamlabs.ai
DREAM_API_URL=https://platform.dreamlabs.ai
DREAM_DATA_URL=https://data.dreamlabs.ai
DREAM_API_KEY=dl_live_REPLACE_ME
```

Restrict the file to the current user:

```bash
chmod 600 ~/.dreamlabs/credentials
```

For a temporary shell, the official client also reads `DREAM_API_KEY` and the
three URL variables from the environment. The repository never reads or
prints the key itself.

## 3. Inspect the physical configuration

The two Piper CAN adapters must be mapped to the intended sides as `can0`
(left) and `can1` (right), both UP at the fixed 1,000,000 bitrate. Verify the
USB-to-side mapping on the current machine before enabling either arm:

```bash
ip -br link show type can
sudo ethtool -i can0
sudo ethtool -i can1
bash scripts/activate_piper_can.sh --list
# Replace these placeholders with the measured USB bus-info values:
bash scripts/activate_piper_can.sh LEFT_USB_BUS_INFO RIGHT_USB_BUS_INFO
```

The wrapper calls `piper_sdk/can_activate.sh` from this repository's `.venv`
for each adapter and verifies both final CAN links. It locates the installed
shell script without importing Python. Use `sudo bash` when invoking the SDK
script directly, as in the [Piper deployment guide](piper_dream1_deploy.md#can-cameras-and-preflight). It requires `ethtool` and
`can-utils` installed on Ubuntu; it does not create a Python environment.
Run `./install.sh station dream ui dev` first to install the SDK. If the
interfaces are not yet named `can0` and `can1`, inspect their actual names with
`ip -br link show type can` and run `ethtool -i` on those names.

The profile selects the Piper SDK V2 protocol. If both arms report firmware
1.8-8 or newer, change `sdk_interface: v2` to `v3`; do not mix protocol
versions speculatively. Joint units and command methods are identical in the
pinned SDK, but the CAN message protocol is not.

The ready-to-use device paths are in
`configs/robots/piper_bimanual_dream1.yaml`:

- left wrist: `pci-0000:06:00.4-usb-0:1:1.0-video-index0`, 640x480 capture;
- overhead ZED Mini: `pci-0000:06:00.3-usb-0:1:1.0-video-index0`, 1344x376;
- right wrist: `pci-0000:06:00.4-usb-0:2:1.0-video-index0`, 640x480 capture.

The wrist views are centre-cropped to 16:9. The ZED frame is split at the
middle, keeping only the left 672x376 eye, then centre-cropped. All policy
frames are 640x360 JPEG quality 85. Optional fisheye rectification is supported
by setting `fisheye_calibration_file` to an NPZ containing `K` and `D`; it is
off until the physical lenses have been calibrated.

Do not enable a generic fisheye correction or reuse another lens's parameters:
that changes object geometry unpredictably. The uncropped radial distortion is
acceptable for the first dry run, but a per-camera OpenCV fisheye calibration is
recommended if straight workspace edges bend materially in the wrist previews.
The source dimensions are strict, so the station refuses a ZED UVC mode other
than 1344x376 instead of silently treating half of a 640-wide image as one eye.

Verify that `left` is mounted on the left wrist and `right` on the right wrist.
The names refer to robot sides, not image position.

HandUMI's bimanual Piper URDF places parallel bases 0.68 m apart, at lateral
coordinates +0.34 m and -0.34 m. Its simulation table calibration places the
tabletop 0.03 m below each base plane. The configured central support is at
0.34 m from either base; its 0.08 m radius and 1.0 m height are assumptions.
Measure the physical support before motion. The `ik_region` lower z value of
-0.05 m is **not** a tabletop clearance bound; Dream's per-arm IK does not
provide a complete table or inter-arm collision check. Check proposed paths
against the physical workcell before permitting autonomous motion.

The committed profile starts with `dry_run: true` as a fail-closed default.
For an approved physical deployment, retain the file default and set
`RSTACK_DREAM_DRY_RUN=false` only in the worker terminal.

## 4. Confirm the Piper tool frame

The supplied kinematics use the Piper SDK joint convention, a TCP 0.145 m
along link6 +Z, approach +Z and jaw axis +Y. The TCP is provisional: the stock
finger joint origins are at 0.1358 m, so the nominal offset is not a physical
measurement. Compare those axes and the point
between the closed fingertips with the real gripper. The official interactive
`dreamcli tcp` tool may be used to measure a replacement arm file.

Once the values in `assets/robots/piper/piper_arm.json` have actually been
checked on the physical gripper, record that fact:

```bash
rstack dream-confirm-tcp \
  --arm-file assets/robots/piper/piper_arm.json \
  --i-measured-this-on-the-real-gripper
```

This stores a digest covering the URDF, base, flange, TCP and axes. Editing any
of them invalidates the confirmation and prevents the worker from starting.

The [asset bundle](../assets/robots/piper/README.md) contains the URDF and all ten
STL meshes, with portable relative paths and the retained asset license. Dream
extracts the six-joint base-to-flange chain and uses link spheres for its IK
collision model. Meshes support geometric inspection and external viewers;
they do not add full mesh or inter-arm collision checks to this worker.

## 5. Preflight

Static checks, without API or hardware access (TCP confirmation is still required):

```bash
rstack dream-preflight --offline
```

Full check on the robot computer:

```bash
rstack dream-preflight --check-devices
```

The full command verifies model entitlement, model camera roles, physical
camera paths, distinct resolved video nodes, `can0`/`can1`, action order,
rates, bounded motion mode, chunk lifetime, URDF, referenced meshes and TCP digest.
The default checkpoint is `dreamlabs/dream-1-beta`, as in the reference deployment.
If the account
lists a different public Dream-1 checkpoint,
replace only `checkpoint` in `configs/deploy/dream1_piper.yaml` with the exact
ID returned by the platform.

## 6. Optional offline replay

Before using the robot computer, a LeRobot episode can be evaluated directly:

```bash
rstack dream-replay \
  --dataset NONHUMAN-RESEARCH/yellow-towel-101-piper-640-v2 \
  --episode 0 --queries 3 --fov-ablation \
  --task "Fold the yellow towel" \
  --output outputs/dream_replay_episode_000
```

The replay never connects to the station. It tests the real Dream endpoint and
client-side IK using recorded camera windows and joints, writes angle comparisons,
and measures crop-versus-letterbox sensitivity against a repeated identical crop.
It remains an open-loop test: later dataset images describe the recorded action,
not the scene that Dream's different action would have produced.

For a sequential kinematic shadow rollout over the episode, run:

```bash
rstack dream-shadow \
  --dataset NONHUMAN-RESEARCH/yellow-towel-101-piper-640-v2 \
  --episode 0 --task "Fold the yellow towel" \
  --output outputs/dream_shadow_episode_000
```

After a 1.6-second recorded warm-up, every new request uses the joint state
produced by the preceding Dream action chunk. `shadow_rollout.mp4` shows the
recorded arms in gray and Dream's arms in color; `trajectory.csv` and
`report.json` contain the raw states and error summary. This mode never opens a
station or CAN connection. Its images continue along the recorded episode after
the simulated state diverges, so it is not a closed-loop cloth simulation.

To preview an advertised equidistant fisheye as a Dream-like virtual pinhole
camera, without making inference calls, run:

```bash
python scripts/preview_fisheye_for_dream.py INPUT_IMAGE_OR_VIDEO \
  --timestamp 8.0 --source-hfov 200 --source-vfov 150 \
  --target-hfov 84 --output outputs/fisheye_preview
```

The comparison contains the original frame, the current 16:9 center crop and a
640x360 rectified preview. This advertised-FOV approximation is for feasibility
checks only. Deployment must use per-camera measured fisheye intrinsics and
distortion coefficients at the actual capture resolution.

## 7. Physical policy deployment

Place the arms in a collision-free working pose, clear the workcell, and keep
the emergency stop accessible. In Terminal 1, start the station and open its
interface at `http://127.0.0.1:8080`:

```bash
env -u PYTHONPATH rstack station \
  --config configs/robots/piper_bimanual_dream1.yaml
```

Verify all three policy camera views. In Terminal 2, start the physical worker;
this explicitly overrides the committed `dry_run: true` default without editing
the shared configuration file:

```bash
env -u PYTHONPATH RSTACK_DREAM_SOLVER=dls RSTACK_DREAM_DRY_RUN=false \
  rstack dream --config configs/deploy/dream1_piper.yaml
```

Wait for `Policy connected`. Enter the task in the UI and press Enter, then use
`Validate`, `Arm`, and `Start`. Keep a person at the physical emergency stop.
`Hold / Pause` invalidates the active action session, empties useful policy
authority and holds measured joints. A camera error, excessive camera skew,
stale chunk, lost worker or empty action queue also causes a local hold.

Following the reference loop, a first successful reply after more than 60
seconds stops the motion worker without sending its first action. Restart the
worker and use Validate, Arm and Start again when the model is warm.
Authentication, invalid configuration and non-retryable API errors stop the
worker; transient errors honor the server's retry delay. An expired server
session also stops this worker and requires another operator start.

The committed profile has `disable_on_disconnect: false`, so stopping the
station does not release motor torque. Support the arms and clear the workspace
before Ctrl+C or power removal; do not assume a process exit disables the arms.

The initial profile is deliberately conservative: 20 Hz policy rows, 100 Hz
bounded CAN streaming, 0.5 rad/s joint caps, a 20% Piper controller speed and
per-arm workspace regions. Do not increase them until repeatable task-level
traces have been reviewed.

The chunk loop follows the official physical-robot implementation: only one
request is in flight, each request carries the last submitted joints, rows that
elapsed during inference are skipped, and the fresh chunk replaces the old
tail. Replies older than three seconds are discarded; an empty queue holds the
last station target, and the station pauses if no valid replacement arrives.

### Hardware-specific differences from the reference loop

The station streams bounded joint targets at 100 Hz and enforces joint speed
and acceleration limits; it does not reproduce the reference robot's motor
tracker, Cartesian end-effector speed limiter or measured-force gripper loop.
Piper uses its configured controller speed and gripper effort. The IK seam
anchor is the last submitted policy row, not measured hardware tracking; slow
motion limits can cause lag and affect zero-shot performance. Inspect tracking
before changing limits. On session changes, old buffered rows and the previous
anchor are discarded. API error recovery follows the reference's retry policy,
with expired sessions requiring an explicit restart here.

## Privacy and networking

Dream inference is a cloud service, so camera images and robot state leave the
robot computer. The official client may also start its platform recording path.
Review the current Dream Labs retention terms and keep people out of view.
Never put an API key into `configs/deploy/dream1_piper.yaml`.
