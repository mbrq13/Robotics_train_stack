# DREAM-1 deployment with two Piper arms

## Physical deployment: station, UI, and policy

Run these commands from `~/Videos/Robotics_train_stack`. This procedure starts
the physical policy; it does not use `--dry-run`.

### 1. Prepare the robot computer

In a clean terminal, activate the project environment. If your shell has ROS
loaded, keep `PYTHONPATH` out of every `rstack` invocation:

```bash
cd ~/Videos/Robotics_train_stack
source .venv/bin/activate
```

Confirm that the API key is available in `.env` or `~/.dreamlabs/credentials`.
Confirm the physical USB-to-arm mapping, then activate the intended CAN links:

```bash
bash scripts/activate_piper_can.sh --list
# Replace these examples only after confirming the physical left/right mapping.
bash scripts/activate_piper_can.sh LEFT_USB_BUS_INFO RIGHT_USB_BUS_INFO
```

Verify the arm file/TCP confirmation and the live hardware contract before
enabling the station:

```bash
env -u PYTHONPATH rstack dream-preflight --check-devices
```

### 2. Start the station and its browser interface

Keep the arms supported, clear the workspace, and keep the emergency stop
reachable. In Terminal 1, start the station:

```bash
env -u PYTHONPATH rstack station \
  --config configs/robots/piper_bimanual_dream1.yaml
```

The station owns CAN, cameras, the local bounded-motion bridge, and the UI API.
Open `http://127.0.0.1:8080` in a browser; do not open `index.html` directly.
From another computer, use the station computer's IP address instead of
`127.0.0.1`. Confirm all three camera previews and that the UI reports the
station as available.

### 3. Start the physical Dream-1 worker

The committed deployment config is deliberately `dry_run: true`. Override it
explicitly for this physical deployment. In Terminal 2:

```bash
cd ~/Videos/Robotics_train_stack
source .venv/bin/activate
env -u PYTHONPATH RSTACK_DREAM_SOLVER=dls RSTACK_DREAM_DRY_RUN=false \
  rstack dream --config configs/deploy/dream1_piper.yaml
```

Wait until the UI reports **Policy connected**. The worker connects to DreamAPI,
runs the single-arm URDF/DLS conversion for both Piper sides, and returns action
chunks to the station. It does not arm or start motion on its own.

### 4. Submit task and begin motion

Enter a natural-language task in **Dream task**, for example:

```text
Pick up the purple bunch of grapes and place it in the yellow basket.
```

Press **Enter** to submit it. **Shift+Enter** inserts a line break, and the
**Save task** button remains available. A submitted station task overrides the
YAML, `.env`, and `--task` value for the current station process. Submission
does not move the robot: it starts a fresh policy episode on the next
observation. The dashboard displays the active task, mean inference time, last
inference time, and sample count.

Once camera framing, task, and the physical workspace are confirmed, select
**Validate → Arm → Start**. Keep a person at the emergency stop. To change a
task, select **Hold / Pause → edit → Enter → Arm → Start**. Editing is
intentionally rejected while armed or running.

## Installation and model contract

```bash
./install.sh station dream ui dev
source .venv/bin/activate
sudo apt install iproute2 ethtool can-utils v4l-utils
```

Set `DREAM_API_KEY` in `.env` or `~/.dreamlabs/credentials`, never in YAML or
Git. If ROS has exported `PYTHONPATH`, use `env -u PYTHONPATH rstack ...`.

Dream uses the single-arm URDF at `assets/robots/piper/piper_arm.urdf` for both
Piper arms. The local DLS solver converts each API result into the bimanual
station action contract. This workflow has joint and velocity limits but does
not provide table or inter-arm collision avoidance; measure and clear the real
workcell before motion. See [IK configuration](ik_configuration.md) for the
solver, model, and environment overrides.

## CAN, cameras, and preflight

Confirm the USB-to-arm mapping before activation. The first adapter becomes
`can0` (left) and the second becomes `can1` (right):

```bash
# Replace these examples only after measuring the physical mapping.
bash scripts/activate_piper_can.sh 1-3.1.2:1.0 1-3.1.1:1.0
```

Both links must be UP at 1,000,000 bit/s. `ERROR-ACTIVE` is normal CAN state
but does not prove that an arm responds:

```bash
timeout 5 candump -n 10 can0
timeout 5 candump -n 10 can1
```

Configure persistent camera paths in `configs/robots/piper_bimanual_dream1.yaml`:
`left` is the left wrist, `right` is the right wrist, and `top` is the overhead
view. Wrist cameras capture 640×480 and the ZED captures 1344×376; Dream receives
three 640×360 frames after the configured crops. Fisheye correction is disabled
until it has measured per-camera calibration data.

If it has not already been completed, validate the arm file and measured tool
frame before deployment:

```bash
rstack dream-confirm-tcp --arm-file assets/robots/piper/piper_arm.json \
  --i-measured-this-on-the-real-gripper
rstack dream-preflight --offline --check-devices
rstack dream-preflight --check-devices
```

## Shutdown

Select **Hold / Pause**, stop the Dream worker with Ctrl+C, then stop the
station with Ctrl+C. With `disable_on_disconnect: false`, the arms can remain
energized after the processes exit. Set it to `true` only while arms are
supported; a power cut, fault, or SIGKILL does not guarantee a held pose.
