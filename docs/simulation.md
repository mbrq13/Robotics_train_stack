# Guided simulation

This workflow validates the policy data path and guided-collection handoff
before connecting a robot. It is a kinematic replay, not a physics simulator:
the policy receives recorded camera frames, its target becomes the simulated
joint state, and the visual panel shows that state. It cannot validate contact,
collisions, camera geometry, inverse kinematics, CAN timing or an emergency
stop.

## 1. Install and prepare replay frames

Install the policy loader, simulation panel and developer checks:

```bash
./install.sh policy-lerobot sim dev
```

Export a short, immutable replay archive from a LeRobot dataset. Camera names
must match the station profile and checkpoint:

```bash
python scripts/prepare_sim_replay.py \
  --dataset <dataset-repository> \
  --cameras left top right \
  --frames 300 \
  --output artifacts/replay.npz
```

The archive contains `images_left`, `images_top` and `images_right` as
`[N,H,W,3]` RGB arrays. Its recorded `states` are retained for inspection; the
policy receives the current simulated state instead, just as it would receive
the current robot state during deployment.

## 2. Run policy replay and guided control

```bash
rstack simulate \
  --checkpoint <checkpoint> \
  --dataset artifacts/replay.npz \
  --output outputs/guided_simulation.npz
```

Open the local URL printed for Viser. The panel has six deliberate controls:

1. **Pause policy** holds the current simulated target.
2. **Take recovery control** grants the operator a recovery segment.
3. **Take correction control** labels the segment as a correction.
4. Move the joint sliders while operator control is active.
5. **Resume policy** ends operator authority and begins a new policy generation.
6. **Save episode** writes the archive and stops the run.

The terminal offers the same sequence: `p` pause, `t` recovery, `c` correction,
`r` resume, `s` save, `q` stop, each followed by Enter. A takeover is rejected
unless the policy is already paused; a resume is rejected unless operator
control is active. This makes invalid transitions visible in simulation instead
of silently accepting them on hardware.

The exported archive has `states`, `actions`, `operator_mask`, `phase` and
`control_generation`. Camera JPEG data is stored as `images_<name>_bytes` with
`images_<name>_offsets`, so it remains portable without Python object arrays.
`operator_mask` selects the recovery/correction samples for review or later
dataset conversion.

## 3. Feed targets from a VR retargeter

For a headset integration test, start the simulation with a localhost-only
target receiver:

```bash
rstack simulate --checkpoint <checkpoint> --dataset artifacts/replay.npz \
  --operator-udp-port 8766
```

After **Pause** and **Take recovery control**, a station-local VR retargeter
can send UDP datagrams to `127.0.0.1:8766` in this form:

```json
{"values": [14 joint targets in station action order]}
```

The receiver drops malformed, non-finite or incorrectly sized targets. It is a
simulation integration point, not a headset driver: headset tracking,
calibration and inverse kinematics remain device-specific. For real hardware,
that adapter must run on the station machine and submit timestamped targets to
`create_operator_target_bridge`; the bridge interpolates known samples at a
fixed output rate and holds on stale input. It must not send VR targets through
the policy-worker connection.

## 4. What this validates

Run this sequence before a real guided rollout: checkpoint contract preflight,
read-only latency profile, policy replay, pause/takeover/resume, recorded
provenance inspection, then a supervised hardware commissioning run. This
matches the human-in-the-loop pattern of policy execution, pause, human
recovery/correction and return to policy described in the [official LeRobot HIL
guide](https://github.com/huggingface/lerobot/blob/main/docs/source/hil_data_collection.mdx).
