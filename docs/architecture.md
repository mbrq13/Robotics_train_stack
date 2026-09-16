# Architecture

The stack has two deployed processes and one source of physical authority.

```text
Piper + CAN + cameras → robot station (PC) ⇄ policy worker (Jetson Thor)
                              └── operations console
```

The station owns the motion lifecycle. The worker owns model loading and inference. The worker cannot bypass station validation.

## Control contract

Each station observation carries an id and the PC monotonic timestamp. The worker echoes both values with a policy action. The station accepts an action only if its session, sequence, schema, age and joint limits are valid. A watchdog pauses the station when actions stop. These comparisons are constant-size checks over the 14-value action and do not add a queue or a trajectory transformation.

`motion_mode: direct` sends a valid policy target to the Piper driver. `motion_mode: bounded` additionally runs a local speed/acceleration trajectory controller; it is deliberately opt-in because it changes the commanded trajectory and must be calibrated on the physical rig.

The protocol is intentionally small and isolated in `link/`; it may be replaced after measured latency tests without changing policy, Piper or UI code.

## Artifact contract

Native artifacts contain `model.pt` and `manifest.json`. The manifest pins policy kind, state/action dimensions, robot schema and training metrics. Standard LeRobot Pi0.5 repositories are also first-class artifacts: the stack reads their `config.json` and processor files, then checks declared state/action order and cameras against the station before inference. Deployment refuses a mismatch before the robot can be armed.
