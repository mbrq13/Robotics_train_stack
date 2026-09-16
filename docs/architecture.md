# Architecture

The stack has two deployed processes and one source of physical authority.

```text
Piper + CAN + cameras → robot station (PC) ⇄ policy worker (Jetson Thor)
                              └── operations console
```

The station owns the motion lifecycle. The worker owns model loading and inference. The worker cannot bypass station validation.

## Control contract

Each station observation carries an id and the PC monotonic timestamp. The worker echoes both values with a policy action. The station accepts an action only if its session, sequence, schema, age and joint limits are valid. A watchdog pauses the station when actions stop.

The protocol is intentionally small and isolated in `link/`; it may be replaced after measured latency tests without changing policy, Piper or UI code.

## Artifact contract

Every deployable model contains `model.pt` and `manifest.json`. The manifest pins policy kind, state/action dimensions, robot schema and training metrics. Deployment refuses a mismatch before the robot can be armed.
