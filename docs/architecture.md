# Architecture

```text
Robot and cameras → station ⇄ policy worker
                       └── operations UI
```

The station owns hardware I/O and the run lifecycle. The policy worker loads
models and returns actions. The wire protocol is isolated in `link/`, while
motion checks remain local to the station.

## Contracts

- Robot schema: state order, action order, limits and camera names.
- Observation: state, images, identifier and monotonic timestamp.
- Action: session, sequence, source observation, schema version and control
  generation.
- Artifact: model metadata and expected features.

An action is accepted only when its session, sequence, age, schema, generation
and limits match the active station. The supplied Piper profile uses physical
joint coordinates in radians and a gripper opening in `0..1`. The driver
converts those values to device units at the CAN boundary. The action-space
label is part of the station/worker handshake, preventing reinterpretation.

Direct target delivery is the default. A bounded motion profile is available
only as a rig-qualified configuration choice; it changes the action trajectory
and is not a transparent safety replacement for a policy trained on direct
targets.

For guided collection, the station can transfer authority from policy to a
local operator. Timestamped operator targets pass through a fixed-rate bridge:
it interpolates between input samples, never extrapolates stale input and holds
the robot when the input stream expires. This path is separate from RTC, which
schedules policy action chunks.

## Layout

```text
hardware/   device adapters
control/    lifecycle and motion shaping
guidance/   handoff, operator target timing and provenance
link/       wire protocol
policy/     checkpoint adapters and action scheduling
learning/   training entrypoints
runtime/    station and worker services
ui/         operations API and web interface
```

Hardware selection is declarative: a `RobotProfile` chooses a registered
driver, while the station only receives the common driver contract. This keeps
the control lifecycle and policy link independent of a specific arm.
