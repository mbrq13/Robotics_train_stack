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
and limits match the active station. The bounded motion profile applies the
configured velocity and acceleration envelope at the hardware boundary.

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
