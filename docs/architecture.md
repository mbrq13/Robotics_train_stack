# Architecture

```text
Robot and cameras → station ⇄ policy worker
                       └── operations UI
```

The station owns hardware I/O and the run lifecycle. The policy worker loads
models and returns actions. The wire protocol is isolated in `transport/`,
while motion checks remain local to the station.

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
config/         configuration loading and validation
robots/         device adapters, cameras and robot profiles
teleoperators/  operator input, VR and authority handoff
policies/       checkpoint adapters and policy interfaces
rollout/        sync/RTC execution primitives and output transforms
control/        station-local lifecycle and motion shaping
transport/      station-to-worker protocol
services/       station and policy-worker processes
training/       training entrypoints and launch helpers
evaluation/     preflight, latency and replay checks
simulation/     offline rollouts and simulated backends
ui/             operations API and web interface
```

Hardware selection is declarative: a `RobotProfile` chooses a registered
driver, while the station only receives the common driver contract. This keeps
the control lifecycle and policy link independent of a specific arm.

The user-editable YAML files are grouped independently under `configs/`:
`robots/`, `deploy/`, `teleop/` and `train/`. Code imports configuration
through `config/`; a command-line entrypoint should not contain deployment or
teleoperation behavior itself.

## Robot adapters

The core is independent of a particular robot. A deployment selects a
`RobotSchema`, cameras and a `RobotProfile`; the profile selects a registered
driver. The driver alone owns device SDK calls and must implement connection,
observation, target delivery, hold and home. The station, worker, simulation
and training modules only use the common contract.

The schema and profile declare the action representation, while the worker
declares the representation expected by the checkpoint. Those three values
must agree before actions are accepted. The bundled bimanual Piper profile is
one adapter, not an assumption made by the rest of the stack.

## Execution and collection are different concerns

`sync` and trained `rtc` are inference backends: they define when a policy
produces actions. Collection is a surrounding workflow: it decides whether a
sample is written and whether its source was the policy or the operator. This
separation matters for DAgger: changing authority must not alter a policy
checkpoint or let old queued actions survive a handoff.
