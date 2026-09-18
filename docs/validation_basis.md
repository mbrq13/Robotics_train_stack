# Technical validation basis

This document records why each nontrivial runtime behavior exists and what
must be checked before it controls a physical arm. It is a design record, not
a substitute for commissioning.

## Checkpoint contract

The preflight reads a checkpoint's declared state/action dimensions, semantic
ordering when available, camera feature shapes, chunk size and RTC metadata.
It rejects a mismatch before a model is loaded or hardware is opened. A
matching dimension alone is insufficient: the explicit station state order is
the required fallback when a checkpoint does not declare names.

Pi0.5 normalizes state and action using statistics saved with the trained
artifact. The station must therefore provide the physical representation used
to collect that artifact. The supplied Piper profile represents joints as
`-100..100` across the per-arm calibrated range and gripper opening as
`0..100`. At connection the driver queries each actuator limit, uses the
documented firmware range only when a query is unavailable, applies the
device's axis convention, and converts at the final CAN call. The conversion
has unit tests for both endpoints and midpoint; it still requires an unloaded
on-rig confirmation of axis direction and gripper endpoints.

## Trained real-time chunking

RTC is enabled only for a checkpoint that declares a positive
`rtc_training_max_delay`. That number is part of training, not a deployment
tuning knob. The worker validates all of the following before inference:

- `training_max_delay <= execution_horizon <= chunk_size - training_max_delay`;
- refill threshold is at least the trained delay;
- an anticipated or measured inference delay never exceeds the trained delay;
- a queued action remains fresh for the whole possible queue lifetime.

The producer and fixed-rate actor run independently. The actor consumes a
previously planned prefix while inference runs. When a new result arrives, the
prefix that could already be in flight is retained and the corresponding
conditioned prefix from the new chunk is skipped. A late result is discarded;
it is never silently merged as if the model had been trained for that delay.

Queue depth, underruns, p95, and recent maximum latency are emitted once per
second as diagnostics. They do not participate in the control path. The
maximum recent latency, not p95, determines conditioning because a tail event
is precisely the event that can invalidate the action prefix.

This behavior follows the public Pi0.5 RTC contract and its reference action
queue implementation:

- <https://github.com/huggingface/lerobot/blob/main/docs/source/pi05.mdx>
- <https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/action_queue.py>

The stack implements trained RTC. It does not present an ordinary checkpoint
as trained RTC. Standard inference remains available for checkpoints without
this metadata.

## Teleoperation and guided collection

Operator targets are local station inputs, not remote policy actions. During a
handoff the station holds first, grants operator authority, accepts only the
current authority generation, holds again, and creates a new policy session on
resume. This prevents buffered policy actions from crossing a handoff.

When an input device and arm command loop differ in frequency, the target bridge
interpolates between two timestamped samples and holds on stale input; it does
not extrapolate a last pose. This is independent of RTC, which schedules policy
chunks. The separation follows the human-in-the-loop collection pattern
documented by LeRobot:

- <https://github.com/huggingface/lerobot/blob/main/docs/source/hil_data_collection.mdx>

## Motion limits

The Piper SDK exposes physical joint and motion-limit controls. This stack does
not invent numeric speed or acceleration values for a policy rollout. Direct
delivery is default. The bounded streamer is kept as an explicit deployment
option for a separately qualified rig because it modifies the trajectory and
can therefore change policy behavior. The SDK interface and physical-limit
commands are documented by the device vendor:

- <https://github.com/agilexrobotics/piper_sdk/blob/master/asserts/V2/INTERFACE_V2.MD>

## Required commissioning evidence

Automated tests validate protocol ordering, unit conversion, static contracts,
queue merging, stale-action rejection, and fake-station flows. They cannot
validate CAN wiring, camera framing, coordinate directions, mechanical
clearance, or emergency-stop behavior. Before an autonomous rollout, retain:

1. static preflight output;
2. read-only latency measurement;
3. live camera preview showing the exact policy frames;
4. unloaded coordinate-direction and gripper-endpoint checks;
5. a low-risk supervised motion check; and
6. the run configuration and checkpoint revision used for the rollout.
