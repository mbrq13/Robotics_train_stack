# Inference modes

The station owns physical I/O. The worker owns policy execution. An inference
mode changes only how the worker produces policy actions; it never gives the
worker direct access to hardware.

## Sync

`execution_mode: sync` performs one complete policy selection for each received
observation, then sends its result. It is the default because it works with
every supported policy and has the smallest state surface. Use it when the
measured end-to-end inference time fits the control period, or when first
validating a checkpoint and a robot profile.

### Optional synchronous action smoothing

`action_smoothing_alpha` is an exponential moving-average coefficient applied
after a synchronous policy has produced an action and before it is sent to the
station. `1.0` is the default and identity: it preserves the checkpoint action
exactly. Values in `(0, 1)` attenuate rapid changes, but also change the action
distribution and introduce history dependence. The smoother resets whenever a
new session or control generation starts.

Use it only after an A/B comparison on the same checkpoint, workcell and task.
Record success rate, completion time, action-delta statistics and end-to-end
latency. It is not available in RTC: its queued action prefix must remain the
same representation that the checkpoint used while predicting the chunk.

## Trained RTC

`execution_mode: rtc` separates a chunk producer from a fixed-rate action
actor. The actor consumes already planned actions while the producer computes
the next chunk. It is valid only when the checkpoint declares a positive
`rtc_training_max_delay` and all preflight bounds pass.

For a relative-action policy, queued absolute targets are re-anchored to the
current state before they are reused as a model-space prefix. This is required
because the robot may have moved since the previous chunk was produced.

RTC configuration belongs to the model/rig pair, not to a generic default:

- `rtc_training_max_delay` comes from the checkpoint.
- `rtc_execution_horizon` is between that delay and `chunk_size - delay`.
- `rtc_refill_threshold` is at least that delay.
- `scheduled_action_age_ms` covers the whole planned queue lifetime.

The worker reports queue depth, underruns and latency diagnostics. A result
outside the trained delay range is discarded instead of being treated as a
valid chunk.

## Not exposed as a runtime mode

LeRobot also offers guided RTC for compatible policies. It changes how a chunk
is constrained and has policy-specific parameters. This stack does not expose
it as a generic switch: a checkpoint that was not trained for RTC runs in
`sync`, rather than silently acquiring a different control objective.

DAgger, continuous recording, highlight recording, and episodic evaluation are
collection strategies, not inference modes. They use the same station and
worker contract, then add recording and operator-control behavior around it.

The classification follows LeRobot's deployment model: sync and RTC are
inference backends, while base, DAgger, sentry, highlight, and episodic are
rollout strategies. See the [official inference guide](https://github.com/huggingface/lerobot/blob/main/docs/source/inference.mdx).

## Filtering decision

An adaptive low-pass filter can be appropriate for a noisy operator-tracking
signal, where the observed input is the quantity to clean. It is not enabled
on learned policy actions by default. A filter can suppress meaningful contact
corrections just as easily as it suppresses noise, and any low-pass filter
trades jitter for delay. Evaluate it as a separate experiment before adopting
it for a rollout path.
