# Guided collection

Guided collection preserves the distinction between a policy rollout and a
human correction. A collection pass follows this order:

1. Run the policy.
2. Put the robot in hold before the operator takes control.
3. Record recovery and correction motion with operator provenance.
4. Return to hold, then resume the policy with a new control generation.

The generation is attached to observations and policy actions. An action
computed before a handoff is rejected after that handoff, including actions
already buffered by RTC.

`GuidedEpisode` exports `states` and `actions`, so its NPZ output is usable by
the native baseline trainer. It also writes `operator_mask`, `phase` and
`control_generation`; these fields retain the information needed to audit,
filter or weight corrections before a subsequent training run.

An input-device adapter is intentionally separate from this contract. It must
perform the physical hold and use the same station schema before it can append
operator samples. This keeps new teleoperation hardware from changing the
policy deployment path.

An adapter drives the station through `begin_operator_recovery` or
`begin_operator_correction`, `apply_operator_target`, `finish_operator_control`
and `resume_policy`. Operator targets are accepted only while that authority
generation is active; resuming creates a fresh policy session.

When an input device and the Piper operate at different rates, use
`TargetBridgeSettings` and `create_operator_target_bridge`. The bridge keeps a
short timestamped history, plays it at a fixed output rate and interpolates
only between known samples. If the source becomes stale, it holds the robot and
requires a new operator epoch rather than extrapolating a late movement.
