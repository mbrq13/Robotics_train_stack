# Robot profiles and adapters

The core stack is robot-neutral. A station needs only a `RobotSchema`, cameras,
and a `RobotProfile`; the hardware-specific code is isolated in an adapter.

```text
schema + robot profile -> registered driver -> station -> policy worker
```

Every profile declares:

- `driver`: registered adapter identifier;
- `action_space`: physical representation consumed by that adapter;
- `options`: adapter-specific connection and calibration settings.

The station schema declares the policy-facing `robot_type`.

The schema and profile must use the same `action_space`. The schema and worker
also declare `robot_type`, which reaches policy preprocessing without a
hardware-specific hard-code. Preflight and the connection handshake reject any
disagreement before action delivery.

## Add a robot

Implement the `RobotDriver` protocol: connect, disconnect, observation,
set_target, hold, and home. Register a factory through
`register_robot_driver(name, factory)`. The factory receives the schema and
profile options; it must reject dimensions, feature order, and physical units
it cannot support.

Do not place device SDK calls in the station, policy worker, simulation, or
training modules. That keeps the lifecycle, protocol, DAgger authority, camera
handling, and checkpoint validation reusable across robots.

The supplied `piper_bimanual` driver is one adapter, not the stack's internal
model of a robot. A new driver becomes deployable only after its own simulated,
protocol, and on-rig commissioning checks pass.
