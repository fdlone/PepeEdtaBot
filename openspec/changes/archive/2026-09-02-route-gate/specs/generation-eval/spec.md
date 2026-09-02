## ADDED Requirements

### Requirement: The route gate is pre-registered and reads the route under test from the data

The evaluation SHALL offer one promotion gate for candidate routes, registered
in `eval_thresholds.yaml` before any arm is run under it. The route under
test SHALL be derived from the data — the single route present in the arm's
pools and absent from the baseline's — and an arm with no new route or with
more than one SHALL read `insufficient data`, never a verdict.

The gate SHALL check, in this order: coverage as the route's presence share
(below the floor: `insufficient data`); the must-improve — the paired drop of
the share of inputs whose selection window holds a single trajectory, which
SHALL be significant and at least the registered size; the must-not-worsen
conditions on affinity without copies, copy and repetition; the absolute pool
ECB invariant; the latency budget; and connectedness from the solo round in
the `ctx` mode, without which the `ctx` verdict SHALL be `insufficient data`.
The gate SHALL declare `requires_both_modes`.

#### Scenario: Route derived from pools

- **WHEN** the arm's pools contain the route `assoc` and the baseline's do not
- **THEN** the gate measures `assoc` without being told its name

#### Scenario: Single-trajectory share does not drop

- **WHEN** the paired delta of the single-trajectory share is not a significant drop of the registered size
- **THEN** the arm reads `fail` with the must-improve named

#### Scenario: No connectedness round

- **WHEN** the `ctx` run carries no solo-round aggregate
- **THEN** the arm reads `insufficient data` and names the missing round
