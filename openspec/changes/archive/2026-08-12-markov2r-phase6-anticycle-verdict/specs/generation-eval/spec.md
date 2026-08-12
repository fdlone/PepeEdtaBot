## ADDED Requirements

### Requirement: A two-dimensional gate closes on a decisively-failing arm

When a phase gate requires two conditions to both hold (rate AND harm, ADR-015),
and one condition is measured *significantly* below its threshold — the
measured interval lies wholly below the bar — the gate SHALL resolve to a
close/fail verdict without waiting on the other condition. A condition whose
manual or slower component is still missing SHALL NOT keep the gate at
`insufficient data` when the other condition already makes the required
conjunction impossible.

This is the "a demonstrated miss outranks a missing part" rule the other gates
already apply, extended to a conjunction: if either arm cannot be met, the
conjunction cannot be met, and the phase closes without implementation
(roadmap: "непревышение любого порога ⇒ фаза закрывается без реализации").

The report SHALL state which arm failed and by how much, so the closed verdict
is auditable from its numbers. If neither arm is decisively below its threshold
and a required arm is unmeasured, the gate SHALL remain `insufficient data`.

#### Scenario: Detection arm decisively below threshold

- **WHEN** the anti-cycle gate's detection rate is measured with its whole confidence interval below the detection threshold, and the harm arm's manual component has not been collected
- **THEN** the gate resolves to close/fail — the phase is closed without implementation — and the report shows the detection rate, its interval, and the threshold it missed

#### Scenario: Neither arm decisive, one unmeasured

- **WHEN** the detection rate is near or above its threshold but the harm arm is still unmeasured
- **THEN** the gate remains `insufficient data`, because the conjunction is not yet decidable
