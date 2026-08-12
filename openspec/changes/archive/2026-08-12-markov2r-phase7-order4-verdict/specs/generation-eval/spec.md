## ADDED Requirements

### Requirement: The shadow order-4 gate renders a verdict once its sample suffices

The Phase 7 shadow gate SHALL report `insufficient data` only while the
shadow-eligible step count is below its minimum; once the eligible count clears
that minimum, the gate SHALL resolve to pass or fail from the measured order-4
selected share against its threshold. A selected share below the threshold at a
sufficient sample SHALL render **fail**, which closes the phase without building
the order-4 index (ADR-002).

The report SHALL show the eligible step count, the selected share, and the
threshold, so the verdict is auditable from its numbers. The shadow selector is
measurement-only: its verdict SHALL NOT depend on, and SHALL NOT change,
generated output.

#### Scenario: Sufficient sample, order-4 never selected

- **WHEN** the shadow-eligible step count is at or above its minimum and the order-4 selected share is below the threshold
- **THEN** the gate renders fail, the phase is closed without implementation, and the report shows the eligible count, the selected share, and the threshold

#### Scenario: Sample still below the minimum

- **WHEN** the shadow-eligible step count is below its minimum
- **THEN** the gate reports `insufficient data`, stating how many eligible steps were observed and how many are required
