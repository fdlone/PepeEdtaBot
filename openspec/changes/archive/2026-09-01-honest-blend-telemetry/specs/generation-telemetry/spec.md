# generation-telemetry — delta (honest-blend-telemetry)

## ADDED Requirements

### Requirement: Blend displacement is measured against raw counts

For every walk step where the temporal blend is enabled, the system SHALL
additionally report the total-variation distance between the step's final
sampling weights and the normalized **raw counts** of the same pool. The
existing displacement-versus-long-layer metric SHALL be kept unchanged. The
raw-count metric SHALL be non-zero whenever the sampled distribution differs
from the raw counts — including the path where the short layer is empty and
the blend degenerates to compressed long weights. Computing the metric SHALL
NOT change generation behavior and SHALL NOT consume random draws.

#### Scenario: Empty short layer no longer reads as inert

- **WHEN** the blend is enabled and a step's short layer has no mass, so the step samples from compressed long weights
- **THEN** the raw-count displacement of that step is positive whenever compression moved the distribution, while the long-layer displacement remains 0

#### Scenario: Neutral knob stays silent

- **WHEN** `markov_alpha_*` is 0
- **THEN** no blend arithmetic runs and all three blend metrics read as absent/zero, byte-identical to the frozen baseline

#### Scenario: Metric reaches the operator

- **WHEN** generations run with the blend enabled
- **THEN** the mean raw-count displacement is visible in the generation trace and in process telemetry alongside the existing coverage/displacement pair
