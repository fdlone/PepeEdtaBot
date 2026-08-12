## ADDED Requirements

### Requirement: The blend reports what it actually did

Every generation SHALL report the mixing weight it sampled at and how much of
the sampled pools the short layer actually covered — the share of sampled steps
where the short layer contributed any weight, and the mean share of probability
mass it contributed at those steps. These numbers SHALL come from the values the
walk used, not from the configured settings, so that a setting which cannot take
effect is distinguishable from one that did.

#### Scenario: Blend active

- **WHEN** a generation runs with a non-zero mixing weight over a chain with a populated short layer
- **THEN** the trace reports the mixing weight together with the short layer's step coverage and its mean mass contribution

#### Scenario: Neutral configuration reads as neutral

- **WHEN** a generation runs with the mixing weight at zero
- **THEN** the reported mixing weight is zero and the short layer's contribution is reported as zero, not as absent

#### Scenario: Blend enabled but short layer empty

- **WHEN** the mixing weight is non-zero but no state in the walk has short-layer weight
- **THEN** the reported contribution is zero, distinguishing "configured but inert" from "not configured"

### Requirement: Temporal telemetry stays within the privacy rules

Temporal telemetry SHALL consist of aggregate numbers only. Observation times
SHALL NOT be logged in a form that reconstructs when an individual message was
sent, and chat identifiers in temporal log lines SHALL be masked exactly as the
existing log-privacy rules require.

#### Scenario: Temporal trace line emitted

- **WHEN** a trace line carrying temporal numbers is written
- **THEN** it contains only aggregates, carries no per-message timestamp, and its chat identifier is masked
