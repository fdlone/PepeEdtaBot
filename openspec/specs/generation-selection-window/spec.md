# generation-selection-window Specification

## Purpose
TBD - created by archiving change selection-knobs. Update Purpose after archive.

## Requirements

### Requirement: The selection window is governed by runtime knobs

The selection margin (`selection_score_margin`), the context-relevance weight
(`context_relevance_weight`) and its cap (`context_relevance_cap`) SHALL be
runtime knobs whose defaults equal the previous module constants (0.3, 1.6,
1.6). At the defaults generation SHALL be byte-identical to the pre-knob
behaviour. The trace and telemetry SHALL report the margin in effect.

#### Scenario: Defaults keep behaviour

- **WHEN** the three knobs are at their defaults
- **THEN** `generation_hash` does not move

#### Scenario: Margin widened at runtime

- **WHEN** `selection_score_margin` is set to 0.8 via `/set` or an eval override
- **THEN** candidates within 0.8 of the best score take part in the softmax draw

### Requirement: A diversity bonus lifts distinct trajectories into the window

Behind the knob `selection_diversity_bonus` (default 0) the generator SHALL
add to every candidate other than the best-scored one a score component
`bonus × (1 − overlap)` when its edge overlap with the best candidate is below
the structural-escape similarity threshold, and nothing otherwise. Overlap
SHALL use the same definition as the structural escape gate (adjacent content
token pairs, normalized by the smaller set). At 0 the generator SHALL compute
nothing and consume no RNG draw. The component SHALL be visible in the
candidate's score breakdown.

#### Scenario: Distinct candidate lifted

- **WHEN** the bonus is 0.2 and a candidate shares no edges with the best one
- **THEN** its total rises by 0.2 and the best candidate's total is unchanged

#### Scenario: Near-duplicate untouched

- **WHEN** the bonus is 0.2 and a candidate is the best one's walk cut short
- **THEN** its total is unchanged

#### Scenario: Bonus off

- **WHEN** the bonus is 0
- **THEN** the pool is returned as is and `generation_hash` does not move
