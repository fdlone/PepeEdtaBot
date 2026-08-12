## ADDED Requirements

### Requirement: Seeded generation reports its two denominators separately

The system SHALL report how often a seeded candidate was present in the pool
and, separately, how often a seeded candidate won when one was present. The two
SHALL NOT be collapsed into a single rate: "a seeded candidate rarely appears"
and "it appears but loses" are different findings with different consequences
for the promotion decision, and a single win-rate over all generations hides
which one is true.

As with the temporal blend and the collocation counters, the configured seeded
share is the intent and these counts are the effect: a configuration whose
seeded branch never produces a candidate SHALL be distinguishable from one that
is not configured.

#### Scenario: Seeded generation active

- **WHEN** replies are generated with a non-zero seeded share
- **THEN** the count of generations with a seeded candidate present, and the count of those won by a seeded candidate, are both reported

#### Scenario: Configured but never anchoring

- **WHEN** the seeded share is non-zero but no chat has a token clearing the minimum seed score
- **THEN** the seeded-present count is zero and is distinguishable from the seeded share being zero
