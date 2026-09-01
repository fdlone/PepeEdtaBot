## ADDED Requirements

### Requirement: Every route's contribution is reported with its two denominators

The report SHALL contain a per-route breakdown for every configuration that
was actually run. For each member of the closed route enumeration it SHALL
show: the route's share of the candidate pool, the share of generations in
which the route placed at least one candidate, its win rate given presence,
the topical affinity without copies and the copy rate of the replies it won,
the mean latency of generations with and without the route in the pool, and
its rejections before the pool grouped by failure class (M3R-021). Pool share
and presence SHALL be printed separately, and a win rate SHALL NOT appear
without its presence denominator. Routes SHALL be attributed by the generator
at candidate creation, never inferred from reply text by the harness.

A route whose mechanism did not run in a configuration SHALL be marked as not
attempted, never printed as a zero row: "ran and produced nothing" and "was
off" are different findings.

Route fields SHALL NOT enter the deterministic metric summary used for
bit-for-bit comparisons between runs and revisions.

#### Scenario: Seeded arm

- **WHEN** a configuration with seeded generation enabled is evaluated
- **THEN** the seeded row shows pool share, presence, win given presence,
  winners' affinity without copies and copy rate, latency with and without
  the route, and rejections by failure class

#### Scenario: Route off in the baseline

- **WHEN** the baseline is evaluated with the seeded knob at zero
- **THEN** the seeded row reads as not attempted, not as zeros

#### Scenario: Summary unchanged by attribution

- **WHEN** two record sets differ only in route attribution fields
- **THEN** their metric summaries are identical
