## MODIFIED Requirements

### Requirement: Every candidate carries the route that built it

The closed route enumeration SHALL include `hot` for candidates whose walk was
seeded by the L1 route, alongside `vanilla`, `seeded`, `mutated` and
`extension`. The two-denominator counters (`attempted` / `present` / `won`)
and the rejection reasons SHALL be kept for `hot` like for every other
member, and the route SHALL appear in the trace for survivors and rejected
candidates alike.

#### Scenario: Hot route counted with its denominators

- **WHEN** the hot route ran in a generation
- **THEN** `hot` is counted as attempted, as present when it placed a candidate, and as won when its candidate was selected
