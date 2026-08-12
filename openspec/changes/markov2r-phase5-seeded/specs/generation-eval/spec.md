## ADDED Requirements

### Requirement: The seeded-generation arm and its promotion gate are computed

The eval matrix SHALL carry the C4 arm (lexical anchoring: seeded candidates in
the pool) as an available configuration once seeded generation exists, and the
report SHALL compute the `phase5_promotion` gate from the run rather than
stating that seeded generation does not exist.

The gate SHALL be computed strictly from the pre-registered thresholds: seeded
present rate, seeded win rate given present, the `context_affinity_without_copy`
delta against the no-seeded arm, latency p95, and storage growth. A promotion
verdict SHALL require all of them; failing any one SHALL NOT read as a pass.

#### Scenario: Seeded arm in a protocol run

- **WHEN** the runner evaluates the matrix with seeded generation available
- **THEN** the C4 arm is run and the report shows its seeded present rate, seeded win rate given present, affinity delta vs the no-seeded arm, p95 and storage against the pre-registered thresholds

### Requirement: The promotion gate withholds a pass until df is accumulated

Document frequency is accumulated on live prod after the reverse-index
migration and cannot be reconstructed from the retention window. Until a
protocol run exists over prod-accumulated df, the `phase5_promotion` gate SHALL
report `insufficient data`, never `pass`: an eval whose df is approximated from
the retained message window measures the machinery, not the phase's real
effect, and a green verdict bought that way would be a pass over the easy half.

#### Scenario: Eval over window-approximated df

- **WHEN** the seeded arm runs with df populated from the retained message window rather than prod-accumulated df
- **THEN** the report computes and prints the automatic gate conditions but the `phase5_promotion` verdict is `insufficient data`, with the reason stated
