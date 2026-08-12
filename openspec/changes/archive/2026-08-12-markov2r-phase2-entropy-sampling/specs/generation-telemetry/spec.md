## MODIFIED Requirements

### Requirement: Distribution diagnostics are computed for every sampled pool

For every transition pool the generation walk samples from, the system SHALL compute entropy in bits (`H = -Σ p_i·log2(p_i)` over the pool's normalized weights), branching factor (pool size), normalized entropy (`H / log2(B)`, defined as 0 when branching ≤ 1), and confidence (`1 − H_norm`). Computing diagnostics SHALL NOT change generation behavior and SHALL NOT consume random draws.

Diagnostics SHALL be computed from the model's raw-count proportions, not from the temperature-adjusted sampling weights, so that a feature consuming the diagnostics cannot feed its own effect back into them. Where a feature does consume them, the behavioral change SHALL be attributable to that feature's own setting: with every such consumer at its neutral setting, output SHALL remain byte-identical to the frozen baseline.

#### Scenario: Diagnostics on a walk step

- **WHEN** a walk step samples a continuation from a pool of B > 1 candidates
- **THEN** the step's entropy, normalized entropy, branching factor, and confidence are available in the generation trace

#### Scenario: Degenerate pool

- **WHEN** a pool contains a single candidate
- **THEN** normalized entropy is 0 and confidence is 1, with no division error

#### Scenario: Behavior unchanged

- **WHEN** the same generation runs with every diagnostics consumer at its neutral setting and with the diagnostics code absent, under the same seed
- **THEN** the generated text is byte-identical

#### Scenario: Diagnostics are not self-referential

- **WHEN** a consumer changes the sampling weights based on a pool's entropy
- **THEN** the entropy reported for that pool is unchanged, because it is computed from raw counts rather than from the adjusted weights
