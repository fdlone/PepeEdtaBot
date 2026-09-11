## MODIFIED Requirements

### Requirement: A diversity bonus lifts distinct trajectories into the window

The generator SHALL add to every candidate other than the best-scored one a
score component `bonus × (1 − overlap)` when its edge overlap with the best
candidate is below the structural-escape similarity threshold, and nothing
otherwise. Overlap SHALL use the same definition as the structural escape
gate (adjacent content token pairs, normalized by the smaller set). The
component SHALL be visible in the candidate's score breakdown.

The bonus SHALL be chosen by context mode, from two runtime knobs: replies
generated **with** context tokens use `selection_diversity_bonus` (default
0.2 since 2026-09-11, promoted together with the phrase route; 0 before);
replies generated **without** context tokens use
`selection_diversity_bonus_noctx` (default 0.2). Each knob SHALL be
settable at runtime independently of the other. At 0 the generator SHALL
compute nothing and consume no RNG draw for that mode.

#### Scenario: Distinct candidate lifted

- **WHEN** the bonus in effect is 0.2 and a candidate shares no edges with the best one
- **THEN** its total rises by 0.2 and the best candidate's total is unchanged

#### Scenario: Near-duplicate untouched

- **WHEN** the bonus in effect is 0.2 and a candidate is the best one's walk cut short
- **THEN** its total is unchanged

#### Scenario: Bonus off

- **WHEN** the bonus in effect is 0
- **THEN** the pool is returned as is and generation matches the pre-bonus behaviour

#### Scenario: Reply with context uses the context knob

- **WHEN** a reply is generated with context tokens, `selection_diversity_bonus` is 0 and `selection_diversity_bonus_noctx` is 0.2
- **THEN** no bonus is applied and generation with context is byte-identical to the pre-bonus behaviour

#### Scenario: Reply without context uses the noctx knob

- **WHEN** a reply is generated without context tokens and `selection_diversity_bonus_noctx` is 0.2
- **THEN** the bonus of 0.2 is applied to the pool before the selection window is read
