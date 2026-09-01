## ADDED Requirements

### Requirement: The L1 seed draw is modelled in the harness and gated

The eval runner SHALL reproduce the reply pipeline's hot-n-gram seed draw for
self-initiated replies: in the `noctx` mode, with the configuration's
`hot_ngram_seed_chance`, `hot_ngram_min_count` and `hot_ngram_recency_share`,
using a deterministic RNG separate from the generation RNG, so that a
configuration whose hot selection is empty produces byte-identical records to
a run without the draw. The runner SHALL NOT seed generations in the `ctx`
mode, because the pipeline never seeds addressed replies.

Each record SHALL carry whether a seed was drawn and the winner's
`start_source`; the L1 gate's coverage SHALL be the share of successful
`noctx` generations whose walk started from the seed, not the share of draws.
The report SHALL print the hot-n-gram draw counters (draws, empty share) so a
channel switched off by data is visible in the report itself.

The gate `l1_hot_channel` SHALL be pre-registered before the grid is run and
SHALL require: coverage at or above its floor (below it the verdict is
`insufficient data`, never `pass`), a significant rise of
`historical_meme_rate` in `noctx`, no significant rise of copy or repetition
in either mode, no significant drop of affinity without copies in `ctx`,
latency within budget, a connectedness round (M3R-020) in `noctx`, and both
modes.

#### Scenario: Baseline unchanged by the draw

- **WHEN** C0 is run in `noctx` and its hot selection is empty
- **THEN** its records equal those of a run without the seed draw
- **AND** the report shows the draws counted with an empty share of 100%

#### Scenario: Addressed replies are never seeded

- **WHEN** any configuration is run in `ctx` mode
- **THEN** no record has a seed drawn

#### Scenario: Coverage below the floor

- **WHEN** an arm's seeded starts cover fewer successful `noctx` generations
  than the pre-registered floor
- **THEN** its `l1_hot_channel` verdict is `insufficient data`, whatever the
  metrics say
