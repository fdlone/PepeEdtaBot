## ADDED Requirements

### Requirement: The pool-composition gate is pre-registered and coverage is a shift

The gate `pool_composition` SHALL be pre-registered before the grid is run and
SHALL require, for an arm against C0: a significant rise of
`context_affinity_without_copy` in `ctx`; no significant rise of copy or
repetition in either mode; the pool ECB invariant (mean at or above the
absolute floor shared with `structural_escape`); no significant drop of the
window escape count; latency within budget; a connectedness round (M3R-020) in
`ctx`; both modes. Its coverage SHALL be the shift of the share of successful
`ctx` replies whose winner started from context (context, hidden context or
spliced anchor) against C0; an arm whose shift is below the floor SHALL read
`insufficient data`, never `pass`.

#### Scenario: Arm that did not move the start budget

- **WHEN** an arm's context-start share differs from C0 by less than the floor
- **THEN** its `pool_composition` verdict is `insufficient data` whatever the
  affinity delta says

#### Scenario: Arm that narrows the window

- **WHEN** an arm raises affinity significantly and lowers the window escape
  count significantly
- **THEN** its verdict is `fail`

### Requirement: The winner's start source survives extension and mutation

The harness SHALL attribute the winner's `start_source` to the attempt that
produced it even when the winning text was extended or mutated after the walk:
the harness follows the trace's extension and mutation events back to the
original attempt instead of looking the final text up alone. Attribution SHALL
be captured outside the timed section.

#### Scenario: Extended winner

- **WHEN** the winning reply is a verbatim extension of a context-started attempt
- **THEN** the record's `start_source` is that attempt's source, not `None`
