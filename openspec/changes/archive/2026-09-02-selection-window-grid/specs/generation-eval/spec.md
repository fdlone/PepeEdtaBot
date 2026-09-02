## ADDED Requirements

### Requirement: The selection-window gate is pre-registered with a single-trajectory coverage

The gate `selection_window` SHALL be pre-registered before the grid is run and
SHALL require, for an arm against C0: the share of successful replies whose
selection window holds a single trajectory drops by at least the registered
floor (coverage; below it the verdict is `insufficient data`, never `pass`);
a significant rise of the window escape count; no significant drop of
affinity without copies; no significant rise of copy or repetition in either
mode; pool ECB at or above the invariant floor; latency within budget; a
connectedness round (M3R-020) in `ctx`; both modes.

#### Scenario: Mean rises, coverage does not

- **WHEN** an arm raises the mean window escape significantly while the share
  of single-trajectory inputs drops by less than the floor
- **THEN** its verdict is `insufficient data`

#### Scenario: Window widened at a topicality price

- **WHEN** an arm raises the window escape and drops affinity without copies
  significantly
- **THEN** its verdict is `fail`
