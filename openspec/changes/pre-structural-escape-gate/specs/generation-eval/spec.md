## ADDED Requirements

### Requirement: Structural diversity is measured per input, as a pair

For every generation the harness SHALL record how many **substantially
different trajectories** the machinery produced, at two scopes:

- over the whole candidate pool (the pool-level count, the safety invariant);
- over the **selection window** — the candidates within the selection score
  margin of the best one, i.e. those the final draw could actually have
  returned.

Two trajectories are substantially different when their **edge overlap** — the
share of adjacent token pairs they have in common — is below a threshold
registered in the thresholds file before any run. Overlap SHALL be normalized
by the smaller of the two edge sets, so that a candidate which is a sub-path of
another counts as the same trajectory rather than as a new one. The count is
the number of groups the window's candidates fall into once every
above-threshold pair is treated as the same trajectory.

Both numbers SHALL be reported **together, always**. A report SHALL NOT publish
the pool count without the window count or the reverse: the gap between them is
itself the finding ("there is diversity, none of it survives into the window"),
and either number alone reads as its opposite.

The quality criterion SHALL be window-relative, not an absolute score bar: the
window is defined by distance from the best candidate of that same generation,
because scores are not calibrated absolutely and their scale differs between
context modes.

Computing the metric SHALL NOT change generation output and SHALL NOT consume
random draws.

#### Scenario: A pool of near-identical candidates

- **WHEN** every candidate in a generation's pool shares most of its adjacent token pairs with the others
- **THEN** both counts are 1 — one trajectory with variants, not several trajectories

#### Scenario: Diversity that does not survive scoring

- **WHEN** the pool holds several substantially different trajectories but only one of them lies within the selection margin
- **THEN** the pool count is greater than one and the window count is 1, and both appear in the report

#### Scenario: A sub-path is not a new trajectory

- **WHEN** one candidate's adjacent token pairs are a subset of another's
- **THEN** the two are counted as the same trajectory

#### Scenario: The pair is never split

- **WHEN** a report publishes the structural metric
- **THEN** it contains the pool count and the window count for the same runs

#### Scenario: No candidates

- **WHEN** a generation collects no candidates at all
- **THEN** both counts are 0 for that generation and it does not silently drop out of the denominator

### Requirement: The structural escape gate is pre-registered

The thresholds file SHALL carry a `structural_escape` block registered before
the first run that reports the metric, holding at minimum: the edge-overlap
threshold that defines "substantially different", the minimum window count a
configuration must reach, and the pool-level safety floor that any change to
the pool composition must preserve.

A configuration SHALL pass the gate only when the mean window count meets its
minimum **and** the mean pool count stays at or above the safety floor. Until a
configuration's numbers exist the gate SHALL report insufficient data rather
than a pass, and it SHALL be subject to the same two-mode rule as every other
gate that declares it.

Thresholds SHALL NOT be changed after a run has been seen; a claim that a route
improved quality SHALL NOT be accepted as a structural result without this
metric alongside it.

#### Scenario: Gate registered before data

- **WHEN** the metric is reported for the first time
- **THEN** its thresholds already exist in the thresholds file and were not derived from the numbers being reported

#### Scenario: Window count too low

- **WHEN** a configuration's mean window count is below the registered minimum
- **THEN** the gate fails for that configuration, whatever its quality metrics show

#### Scenario: Pool safety floor breached

- **WHEN** a change to the pool composition drops the mean pool count below the registered floor
- **THEN** the gate fails even if the window count rose
