## ADDED Requirements

### Requirement: Every configuration is evaluable in both context modes

The eval runner SHALL support two generation modes and SHALL be able to run any
configuration of the matrix in either:

- `ctx` — the prompt is supplied to the generator as context (the historical
  behavior of every run to date);
- `noctx` — the generator receives no context tokens; the prompt still selects
  the generation and seeds the RNG, so the two modes remain paired per prompt.

The mode SHALL be selectable per run and SHALL default to `ctx`, so an
invocation written before this requirement keeps producing the same numbers.
Within a run the mode SHALL be fixed: a single report SHALL NOT mix modes into
one metric value.

#### Scenario: Explicit noctx run

- **WHEN** the runner is invoked in `noctx` mode
- **THEN** every generation is requested without context tokens, and the run completes with the same metric set as a `ctx` run

#### Scenario: Default mode is unchanged

- **WHEN** the runner is invoked without naming a mode
- **THEN** it runs in `ctx` mode and reproduces the numbers of the pre-existing invocation bit-for-bit under the same seeds and snapshot

#### Scenario: Modes are paired per prompt

- **WHEN** the same configuration, snapshot, and seeds are run in both modes
- **THEN** both runs draw the same prompts in the same order, so metric deltas between modes are attributable to the context alone

### Requirement: A report names its mode, and a one-mode verdict is not a passed gate

Every eval report SHALL state the mode it was produced in, in a place a reader
cannot miss, and SHALL carry the mode in its machine-readable output so a
verdict note cannot cite the wrong one.

A gate whose definition requires both modes SHALL NOT be reported as passed on
the strength of one: with only one mode measured, the gate result SHALL be
`insufficient data`, exactly as a gate lacking any other pre-registered input.
Where a verdict weighs the two modes together, the weight SHALL come from the
measured production share of each mode, and the report SHALL name the share it
used and where it came from.

#### Scenario: Mode is visible in the report

- **WHEN** a report is produced in either mode
- **THEN** the mode is stated in the report header and present in the machine-readable output

#### Scenario: Gate needing both modes, only one measured

- **WHEN** a gate requiring both modes is evaluated from a single-mode run
- **THEN** the gate reports `insufficient data`, never `pass`

#### Scenario: Weighted verdict states its weight

- **WHEN** a verdict combines the two modes into one number
- **THEN** the report states the production share used as the weight and its source
