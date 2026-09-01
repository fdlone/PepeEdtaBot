## MODIFIED Requirements

### Requirement: The promotion gate withholds a pass until df is accumulated

Document frequency is accumulated on live prod after the reverse-index
migration and cannot be reconstructed from the retention window. Until a
protocol run exists over prod-accumulated df, the `phase5_promotion` gate SHALL
report `insufficient data`, never `pass`: an eval whose df is approximated from
the retained message window measures the machinery, not the phase's real
effect, and a green verdict bought that way would be a pass over the easy half.

Prod-accumulated df SHALL NOT by itself satisfy this precondition. The gate
SHALL additionally require a pre-registered floor on corpus **volume** and a
pre-registered ceiling on the **share of tokens observed in exactly one
document**, both fixed in the thresholds file before the gated run. Where either
is not met, the verdict SHALL be `insufficient data`, never `pass` or `fail`.

Both parts are required because each is blind to the other failure. Volume alone
repeats a bar the project has already retired: the "≥1000 phrases" floor was
dropped once it was shown to measure supply while selection is what decides.
A corpus of any size in which nearly every token appears in one document gives
`log(n_docs / (1 + df))` the same value for nearly every token, so seed ranking
has nothing to rank. Conversely a favourable singleton share over a handful of
documents is noise, not evidence.

The report SHALL print both measured quantities next to the verdict, so that
"not yet" is distinguishable from "measured and did not hold" without reading
the database.

The precondition SHALL NOT be expressed as elapsed time since deployment.
Accumulation is bounded by chat activity, which is not constant and is not under
the project's control.

#### Scenario: Eval over window-approximated df

- **WHEN** the seeded arm runs with df populated from the retained message window rather than prod-accumulated df
- **THEN** the report computes and prints the automatic gate conditions but the `phase5_promotion` verdict is `insufficient data`, with the reason stated

#### Scenario: Prod-accumulated df below the volume floor

- **WHEN** the seeded arm runs over prod-accumulated df whose document count is below the pre-registered floor
- **THEN** the verdict is `insufficient data`, the report names the volume floor and the measured document count, and no `pass` or `fail` is issued

#### Scenario: Enough documents, but df cannot discriminate

- **WHEN** the document count meets the floor while the share of tokens observed in exactly one document exceeds the pre-registered ceiling
- **THEN** the verdict is `insufficient data`, the report names the ceiling and the measured share, and no `pass` or `fail` is issued

#### Scenario: Both parts of the precondition met

- **WHEN** the document count meets the floor and the singleton share is within the ceiling
- **THEN** the precondition is satisfied, both measured quantities are printed, and the gate resolves to `pass` or `fail` strictly from the remaining pre-registered conditions

#### Scenario: Thresholds are read, not inferred

- **WHEN** the run needs the volume floor or the singleton ceiling
- **THEN** the values come from the thresholds file, and a run whose thresholds file lacks them reports `insufficient data` rather than substituting a default
