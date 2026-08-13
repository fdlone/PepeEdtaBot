## ADDED Requirements

### Requirement: The collocation registry and its effect are observable

The system SHALL expose how many collocations a chat has in each status, and how
often the bonus and the penalty were actually applied during generation. As with
the temporal blend, the configured weight is the intent and the application
counts are the effect: a configuration whose rules never match SHALL be
distinguishable from one that is not configured.

The withheld penalty — the case where a candidate broke a collocation but the
chain never offered its right token — SHALL be counted separately, because it is
the guard against punishing the corpus and its size is the evidence that the
guard matters.

#### Scenario: Scoring with an active registry

- **WHEN** replies are generated in a chat with active collocations
- **THEN** the counts of applied bonuses, applied penalties and withheld penalties are reported

#### Scenario: Configured but never matching

- **WHEN** the bonus is non-zero but no candidate ever contains an active collocation
- **THEN** the reported application count is zero while the registry size is reported as non-zero

#### Scenario: Registry visible per chat

- **WHEN** chat statistics are requested
- **THEN** they include the number of collocations by status for that chat

### Requirement: Analyzer cost is reported, not assumed

Each maintenance pass of the analyzer SHALL record how long it took and how many
pairs it scored, so growth in the corpus shows up as a number before it shows up
as a stall.

#### Scenario: Pass completes

- **WHEN** the analyzer finishes a pass
- **THEN** its duration and the number of scored pairs are available in the telemetry, with chat identifiers masked per the existing log-privacy rules
