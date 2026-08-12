## ADDED Requirements

### Requirement: Temporal snapshot is a buildable artifact

The eval harness SHALL be able to build a temporal snapshot — a chain whose
transitions carry real, differing observation times — from the retained messages
of an existing snapshot, using their stored timestamps and a deterministic
replay. The builder SHALL write to a separate evaluation database and SHALL NOT
modify the source snapshot or any live chain.

The report SHALL state, for every run on such a snapshot, that the temporal
distribution was reconstructed from a retention window rather than accumulated
live, and SHALL print the resulting time span and the size of the fresh slice.
Freshness metrics computed on a reconstructed snapshot SHALL NOT be presented as
equivalent to metrics from live accumulation.

#### Scenario: Building the temporal snapshot twice

- **WHEN** the builder runs twice over the same source snapshot with the same seed
- **THEN** the two resulting databases produce byte-identical metric results

#### Scenario: Source snapshot untouched

- **WHEN** the builder runs
- **THEN** the source snapshot's tables are unchanged and the reconstructed chain lives in a separate file

#### Scenario: Provenance stated in the report

- **WHEN** a run uses a reconstructed temporal snapshot
- **THEN** the report names the snapshot as reconstructed, prints its time span and fresh-slice size, and freshness metrics carry that qualification

### Requirement: Freshness is measured against a fresh slice

On a snapshot carrying a temporal distribution, the runner SHALL report the
share of generated tokens drawn from the fresh slice of the corpus, and SHALL
report it for each configuration with a confidence interval and a delta against
C0, exactly as every other protocol metric. Where the snapshot has no temporal
distribution, the metric SHALL report insufficient data rather than a number.

#### Scenario: Freshness on a temporal snapshot

- **WHEN** a configuration is evaluated on a snapshot with a fresh slice
- **THEN** the report shows its fresh-token share with a confidence interval and a delta against C0

#### Scenario: Freshness without a temporal snapshot

- **WHEN** the same configuration is evaluated on a snapshot without observation times
- **THEN** the metric reports insufficient data and no number is printed

### Requirement: Memes from the historical slice must survive the blend

The runner SHALL evaluate whether n-grams belonging to the historical slice
remain reproducible when the blend is active. A configuration SHALL fail this
check when the historical n-gram list is reproduced significantly less often
than under C0 — the blend is required to add fresh language, not to overwrite
the chat's long memory.

#### Scenario: Blend erases long memory

- **WHEN** a configuration reproduces significantly fewer historical n-grams than C0
- **THEN** the check fails for that configuration and the report names the shortfall with its interval

### Requirement: Phase 3 gate is pre-registered and two-sided

The `phase3_temporal` gate SHALL be registered in the thresholds file before any
Phase 3 grid is run, and SHALL be two-sided: a configuration passes only if the
fresh-token share rises significantly AND the historical-meme check holds AND
copying does not rise significantly AND topicality measured without copies does
not drop significantly AND latency stays within budget. The runner SHALL print
the numbers that produced each verdict and SHALL report insufficient data when
an arm is missing.

#### Scenario: Arm improves freshness at the cost of long memory

- **WHEN** an arm raises the fresh-token share significantly but fails the historical-meme check
- **THEN** the gate verdict for that arm is fail, and the report shows both numbers

#### Scenario: Missing arm

- **WHEN** the grid lacks an arm the gate refers to
- **THEN** the verdict is insufficient data, never a pass
