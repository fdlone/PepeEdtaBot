## MODIFIED Requirements

### Requirement: Report format and storage

The runner SHALL emit a markdown report with the sections defined in doc 05 §6 (header with snapshot/prompts/seeds, config matrix, metric table with CIs and deltas, per-category breakdown, gates, manual-eval summary when available, per-phase verdict). Reports SHALL be stored under `docs/eval_reports/` and referenced from the phase's change; phase decisions (including "closed without implementation") SHALL cite a concrete report.

Every section SHALL describe the run that produced it. A section SHALL NOT state that an input was absent when that input was supplied: where a section has a "not available" wording, that wording SHALL be reachable only when the corresponding input is genuinely missing.

#### Scenario: Report written

- **WHEN** a protocol run completes
- **THEN** a dated markdown report exists under `docs/eval_reports/` containing all mandatory sections

#### Scenario: Manual rating supplied

- **WHEN** a protocol run is given a manual rating aggregate
- **THEN** the manual-eval section reports that aggregate — the rated and genuine counts of both halves, the decoy counts, the number of raters and the agreement — and does not state that no round was conducted

#### Scenario: Manual rating absent

- **WHEN** a protocol run is given no manual rating aggregate
- **THEN** the manual-eval section says the round was not conducted in this run

## ADDED Requirements

### Requirement: The frozen-baseline generation hash is recorded and checked

The generation hash that defines the frozen baseline SHALL be recorded in a single machine-readable location, and the hash guard SHALL compare its computed value against that record and report a mismatch as a failure. Prose in documents and change tasks MAY cite the value, but SHALL NOT be the place it is defined.

The recorded value SHALL be changed only by a deliberate edit that states which change moved it and why the move is legitimate. A run that produces a different hash SHALL therefore surface as a failing check at the moment it happens, rather than as a discrepancy noticed later against a number quoted in a document.

#### Scenario: Hash matches the record

- **WHEN** the hash guard runs on the frozen snapshot at neutral settings and computes the recorded value
- **THEN** the guard reports a match and exits successfully

#### Scenario: Hash differs from the record

- **WHEN** the hash guard runs and computes a value other than the recorded one
- **THEN** the guard reports the mismatch with both values and exits with a failure status, so the drift is attributed to the change that caused it

#### Scenario: Re-anchoring the baseline

- **WHEN** a change legitimately alters generation output at neutral settings and the recorded value is updated
- **THEN** the record carries the reason and the originating change, so a later reader can tell a deliberate re-anchor from an unnoticed regression
