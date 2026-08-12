# generation-telemetry Specification

## Purpose
Defines what the Markov generator must be able to report about its own distributions and machinery — uncertainty diagnostics, shadow order-selection statistics, cache effectiveness — where those numbers surface, and what must never leak into them. Normative sources: TZ §6 (formulas), §20 (observability), doc 03 M2R-010/020.
## Requirements
### Requirement: Distribution diagnostics are computed for every sampled pool

For every transition pool the generation walk samples from, the system SHALL compute entropy in bits (`H = -Σ p_i·log2(p_i)` over the pool's normalized weights), branching factor (pool size), normalized entropy (`H / log2(B)`, defined as 0 when branching ≤ 1), and confidence (`1 − H_norm`). Computing diagnostics SHALL NOT change generation behavior and SHALL NOT consume random draws.

Diagnostics SHALL be computed from the model's raw-count proportions, not from the temperature-adjusted sampling weights, so that a feature consuming the diagnostics cannot feed its own effect back into them. Where a feature does consume them, the behavioral change SHALL be attributable to that feature's own setting: with every such consumer at its neutral setting, output SHALL remain byte-identical to the frozen baseline.

#### Scenario: Diagnostics on a walk step

- **WHEN** a walk step samples a continuation from a pool of B > 1 candidates
- **THEN** the step's entropy, normalized entropy, branching factor, and confidence are available in the generation trace

#### Scenario: Degenerate pool

- **WHEN** a pool contains a single candidate
- **THEN** normalized entropy is 0 and confidence is 1, with no division error

#### Scenario: Behavior unchanged

- **WHEN** the same generation runs with every diagnostics consumer at its neutral setting and with the diagnostics code absent, under the same seed
- **THEN** the generated text is byte-identical

#### Scenario: Diagnostics are not self-referential

- **WHEN** a consumer changes the sampling weights based on a pool's entropy
- **THEN** the entropy reported for that pool is unchanged, because it is computed from raw counts rather than from the adjusted weights

### Requirement: Shadow order-4 selection is measured without an order-4 index

The system SHALL estimate, for each order-3 walk step with at least 4 tokens of history, whether a variable-order 4→3→2 selector would have chosen order 4 — using order-4 support estimated from the retained message window (no order-4 storage exists before Phase 7; the estimate feeds that phase's pre-registered gate). The shadow computation SHALL NOT alter the walk, SHALL NOT consume random draws, and SHALL be disableable by a runtime knob. Telemetry SHALL publish both the number of eligible steps and the share where order 4 would have been selected, and SHALL record that the estimator is window-based (a conservative lower bound).

#### Scenario: Shadow counters accumulate

- **WHEN** generations run with the shadow selector enabled
- **THEN** telemetry reports eligible-step count and would-select-order-4 share for the process lifetime

#### Scenario: Shadow disabled

- **WHEN** the shadow knob is off
- **THEN** no shadow computation runs and generation output is unchanged either way

### Requirement: Cache effectiveness is observable

The distribution caches SHALL count hits and misses, and `/stats` SHALL report the hit-rate alongside the model volume. Counters SHALL be per-process (reset on restart) and cheap enough to leave always on.

#### Scenario: Hit-rate in /stats

- **WHEN** `/stats` is called after some generations
- **THEN** the reply includes the cache hit-rate derived from the process counters

### Requirement: Telemetry stays inside the privacy rules

Telemetry values SHALL be numbers and enum-like labels only: no raw message or candidate text outside the existing opt-in trace log, and chat identifiers only in masked form in any log line (the existing log-privacy rules extend to every new telemetry line).

#### Scenario: New telemetry log line

- **WHEN** a telemetry line mentioning a chat is written to the log
- **THEN** the chat identifier appears only masked, and no message text appears

### Requirement: The blend reports what it actually did

Every generation SHALL report the mixing weight it sampled at and how much the
blend actually changed what was sampled — the share of sampled steps where the
short layer contributed any weight, and how far the blended distribution moved
away from the long layer alone, averaged over the walk. These numbers SHALL come
from the values the walk used, not from the configured settings, so that a
setting which cannot take effect is distinguishable from one that did.

A mixing weight is a configured intent; the distance the distribution actually
moved is the effect. Reporting only the former is how a knob that changes
nothing gets read as a knob that works.

#### Scenario: Blend active

- **WHEN** a generation runs with a non-zero mixing weight over a chain with a populated short layer
- **THEN** the trace reports the mixing weight together with the short layer's step coverage and the mean distance the blend moved the distribution

#### Scenario: Neutral configuration reads as neutral

- **WHEN** a generation runs with the mixing weight at zero
- **THEN** the reported mixing weight is zero and the reported movement is zero, not absent

#### Scenario: Blend enabled but short layer empty

- **WHEN** the mixing weight is non-zero but no state in the walk has short-layer weight
- **THEN** the reported coverage and movement are both zero while the mixing weight is reported as set, distinguishing "configured but inert" from "not configured"

### Requirement: Temporal telemetry stays within the privacy rules

Temporal telemetry SHALL consist of aggregate numbers only. Observation times
SHALL NOT be logged in a form that reconstructs when an individual message was
sent, and chat identifiers in temporal log lines SHALL be masked exactly as the
existing log-privacy rules require.

#### Scenario: Temporal trace line emitted

- **WHEN** a trace line carrying temporal numbers is written
- **THEN** it contains only aggregates, carries no per-message timestamp, and its chat identifier is masked

### Requirement: Seeded generation reports its two denominators separately

The system SHALL report how often a seeded candidate was present in the pool
and, separately, how often a seeded candidate won when one was present. The two
SHALL NOT be collapsed into a single rate: "a seeded candidate rarely appears"
and "it appears but loses" are different findings with different consequences
for the promotion decision, and a single win-rate over all generations hides
which one is true.

As with the temporal blend and the collocation counters, the configured seeded
share is the intent and these counts are the effect: a configuration whose
seeded branch never produces a candidate SHALL be distinguishable from one that
is not configured.

#### Scenario: Seeded generation active

- **WHEN** replies are generated with a non-zero seeded share
- **THEN** the count of generations with a seeded candidate present, and the count of those won by a seeded candidate, are both reported

#### Scenario: Configured but never anchoring

- **WHEN** the seeded share is non-zero but no chat has a token clearing the minimum seed score
- **THEN** the seeded-present count is zero and is distinguishable from the seeded share being zero

