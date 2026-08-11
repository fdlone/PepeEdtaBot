# generation-telemetry Specification

## Purpose
Defines what the Markov generator must be able to report about its own distributions and machinery — uncertainty diagnostics, shadow order-selection statistics, cache effectiveness — where those numbers surface, and what must never leak into them. Normative sources: TZ §6 (formulas), §20 (observability), doc 03 M2R-010/020.

## Requirements

### Requirement: Distribution diagnostics are computed for every sampled pool

For every transition pool the generation walk samples from, the system SHALL compute entropy in bits (`H = -Σ p_i·log2(p_i)` over the pool's normalized weights), branching factor (pool size), normalized entropy (`H / log2(B)`, defined as 0 when branching ≤ 1), and confidence (`1 − H_norm`). Computing diagnostics SHALL NOT change generation behavior and SHALL NOT consume random draws.

#### Scenario: Diagnostics on a walk step

- **WHEN** a walk step samples a continuation from a pool of B > 1 candidates
- **THEN** the step's entropy, normalized entropy, branching factor, and confidence are available in the generation trace

#### Scenario: Degenerate pool

- **WHEN** a pool contains a single candidate
- **THEN** normalized entropy is 0 and confidence is 1, with no division error

#### Scenario: Behavior unchanged

- **WHEN** the same generation runs with diagnostics enabled and with the diagnostics code absent, under the same seed
- **THEN** the generated text is byte-identical

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
