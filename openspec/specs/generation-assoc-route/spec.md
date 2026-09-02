# generation-assoc-route Specification

## Purpose
TBD - created by archiving change assoc-route-pilot. Update Purpose after archive.

## Requirements

### Requirement: Associates of the message's anchors enter the pool as a route with a slot budget

The generator SHALL offer an associative route: when `assoc_slot_ratio` is
above zero, up to `route_slot_budget(target, ratio)` candidates SHALL be
assembled around associates — tokens that co-occur at distance 1 with an
anchor of the incoming message, ranked by normalized PMI weighted by pair
support, read from the chain's own transition counts (no new accumulator).
Anchors SHALL be the message's scorable content tokens; a token of the message
itself SHALL NOT be an associate. Associates SHALL be taken round-robin across
anchors, without repeating a token.

An associate candidate SHALL be assembled by the same bidirectional assembler
as a seeded candidate and SHALL pass the same finalization, form gates,
staleness and verbatim checks and the same scorer, entering the pool without
privilege and attributed to the route `assoc` at creation. The pool SHALL NOT
grow: route slots come from inside the target, and the plain walk SHALL keep
at least one slot whatever combination of routes is enabled.

At the default ratio 0 the generator SHALL perform no chain read and no RNG
draw for the route, and generation SHALL be byte-identical to the pre-route
behaviour.

#### Scenario: Route on, two slots

- **WHEN** the ratio is 0.4, the pool target is 5 and the message carries two anchors with associates
- **THEN** two candidates attributed to `assoc` are assembled around associates of two different anchors
- **AND** the pool holds at most five candidates and at least one from the plain walk

#### Scenario: Message token is not an associate

- **WHEN** a token of the message co-occurs with an anchor
- **THEN** it is not chosen as an associate (that is the seeded route's job)

#### Scenario: No associates

- **WHEN** the ratio is above zero but no anchor has a supported neighbour
- **THEN** the draw is counted as empty, the route is counted as attempted and not present
- **AND** the walk fills the whole pool

#### Scenario: Default keeps behaviour

- **WHEN** the ratio is 0
- **THEN** `generation_hash` does not move and the chain is not read for the route

### Requirement: The pilot is measured against pre-registered viability bars, not a promotion gate

The evaluation SHALL answer the pilot's four questions per arm — whether the
route builds (presence share), whether it adds trajectories (paired pool ECB
and window escape deltas), what it costs (latency) and whether it carries an
early signal (affinity without copies, copy, repetition) — against bars
registered in `eval_thresholds.yaml` before the run. The verdict vocabulary
SHALL be `viable` / `not viable` / `insufficient data`; a presence share below
the floor SHALL read `insufficient data`, never `viable`. The pilot verdict
SHALL NOT be presented as a promotion decision.

#### Scenario: Route did not exercise

- **WHEN** the route placed a candidate in fewer generations than the presence floor
- **THEN** the arm reads `insufficient data` whatever the metric deltas say

#### Scenario: Route duplicates the walk

- **WHEN** the pool ECB drops significantly against the baseline
- **THEN** the arm reads `not viable`
