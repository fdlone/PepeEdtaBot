## MODIFIED Requirements

### Requirement: Hot n-grams enter the pool as a route with a slot budget

The generator SHALL offer an L1 route: when `hot_ngram_slot_ratio` is above
zero and the request carries no context, the first `route_slot_budget(target,
ratio)` attempts of the pool-building loop SHALL each be seeded by a distinct
hot n-gram drawn without replacement from the chat's hot selection (current
hotness thresholds), using the generation RNG. Those candidates SHALL take
the same path as every other attempt — walk, finalization, form gates,
extension, mutation — and SHALL be attributed to the route `hot` at creation.
The pool SHALL NOT grow: route slots come from inside the target.

Since 2026-09-11 the route is on by default: `hot_ngram_slot_ratio` defaults
to 0.4 and the hotness thresholds `hot_ngram_min_count` / `hot_ngram_recency_share`
to 2 / 0.25 — the configuration that took the `l1_hot_channel` gate. At
ratio 0 the generator SHALL consume no RNG draw and perform no hot-n-gram
read for the route, and generation SHALL be byte-identical to the pre-route
behaviour. With context present the route SHALL NOT run and SHALL NOT be
counted as attempted.

The route SHALL be the hot channel's only reader, and its slot ratio SHALL
also gate the channel's write: the learn path SHALL record a message's content
n-grams into the hot window only while `hot_ngram_slot_ratio` is above zero,
so a switched-off route leaves the learn path write-free and a channel cannot
be silenced by a knob that no reader depends on (the former per-reply seed
draw and its knob were removed 2026-09-11).

#### Scenario: Route on, self-initiated reply

- **WHEN** the ratio is 0.4, the pool target is 5 and the request has no context
- **THEN** two attempts are seeded by two different hot n-grams
- **AND** the pool holds at most five candidates and at least one from the plain walk

#### Scenario: Route on, addressed reply

- **WHEN** the request carries context tokens
- **THEN** no attempt is seeded by the route and `hot` is not among the attempted routes

#### Scenario: Hot selection empty

- **WHEN** the ratio is above zero but the hot selection is empty
- **THEN** the draw is counted as empty, the route is counted as attempted and not present
- **AND** the walk fills the whole pool

#### Scenario: Default keeps behaviour

- **WHEN** the ratio is set to 0
- **THEN** generation is byte-identical to the pre-route behaviour for the same seed

#### Scenario: Window written only while the route can read it

- **WHEN** a message is learned with the ratio above zero
- **THEN** its content n-grams are recorded into the hot window
- **WHEN** a message is learned with the ratio at zero
- **THEN** nothing is recorded
