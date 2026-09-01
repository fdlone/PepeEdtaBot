# generation-hot-route Specification

## Purpose
TBD - created by archiving change l1-hot-route. Update Purpose after archive.

## Requirements

### Requirement: Hot n-grams enter the pool as a route with a slot budget

The generator SHALL offer an L1 route: when `hot_ngram_slot_ratio` is above
zero and the request carries no context, the first `route_slot_budget(target,
ratio)` attempts of the pool-building loop SHALL each be seeded by a distinct
hot n-gram drawn without replacement from the chat's hot selection (current
hotness thresholds), using the generation RNG. Those candidates SHALL take
the same path as every other attempt — walk, finalization, form gates,
extension, mutation — and SHALL be attributed to the route `hot` at creation.
The pool SHALL NOT grow: route slots come from inside the target.

At the default ratio 0 the generator SHALL consume no RNG draw and perform no
hot-n-gram read for the route, and generation SHALL be byte-identical to the
pre-route behaviour. With context present the route SHALL NOT run and SHALL
NOT be counted as attempted.

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

- **WHEN** the ratio is 0
- **THEN** `generation_hash` does not move and the hot selection is not read for the route
