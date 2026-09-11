# generation-phrase-route Specification

## Purpose
Даёт генерации маршрут, который вставляет в ответ фразу, действительно
произносившуюся в чате, как единицу с известной поддержкой — связный по
построению кандидат около темы входа, но не копия входа, — под тем же гейтом
промоушена, что и любой другой маршрут.

## Requirements

### Requirement: Phrases of the chat's index enter the pool as a route with a slot budget

The generator SHALL offer a phrase route: when `phrase_slot_ratio` is above
zero, up to `route_slot_budget(target, ratio)` candidates SHALL be assembled
around phrases of the chat's cumulative phrase index — adjacent content
bigrams or trigrams with an all-time support of at least `phrase_min_count`
— that contain an anchor of the incoming message. Anchors SHALL be the
message's scorable content tokens, the same selection the seeded and
associative routes use. A phrase every token of which occurs in the message
SHALL NOT be chosen: that is a copy of the input, not a route. Phrases SHALL
be taken best-supported first, round-robin across anchors, and a phrase
whose tokens are all contained in an already chosen phrase SHALL NOT be
chosen again for the same generation.

The phrase SHALL be inserted as a unit: every token of the phrase SHALL
appear in the candidate contiguously and in order, and the candidate SHALL
grow on both sides of it by the same bidirectional assembler a seeded
candidate uses. A phrase candidate SHALL pass the same finalization, form
gates, staleness and verbatim checks and the same scorer as every other
candidate, entering the pool without privilege and attributed to the route
`phrase` at creation. The pool SHALL NOT grow: route slots come from inside
the target, and the plain walk SHALL keep at least one slot whatever
combination of routes is enabled.

Phrase selection SHALL be deterministic and SHALL NOT consume random draws;
the index SHALL be read at most once per generation. At the default ratio 0
the generator SHALL NOT read the index and SHALL NOT draw from the RNG for
the route, and generation SHALL be byte-identical to the pre-route
behaviour.

#### Scenario: Route on, two slots

- **WHEN** the ratio is 0.4, the pool target is 5 and the message carries two anchors that occur in supported phrases
- **THEN** two candidates attributed to `phrase` are assembled around phrases of two different anchors
- **AND** each candidate contains its phrase contiguously and in order
- **AND** the pool holds at most five candidates and at least one from the plain walk

#### Scenario: Support below the threshold

- **WHEN** the only phrases containing an anchor have support below `phrase_min_count`
- **THEN** no phrase candidate is assembled for that anchor

#### Scenario: Phrase made of the input's own tokens

- **WHEN** a supported phrase consists entirely of tokens of the message
- **THEN** it is not chosen

#### Scenario: Slice of a chosen phrase

- **WHEN** a bigram is contained in a trigram already chosen for the generation
- **THEN** the bigram is not chosen as a second phrase

#### Scenario: No phrases

- **WHEN** the ratio is above zero but no anchor occurs in a supported phrase
- **THEN** the draw is counted as empty, the route is counted as attempted and not present
- **AND** the walk fills the whole pool

#### Scenario: Default keeps behaviour

- **WHEN** the ratio is 0
- **THEN** `generation_hash` does not move, the index is not read and the RNG is not drawn for the route

### Requirement: The support threshold is a runtime knob, not a constant

`phrase_min_count` SHALL be a runtime knob (environment variable and `/set`),
an integer of at least 2, with the default 3. It SHALL take effect on the
next generation without restart. The knob is the grid arm of the route's
measurement: which support is meaningful is decided by the gate, not by the
index or by the code.

#### Scenario: Threshold changed live

- **WHEN** the threshold is raised from 2 to 5 through `/set`
- **THEN** the next generation chooses only phrases with support of at least 5

#### Scenario: Threshold below the floor

- **WHEN** a value below 2 is given
- **THEN** it is rejected as invalid and the previous value stays

### Requirement: The route is measured under the pre-registered route gate with the support threshold as its grid

The phrase route SHALL be measured under the existing `route_gate` block —
one block for every route, the route under test read from the data — and
SHALL NOT register thresholds of its own. The grid SHALL sweep the support
threshold (2 / 3 / 5) at one slot ratio, in both context modes, and SHALL
carry the recognized-unit verbatim exemption switched on in every route
arm, with one control arm carrying the exemption alone so the exemption's
own effect is attributable. The verdict vocabulary is the gate's: `pass` /
`fail` / `insufficient data`; a presence share below the gate's floor SHALL
read `insufficient data`, whatever the deltas.

#### Scenario: Threshold too strict for coverage

- **WHEN** at a support threshold the route placed a candidate in fewer pools than the gate's presence floor
- **THEN** that arm reads `insufficient data` and the threshold is recorded as limiting coverage, not as a failed route

#### Scenario: Control arm carries no route

- **WHEN** the control arm switches on the verbatim exemption without the route
- **THEN** the gate reads `insufficient data` for it (no new route in its pools) and its breakdown is reported for attribution only
