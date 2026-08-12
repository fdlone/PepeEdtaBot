# generation-temporal-layer Specification

## Purpose
Gives the transition model a sense of time: a decaying short-term counter that
tracks what the chat says now, a long-term count that never decays and holds
what it has always said, and a blend between them whose strength follows the
chat's mood (normative source: `docs/v2/02_MARKOV_2_0R_TZ.md` §7–8).
## Requirements
### Requirement: Every transition carries its temporal record

Each stored transition and start SHALL carry the time it was first observed, the
time it was last observed, and the short-layer pair (accumulated value, the time
that value was last updated). Learning SHALL update the long-term count and the
short-layer pair in the same atomic operation that already persists the message,
so a reader can never observe one layer advanced and the other not.

Rows that existed before the temporal record was introduced SHALL be stamped
with the moment of introduction and an empty short layer. The system SHALL NOT
present those stamps as observations: any report or metric derived from them
SHALL state that the temporal record starts at that moment.

#### Scenario: A transition observed for the first time

- **WHEN** a message introduces a transition the chat has never produced
- **THEN** its first-seen and last-seen are the observation time, its long count is 1, and its short value is 1

#### Scenario: A transition observed again

- **WHEN** the same transition is observed later
- **THEN** its first-seen is unchanged, its last-seen is the new observation time, and its long count is incremented

#### Scenario: Learning fails midway

- **WHEN** persisting a message fails after some transitions were updated
- **THEN** neither layer retains a partial update for that message

#### Scenario: Rows predating the temporal record

- **WHEN** the temporal record is introduced over an existing chain
- **THEN** every existing row reports the introduction time as its first-seen and an empty short layer, and this is recorded as a known limitation rather than treated as history

### Requirement: The short counter is an exact sum of decayed observations

The short layer's effective weight at any moment SHALL equal the sum of each
past observation decayed by its own age at the configured half-life — that is,
an observation made one half-life ago SHALL contribute exactly half of a
just-made one, and an observation made now SHALL contribute exactly one.

The result SHALL NOT depend on the order in which observations arrived, and
between two observations the effective weight SHALL be non-increasing over time.
The stored representation SHALL be constant-size per transition: no per-event
rows and no time buckets.

#### Scenario: Contribution of an aged observation

- **WHEN** the effective weight is read one half-life after a single observation
- **THEN** it equals half the weight read immediately after that observation

#### Scenario: Order independence

- **WHEN** the same set of observations at the same times is applied in two different orders
- **THEN** the effective weight read at a later fixed moment is the same in both cases

#### Scenario: No observation between two reads

- **WHEN** the effective weight is read twice with no observation in between
- **THEN** the later read is not greater than the earlier one

### Requirement: Reading the short layer requires an explicit moment

Every computation of the short layer's effective weight SHALL take the moment it
is evaluated at as an explicit input. The system SHALL NOT read the machine
clock inside sampling, caching, or scoring.

This makes generation reproducible: the same seed, the same settings, and the
same evaluation moment SHALL produce the same reply, including when the pool
was served from a cache populated earlier.

#### Scenario: Reproducing a generation

- **WHEN** a generation is repeated with the same seed, settings, and evaluation moment
- **THEN** the output is identical, whether or not the transition pool was cached in between

#### Scenario: A cached pool read later

- **WHEN** a pool cached at one moment is blended at a later moment
- **THEN** the result equals blending a freshly read pool at that same moment — the cache stores the observation record, never a resolved weight

#### Scenario: The reading moment does not shift the distribution

- **WHEN** the same pool is blended at two different moments with no new observation in between
- **THEN** the blended weights are identical, because every candidate's short weight decays by the same factor and the layer is normalized within the pool
- **AND** a newer observation on one candidate does shift the weights, which is the only thing that can

### Requirement: Layers are blended over the union of their tokens

The sampled distribution SHALL be a weighted mix of the short and long layers,
normalized over the union of the tokens either layer offers, with the mixing
weight taken from the chat's mood. A token present in only one layer SHALL still
be reachable, with the weight that layer gives it.

Before mixing, the long layer's raw counts SHALL be compressed sublinearly so
that a count of ten thousand does not make every other token unreachable; the
compression SHALL preserve the order of preference between tokens.

When one layer is empty for a state, the blend SHALL degenerate to the other
layer rather than producing an empty or invalid distribution.

#### Scenario: Token known only to the fresh layer

- **WHEN** a token has short-layer weight but no long-layer count for the state
- **THEN** it appears in the blended distribution with positive probability

#### Scenario: Empty short layer

- **WHEN** no transition from a state has been observed since the temporal record began
- **THEN** the blended distribution equals the long layer alone and remains valid

#### Scenario: Dominant historical count

- **WHEN** one token's long count exceeds another's by three orders of magnitude
- **THEN** compression keeps the first preferred over the second while leaving the second's probability non-negligible

#### Scenario: Blended distribution is valid

- **WHEN** any two valid layer distributions are blended at any legal mixing weight
- **THEN** all probabilities are finite and non-negative and sum to one

### Requirement: A disabled blend is exactly the previous behavior

When the mixing weight is zero the system SHALL take the pre-existing sampling
path unchanged, producing output byte-identical to the version before the
temporal layer existed. In particular, the long layer's compression SHALL apply
only inside the blend path, so a disabled blend SHALL NOT reshape the sampled
weights.

The mixing weight SHALL default to zero for every mood, so that installing this
capability changes stored data but not generated text.

#### Scenario: Default configuration

- **WHEN** generation runs with default settings after the temporal layer is installed
- **THEN** the generation hash matches the frozen pre-temporal baseline

#### Scenario: Blend switched off at runtime

- **WHEN** the mixing weight is set back to zero without a restart
- **THEN** subsequent generations match the pre-temporal baseline again

### Requirement: Changing the half-life resets the short layer

The short counter is only meaningful against the half-life it accumulated under.
When the half-life changes, the system SHALL reset the short layer to empty and
SHALL report the reset explicitly to whoever made the change. The reset SHALL
NOT happen silently, and the long layer SHALL NOT be touched by it.

#### Scenario: Half-life changed at runtime

- **WHEN** the half-life setting is changed
- **THEN** the short layer is emptied, the change is acknowledged with an explicit warning that fresh-language memory was discarded, and long counts are unchanged

#### Scenario: Half-life set to its current value

- **WHEN** the half-life is set to the value it already has
- **THEN** the short layer is left intact

### Requirement: Chat data deletion covers the temporal record

Deleting a chat's data SHALL remove its temporal record along with the rest of
its model, leaving no row whose short layer or timestamps survive the deletion.

#### Scenario: Chat wipe

- **WHEN** a chat's data is deleted
- **THEN** no transition, start, or temporal value for that chat remains

