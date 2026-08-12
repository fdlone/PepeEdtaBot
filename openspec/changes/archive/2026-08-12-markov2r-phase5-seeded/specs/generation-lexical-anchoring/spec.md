## Purpose

Grows a reply around an anchor token the chat actually uses — chosen by
association strength, not raw frequency, and assembled in both directions from
the anchor — and lets those seeded candidates compete without privilege
(normative sources: `docs/v2/02_MARKOV_2_0R_TZ.md` §9, ADR-008, ADR-012
Provisional). This is an experiment: it ships disabled, and raising its weight
requires the phase's gate to pass.

## ADDED Requirements

### Requirement: Seeds are chosen by a composite score, not by IDF alone

The system SHALL rank candidate seed tokens by a score that combines how
informative the token is (inverse document frequency over the chat's whole
history), how well supported it is (its total use in the model), and whether
its branching in both directions is usable. A token that is informative only
because it is unique junk SHALL NOT be chosen: high IDF with negligible support
scores low.

Branching quality SHALL be a band, not a threshold: a token whose forward or
reverse branching is below a floor is unusable (generation would stall), a
token whose branching is far above an ideal is a weak anchor, and the middle is
preferred. The band's bounds SHALL be configurable.

Tokens shorter than a configured length and stopwords SHALL be excluded before
scoring.

#### Scenario: Junk unique token is not chosen

- **WHEN** a token appears once, has maximal IDF, and has almost no support in the model
- **THEN** its seed score is low and it is not selected as an anchor

#### Scenario: Well-used distinctive token is chosen

- **WHEN** a token is distinctive (high IDF) and well supported, with branching inside the band
- **THEN** it scores high and is eligible as an anchor

#### Scenario: Branching outside the band

- **WHEN** a token's branching is below the floor or far above the ideal
- **THEN** its branching-quality factor drives the seed score down

### Requirement: Generation from a seed grows in both directions

Given a chosen seed token, the system SHALL assemble a candidate by growing the
tail forward on the forward chain and the head backward on the reverse order-2
index, so the anchor can sit anywhere in the reply rather than only at its
start. The entropy and temporal rules that govern forward generation SHALL
apply to the reverse direction as well. The head/tail length split SHALL be
configurable within the existing reply-length budget.

#### Scenario: Anchor appears mid-reply

- **WHEN** a seed candidate is generated for an anchor token
- **THEN** the anchor may appear in the middle of the reply, with tokens both before and after it drawn from the chain

#### Scenario: Reverse data absent for the anchor

- **WHEN** the reverse index holds no predecessor for the anchor's state
- **THEN** the head simply does not grow past the anchor, and the candidate is still assembled from the tail

### Requirement: Seeded candidates compete without priority

Seeded candidates SHALL join the best-of-N pool and be judged by the same
scorer as every other candidate, with no bonus for being seeded (ADR-008). The
share of the pool that is seeded SHALL be configurable.

#### Scenario: Seeded candidate wins on merit

- **WHEN** a seeded candidate and ordinary candidates are scored together
- **THEN** the seeded candidate wins only if its score is best, by the same rule applied to all candidates

### Requirement: The seeded branch fails transparently

When no seed token clears the configured minimum score, or generation from the
chosen seed yields nothing usable, the system SHALL skip the seeded branch and
produce the reply exactly as it would without seeding. A chat with no usable
anchor SHALL never fail to answer for that reason.

#### Scenario: No token clears the minimum

- **WHEN** no candidate seed token reaches the minimum seed score
- **THEN** the seeded branch is skipped and generation proceeds unseeded

### Requirement: Neutral default leaves replies unchanged

The seeded-candidate share SHALL default to zero, so installing this capability
changes what the system can do without changing what it says. With the default,
the reverse index and df aggregate SHALL NOT be read on the reply path, and
generation SHALL be byte-identical to the previous version for the same inputs.
Raising the default SHALL require the phase's promotion gate to pass.

#### Scenario: Default configuration

- **WHEN** generation runs with default settings after this capability is installed
- **THEN** the produced text is identical to what the previous version produced for the same inputs, and no seeded read occurs

### Requirement: Freezing the experiment is safe

Disabling the feature (seeded share back to zero) SHALL stop all seeded reads
from the next reply onward without altering the chain, the reverse index, the
df aggregate, or any stored text. This is ADR-012's cheap-refusal path and
SHALL require no restart.

#### Scenario: Feature disabled at runtime

- **WHEN** the seeded share is set to zero
- **THEN** seeded generation stops immediately, the reverse and df structures are untouched, and unseeded replies are unaffected
