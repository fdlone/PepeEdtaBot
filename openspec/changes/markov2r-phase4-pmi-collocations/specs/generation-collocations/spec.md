## Purpose

Distinguishes the chat's real recurring phrases from merely frequent word pairs
using association strength rather than raw counts, and lets those phrases nudge
candidate selection without ever changing how text is tokenized or stored
(normative sources: `docs/v2/02_MARKOV_2_0R_TZ.md` §10, ADR-016).

## ADDED Requirements

### Requirement: Memes are ranked by association, not frequency

The system SHALL rank candidate collocations by a score combining the strength
of association between the two tokens with how well supported and how recent the
pair is. A pair that is common only because its parts are common SHALL rank
below a pair whose parts occur together far more often than chance predicts.

A pair SHALL NOT be considered at all below a configured minimum number of joint
occurrences. This threshold is mandatory: on real data the overwhelming majority
of pairs occur exactly once, and a single co-occurrence carries no evidence of
anything.

#### Scenario: Frequent pair of frequent tokens

- **WHEN** two individually very common tokens co-occur no more often than their individual frequencies predict
- **THEN** the pair scores low and does not enter the registry as a meme

#### Scenario: Rare pair that always occurs together

- **WHEN** two tokens are individually uncommon but nearly always appear together, above the support threshold
- **THEN** the pair scores high

#### Scenario: Pair seen once

- **WHEN** a pair's joint count is below the configured minimum
- **THEN** it is excluded before scoring, and no score is computed or stored for it

### Requirement: The analyzer runs on a schedule and states its cost

The analysis SHALL run as part of the bot's existing periodic maintenance rather
than requiring a manual invocation, so the registry reflects the chat as it is
now rather than as it was when someone last ran a script.

The pass SHALL apply its support threshold when reading from storage rather than
after loading, and its measured duration SHALL be recorded in the change that
introduces it and re-checked when the corpus grows. A maintenance pass SHALL NOT
block message handling for longer than its stated budget.

#### Scenario: Scheduled pass

- **WHEN** the maintenance interval elapses and a message arrives
- **THEN** the analyzer runs once, updates the registry, and does not run again until the next interval

#### Scenario: Pass fails

- **WHEN** the analyzer raises
- **THEN** the failure is reported through the existing maintenance-alert path, the previous registry contents remain usable, and message handling continues

### Requirement: The collocation registry has an explicit lifecycle

Each chat SHALL have its own registry of collocations, each carrying a status of
candidate, active or retired, and the registry SHALL be capped at a configured
maximum number of entries. Only active entries SHALL influence scoring.

Retirement SHALL be safe by construction: it removes a scoring rule and SHALL
NOT alter the chain, the tokenization, or any stored text. Deleting a chat's
data SHALL remove its registry.

#### Scenario: Registry at capacity

- **WHEN** more pairs qualify than the configured maximum
- **THEN** the highest-scoring pairs are kept and the rest do not enter the registry

#### Scenario: Collocation retired

- **WHEN** an active collocation is retired
- **THEN** it stops affecting scoring from the next reply onward, and the chain's transitions and counts are unchanged

#### Scenario: Chat wipe

- **WHEN** a chat's data is deleted
- **THEN** no collocation entry for that chat remains

### Requirement: Collocations influence scoring only

An active collocation MAY raise the score of a candidate reply that contains it
as an unbroken sequence, and MAY lower the score of a candidate that starts the
collocation and continues with something else. It SHALL NOT change tokenization,
SHALL NOT be stored as a single token, and SHALL NOT alter what the chain
learns.

The penalty SHALL apply only when the chain actually offered the collocation's
right token as a continuation at that point. A candidate that "broke" a
collocation the chain could not have completed SHALL NOT be penalized — that is
the corpus's limitation, not the candidate's fault.

#### Scenario: Candidate reproduces a collocation

- **WHEN** a candidate contains an active collocation as an adjacent pair
- **THEN** its score is raised by the configured bonus

#### Scenario: Candidate breaks an available collocation

- **WHEN** a candidate contains the left token followed by a different token, and the chain does hold a transition to the collocation's right token from that state
- **THEN** its score is lowered by the configured penalty

#### Scenario: The right token was never available

- **WHEN** a candidate contains the left token followed by a different token, and the chain holds no transition to the collocation's right token from that state
- **THEN** no penalty is applied

#### Scenario: Tokenization is untouched

- **WHEN** collocations are active
- **THEN** learning, storage and generation tokenize exactly as they did before, and no glued token appears anywhere

### Requirement: Neutral defaults leave replies unchanged

The bonus and the penalty SHALL default to zero, and the meme-aware ordering of
hot n-grams SHALL be off by default, so installing this capability changes what
the system knows without changing what it says. Raising the defaults SHALL
require the phase's gate to pass.

#### Scenario: Default configuration

- **WHEN** generation runs with default settings after this capability is installed
- **THEN** the produced text is identical to what the previous version produced for the same inputs

### Requirement: Meme lists are chat content

A ranked meme list contains verbatim phrases from a private chat. Such a list
SHALL NOT be committed to the repository, and only aggregate results — counts,
shares, verdicts — SHALL appear in committed reports.

#### Scenario: Manual rating recorded

- **WHEN** the top meme list is rated by hand
- **THEN** the rating sheet stays outside the repository and the committed report carries only the aggregate outcome
