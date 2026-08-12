## Purpose

Makes the data statistical lexical anchoring will need available — reverse
order-2 lookups and an incremental document-frequency record — without
changing a single generated reply. This change is the data layer only
(M2R-400, TZ §9.2, §9.3, ADR-012 Provisional); anything that reads these
structures for generation arrives with the seeded-generation change and its
own gate.

## ADDED Requirements

### Requirement: Reverse order-2 lookups agree with the forward chain

For any two-token state of a chat, the system SHALL be able to answer which
tokens preceded that state, how often, and with the same temporal record (long
count and short layer) the forward chain carries. The answer SHALL always
agree with the forward chain: a learned message updates both views atomically,
and a failure while learning leaves neither view updated.

Reverse lookups SHALL be provided only for order 2. Order-3 reverse data is
out of scope by decision (TZ §9.2): a sentence head is short and passes the
common scorer, and the restriction is what keeps the storage risk acceptable.

#### Scenario: Message learned

- **WHEN** a message is learned
- **THEN** reverse lookups over its states reflect it immediately, with the same counts and observation moment as the forward chain

#### Scenario: Learning fails mid-way

- **WHEN** any part of learning a message fails
- **THEN** neither forward nor reverse views contain any part of that message

### Requirement: Document frequency is incremental and never recomputed

The system SHALL maintain, per chat, a document-frequency aggregate — for each
token, in how many learned messages it has appeared (+1 per unique appearance
per message) — and a total count of learned messages. Both SHALL be updated in
the same transaction as the rest of learning.

Raw messages are not retained beyond the existing retention window, so this
aggregate is the only durable record of document frequency: the system SHALL
NOT derive df by re-reading stored messages.

#### Scenario: Token repeats inside one message

- **WHEN** a learned message contains the same token three times
- **THEN** that token's document frequency rises by exactly one, and the chat's message count rises by exactly one

#### Scenario: Retention trims old messages

- **WHEN** messages older than the retention window are deleted
- **THEN** the document-frequency aggregate and the message count are unchanged

### Requirement: Installation covers already-learned chats

Installing this capability SHALL make reverse lookups available for chats that
were learned before it existed, preserving their counts and temporal record.
The df aggregate SHALL NOT be backfilled from the retention window: inventing
full-history frequencies from a trimmed window would be inventing history, so
df starts counting from installation.

#### Scenario: Existing chat after migration

- **WHEN** the migration runs on a database with existing forward transitions
- **THEN** every forward transition is reachable through reverse lookup with the same count, and the migration's duration and storage growth are measurable

### Requirement: Nothing reads the reverse structures for generation

Generation SHALL be byte-identical with this capability installed: no reply
path may read reverse lookups or the df aggregate until the seeded-generation
change lands behind its own gate. Freezing the experiment later (ADR-012)
means exactly this state — data maintained, never read by generation.

#### Scenario: Same seed, same corpus

- **WHEN** generation runs with a fixed seed before and after this capability is installed
- **THEN** the produced text is identical

### Requirement: The capability is removable without touching the chain

Removing this capability (ADR-012's cheap-refusal path) SHALL leave the
forward chain byte-identical and learning working: reverse lookup support and
the df aggregate can be dropped without rewriting, migrating or re-learning
any forward data.

#### Scenario: Capability removed

- **WHEN** reverse lookup support and df maintenance are removed
- **THEN** forward aggregates are unchanged, learning continues to work, and generation is unaffected

### Requirement: Chat deletion removes the chat's anchoring data

`/clear confirm` SHALL leave no reverse-lookup answer, no document-frequency
row and no message count for the wiped chat.

#### Scenario: Chat wipe

- **WHEN** a chat's data is deleted
- **THEN** reverse lookups for that chat return nothing, and no document-frequency row or message count for it remains
