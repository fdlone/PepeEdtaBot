# Design: markov2r-phase5-reverse-index

## Context

See proposal.md — Why. Design-relevant state:

- `transitions` already stores every order-2 fact as full rows
  `(chat_id, w1, w2, w3, cnt, first_seen, last_seen, s_value, s_updated_at)`,
  PK `(chat_id, w1, w2, w3)`. A "reverse order-2 transition" — given state
  `(a, b)`, which `x` preceded it — is exactly the same rows read by
  `WHERE chat_id=? AND w2=a AND w3=b`, answering `w1` with its counts and
  temporal record. No fact needed by reverse lookups is absent from this table.
- Migration 008 dropped secondary indexes whose keys were leftmost prefixes of
  a PK: write amplification with **no read benefit**. That is the recorded
  rule — not "no secondary indexes ever".
- `messages` is retention-trimmed per chat, so document frequency cannot be
  recomputed later; TZ §9.3 already prescribes an incremental aggregate.
- `chat_model_volume` is the existing per-chat metadata table (PK `chat_id`) —
  the place TZ §13 points at for `N_docs`.
- The whole process shares one SQLite connection behind one lock: everything
  the learn path adds is time the bot is not handling messages.

## Goals / Non-Goals

**Goals:**

- Reverse order-2 lookups whose answers can never disagree with the forward
  chain.
- A df record that survives message retention.
- The write path's true cost as numbers: index build, storage growth, learning
  latency delta.

**Non-Goals:**

- Seed scoring, bidirectional generation, seeded candidates (M2R-410),
  IDF scoring knobs (M2R-420), the promotion decision (M2R-430).
- Any change to what the bot says: `generation_hash` must stay bit-identical.
- Order-3 reverse data (TZ §9.2 excludes it deliberately).

## Decisions

### D1. Reverse lookups are an index over `transitions`, not a second table

TZ §13 sketches a separate `markov_reverse` table (`chat_id, token,
state_hash, count, s_value, s_updated_at, last_seen`). This design deviates:
one secondary index `idx_transitions_reverse (chat_id, w2, w3)` on the
existing table, and reverse reads become an ordinary indexed query.

Why the index wins on every axis that matters here:

- **Consistency by construction.** The reverse "view" IS the forward row.
  Atomic learning, crash-safety, decay arithmetic, `/clear` — all inherited,
  none re-implemented. A second table would duplicate `cnt`, `s_value`,
  `s_updated_at` and could drift from the forward row on any future bug; the
  index cannot drift, SQLite maintains it in the same transaction.
- **Storage.** The index stores key columns + rowid; the TZ table would
  duplicate the key AND the counters AND its own PK b-tree. On the same data
  the index is a fraction of the duplicate table's footprint — and storage
  growth is one of the promotion gate's four criteria.
- **Backfill.** `CREATE INDEX` on existing data *is* the backfill — one
  statement, measured, no reconciliation code, nothing to get out of sync.
- **Removal.** `DROP INDEX` is the whole rollback (ADR-012's cheap refusal).

Against migration 008's rule: that migration removed indexes with no read
benefit. This one serves a query the PK cannot (`w2, w3` is not a leftmost
prefix), and its write cost replaces the strictly larger cost of maintaining a
duplicate table. The TZ §13 amendment rides with this change's sync.

*Write cost is not free*: every transitions upsert now maintains one extra
b-tree. This is measured (task 4.2) — it is the number M2R-430 will weigh.

### D2. df lives in `markov_token_df`; `n_docs` joins `chat_model_volume`

`markov_token_df(chat_id, token, messages_seen)`, PK `(chat_id, token)`,
WITHOUT ROWID — same shape and access pattern as `chat_verbatim_ngrams`.
`n_docs` is a new `INTEGER NOT NULL DEFAULT 0` column on `chat_model_volume`,
exactly where TZ §13 puts it; the volume row is already upserted per learned
message, so `n_docs = n_docs + 1` rides the same statement.

Per message the df write is one `executemany` over the message's *unique*
tokens — the same batch shape as the hot-ngram bump, which has lived on the
learn path since L1 without a knob.

### D3. No runtime kill switch

The proposal draft considered one; it is dropped on precedent. Every existing
learn-path write (hot n-grams, emoji stats, verbatim n-grams, temporal record)
ships without a toggle — M2R-030's kill switch guarded a *behavioral cache
change*, not a write. The index cannot be toggled by a knob anyway (only
dropped), and a df toggle would leave silently undercounted history — a gap a
later reader cannot distinguish from "the chat was quiet". Rollback is a
migration (`DROP INDEX` + stop writing df), per D1/ADR-012.

### D4. df counts every token, filtering happens at read time

TZ §9.3 defines the aggregate as "+1 per unique token per message" with no
exclusions, and TZ §9.4 puts the stopword/length filter in seed *scoring*.
Storing everything keeps the writer trivial and the aggregate reusable; the
seed scorer (M2R-410) filters. Punctuation tokens inflate the table by a
bounded, measured amount — recorded with the storage numbers.

### D5. A read API exists but only tests and M2R-410 may call it

`MarkovRepo` (or a sibling following `BaseRepo`) gains
`get_reverse_transitions(chat_id, w2, w3)` returning the same `TransitionRow`
shape as forward reads, plus df readers. No production caller in this change —
the spec's "nothing reads them for generation" requirement is enforced by the
byte-identity check (`generation_hash`) and by tests asserting the reply path
issues no reverse/df queries.

## Risks / Trade-offs

- **Permanent write amplification on the hottest table** → measured on the
  prod copy before merge; if the learning delta is already ugly, that is a
  finding for M2R-430, not a surprise after M2R-410 is built. `DROP INDEX` is
  always available.
- **df starts empty for existing chats** → accepted and spec'd: inventing
  full-history frequencies from a trimmed window would bias IDF toward
  whatever survived retention. Seed scoring lands weeks later; df accumulates
  in the meantime — same accumulation-window pattern Phase 3 used for the
  short layer.
- **Deviation from TZ §13** → recorded here and in the proposal; the TZ
  amendment (index instead of `markov_reverse`, same §9.2 semantics) rides
  with this change's sync so the docs never contradict the schema.

## Migration Plan

1. Migration `020_reverse_index_and_token_df.sql`: `CREATE INDEX
   idx_transitions_reverse`; `CREATE TABLE markov_token_df`; `ALTER TABLE
   chat_model_volume ADD COLUMN n_docs`. Measured on `db_prod_copy` (index
   build time + file size delta), numbers recorded in tasks.md.
2. Learn path gains the df upsert + `n_docs` increment inside the existing
   transaction.
3. **Rollback**: drop the index, stop the df writes (revert), leave
   `markov_token_df` inert — reading nothing costs nothing; dropping the table
   is a separate migration and not required to disable the capability.

## Open Questions

- None blocking. The seed-score knobs, branching band and candidate-ratio
  questions belong to M2R-410's proposal.
