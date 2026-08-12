# Proposal: markov2r-phase5-reverse-index

## Why

Phase 5 (statistical lexical anchoring, ADR-012 Provisional) needs two things
that do not exist yet before any seeded generation can be attempted: reverse
order-2 lookups to grow a sentence head leftwards from a seed token, and an
incremental document-frequency aggregate to score seed candidates by IDF over
the full history — the `messages` table is retention-trimmed, so the existing
window-IDF cannot serve as the seed score's input. The roadmap splits the phase
deliberately (`00_STATUS.md`: reverse-index → seeded): data access first, so
its real cost — learning latency, storage growth, migration time — is measured
on its own before the experiment that reads it is built. This is M2R-400;
M2R-410/420/430 come as the next change.

## What Changes

- Reverse order-2 lookups ("which token preceded this two-token state, how
  often, and how recently") become answerable per chat. The forward
  `transitions` table already stores every fact needed; the design serves
  reverse lookups from it via one new secondary index instead of the duplicate
  `markov_reverse` table sketched in TZ §13 — a deliberate, recorded deviation
  (design D1) that removes the atomicity, backfill-drift and double-storage
  costs entirely.
- New table `markov_token_df` plus a per-chat `n_docs` counter (TZ §9.3):
  incremental document frequency — +1 per unique token per learned message —
  because raw messages are not stored and df can never be recomputed. Written
  in the same transaction as the rest of learning.
- Migration measured on `db_prod_copy`: index build time and actual storage
  growth recorded (the promotion gate later compares growth against the agreed
  cap, `storage_growth_max_share`); learning cost per message re-measured.
- `/clear confirm` removes the chat's df rows and its `n_docs` (reverse
  lookups die with the chat's transitions, which are wiped already).
- Removability by construction (ADR-012's cheap-refusal right): dropping the
  index and the df writes leaves the forward chain byte-identical.
- Nothing reads the new structures in this change: generation is untouched and
  `generation_hash` stays bit-identical by construction.

## Capabilities

### New Capabilities

- `generation-reverse-index`: reverse order-2 lookup availability and its
  consistency with learning, the incremental df aggregate, backfill semantics,
  removability/freeze behavior, wipe on chat deletion, and the "nothing reads
  them yet" contract.

### Modified Capabilities

<!-- none: no existing spec's requirements change; generation behavior is
     explicitly unchanged in this change -->

## Impact

- **Schema**: migration 020 — one secondary index on `transitions`, one new
  table, one new column on `chat_model_volume`.
- **Code**: learning path (`save_message_and_update_model`) gains the df
  upsert; small repository read API for reverse lookups (unused by the bot
  until M2R-410, exercised by tests); `/clear confirm` wipe extended.
- **Performance**: every transitions write now maintains one extra B-tree, and
  learning gains one small upsert batch — both measured and recorded;
  generation path untouched.
- **Risk**: the main cost named by ADR-012 is exactly this write path.
  Shipping it alone keeps the cheap-refusal option: if the cost is already
  bad, M2R-410 need not be built. TZ §13's `markov_reverse` schema is
  intentionally not followed; the TZ amendment rides with this change's sync.
