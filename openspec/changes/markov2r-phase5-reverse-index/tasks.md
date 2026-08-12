# Tasks: markov2r-phase5-reverse-index

Phase 5, first half (M2R-400). Data layer only: nothing here may change a
generated reply, and the closing check proves it with `generation_hash`.

## 1. Schema

- [x] 1.1 Migration `020_reverse_index_and_token_df.sql`: `idx_transitions_reverse (chat_id, w2, w3)` on `transitions`; `markov_token_df (chat_id, token, messages_seen)` WITHOUT ROWID; `chat_model_volume.n_docs INTEGER NOT NULL DEFAULT 0` (design D1, D2)
- [x] 1.2 Measure the migration on `db_prod_copy`: index build time and file-size delta (the storage number M2R-430 will compare against `storage_growth_max_share`); record both here — **~19 ms build, +1012 KiB = +13.9% of the DB** (VACUUM-normalized; the TZ-table shape would have been a multiple of this), well under the 0.35 cap
- [x] 1.3 Schema-tables test updated (`test_schema_contains_expected_tables`)

## 2. Learning path

- [x] 2.1 df upsert (+1 per unique token per message) and `n_docs` increment inside the existing `save_message_and_update_model` transaction (design D2, D4)
- [x] 2.2 Measure learning latency per message on `db_prod_copy` before/after (index maintenance + df batch — the write-amplification number named by ADR-012); record the delta here — paired same-DB comparison (index on vs dropped, 300 messages): **delta within noise (<0.1 ms)**; absolute learn ~1.4 ms median with everything on; main-branch baseline run landed at 1.8 ms median, i.e. run-to-run noise exceeds the component
- [x] 2.3 A learning failure leaves no partial write: covered by the existing transaction — add a test asserting df/n_docs and transitions move together or not at all

## 3. Read API (tests and M2R-410 only — no production caller)

- [x] 3.1 `get_reverse_transitions(chat_id, w2, w3)` returning the forward `TransitionRow` shape, served by the new index (design D5); `EXPLAIN QUERY PLAN` test pins the index, not a scan
- [x] 3.2 df readers: per-token `messages_seen` and the chat's `n_docs`
- [x] 3.3 Test: reverse lookup answers agree with forward rows after learning (same counts, same temporal record) — by construction, but the test is what keeps D1 honest if the design ever changes

## 4. Wipe and hygiene

- [x] 4.1 `/clear confirm` removes the chat's `markov_token_df` rows and zeroes/removes its `n_docs`; extend the orphaned-structures test
- [x] 4.2 df unaffected by message retention trimming — test with retention window smaller than learned history

## 5. Neutrality (the contract this change ships under)

- [x] 5.1 No reply-path read of reverse lookups or df: test asserts the generation pipeline issues no such queries
- [x] 5.2 `python -m tools.generation_hash --db db_prod_copy/markov.db` — hash identical to the frozen baseline (`5a72e2d4…`) with the migration applied — confirmed 2026-08-12
- [x] 5.3 CI smoke (`python -m tools.eval --smoke`) stays green — confirmed

## 6. Documentation

- [x] 6.1 TZ §13 amendment: reverse lookups served by index over `transitions` instead of `markov_reverse` (same §9.2 semantics); note in `docs/v2/04_MARKOV_2_0R_ADR.md` under ADR-012 that the storage shape changed and why (design D1)
- [x] 6.2 `docs/v2/00_STATUS.md`: Phase 5 row (reverse-index half) + next-session pointer to M2R-410
- [x] 6.3 `docs/ARCHITECTURE.md` / `GENERATION_PIPELINE.md`: one paragraph on the df aggregate and the index, and that nothing reads them yet — data-model table rows in ARCHITECTURE.md (+ the missing `markov_collocations` row rode along); GENERATION_PIPELINE untouched on purpose, generation did not change

## 7. Close-out

- [x] 7.1 `openspec validate --strict` green for this change
- [x] 7.2 Full test suite + lint/type checks — 1103 tests OK, ruff clean, mypy clean (2026-08-12)
- [ ] 7.3 Archive this change after merge — rides in the next change's PR (established convention)
