## Context

See proposal.md — Why. Design-relevant state of the system today:

- The chain lives in four tables (`starts`, `starts3`, `transitions`,
  `transitions3`) whose only payload is an integer `cnt`. Learning is one
  transaction per message in `database.py:201-349`; the whole process shares a
  single `aiosqlite` connection behind one lock, so every write is serialized
  already.
- Sampling weights a pool as `max(cnt, 1) ** power` in
  `weighted_next_choice` (`markov.py:736`). Phase 2 adjusts `power` per step
  from the pool's normalized entropy; Phase 1 computes that entropy in
  `pool_diagnostics` from the same counts.
- Phase 1 caches transition pools per chat and folds learning deltas into them
  incrementally instead of dropping the cache.
- The eval protocol demands bit-for-bit reproducibility, and `db_prod_copy`
  (2026-07-13, 1062 retained messages, ~67k chain rows) is the measurement
  surface.
- ADR-005 forbids rebuilding the live chain, so historical observation times
  cannot be recovered — only newly learned rows will ever have true ones.

## Goals / Non-Goals

**Goals:**

- One arithmetic definition of the decayed counter, shared by writer, reader,
  cache and tests, with no second implementation in SQL.
- Determinism preserved under a time-dependent weight.
- A neutral default that is byte-identical to today, provable by
  `generation_hash` — the same contract Phases 1 and 2 shipped under.
- Phase 3 measurable in this change rather than after a live accumulation
  window.

**Non-Goals:**

- Choosing the shipped α profile or compression shape — that is M2R-215
  (calibration), gated separately. This change ships the mechanism at neutral
  values and the grid that will later inform the choice.
- GC by age (M2R-220), the reverse index (Phase 5), and any change to message
  retention.
- Backfilling plausible historical timestamps. Fabricated history would poison
  every freshness metric that follows.

## Decisions

### D1. The short counter is stored as `(s_value, s_updated_at)` and decayed in Python

TZ §7.1's identity — observe: `s_value = s_value·2^(−Δt/hl) + 1`; read:
`s_eff = s_value·2^(−Δt/hl)` — is implemented once as a pure function taking
`(s_value, s_updated_at, now, half_life)`. Both the learn path and the read path
call it.

*Alternative rejected — do the decay in SQL.* `pow()` exists in SQLite 3.50
locally, but SQLite's math functions are a compile-time option; CI runs three
Python versions on ubuntu and the bot ships in a Docker image, and a silent
`no such function: pow` in the learn path is a bad way to discover that. Doing
it in Python also keeps `now` out of SQL, which D3 needs anyway.

*Alternative rejected — epoch-anchored additive form.* Storing
`S = Σ 2^(t_i/hl)` makes observation a pure `+=` with no read, but `S` doubles
every half-life and overflows a REAL within years, requiring a rebase job. Not
worth avoiding one indexed SELECT.

Cost: the learn transaction gains one SELECT of the rows this message touches
(bounded by message length, primary-key lookups) before the existing
`executemany`. The write becomes an explicit new value instead of `cnt + 1`.

### D2. Time is stored as integer Unix seconds, not the TEXT datetime used elsewhere

`messages.created_at` is TEXT `datetime('now')`. The chain's temporal columns are
read and subtracted on every sampled step; parsing text there would be a cost
paid for cosmetic consistency. The migration and the fixture builder are the two
places that convert between the two representations.

### D3. `now` is an explicit argument, never read inside sampling

The evaluation moment is captured once per generation (the same shape Phase 2
used for `EntropySampling`) and threaded into blending and cache reads. Nothing
below that call site reads the clock.

Without this, two runs of the same seed differ by however long the run took, and
the protocol's bit-for-bit requirement — the thing that makes every phase verdict
auditable — quietly stops holding. The eval runner and the tests pass a fixed
moment.

**Correction found during implementation.** The numeric influence of `now` turns
out to be nil: every candidate's short weight decays by the same factor between
two reads, and the short layer is normalized *within the pool*, so the factor
divides out. The blended distribution depends on when each token was last
observed *relative to the others*, never on when the pool is read. Two
consequences, both good and both now covered by tests: a cached pool cannot go
stale in the weights sense (no time-based cache invalidation needed), and the
fixture's choice of evaluation moment cannot bias the grid. The injection stays
— it keeps the clock out of sampling, which is what makes runs testable — but
the claim it defends is hygiene, not sensitivity.

### D4. The blend returns the pool unchanged when α = 0, so there is one code path

`blend_pool(pool, alpha, now, ...) -> list[tuple[str, float]]` returns its input
list untouched at α = 0. The sampler's clamp changes from `max(cnt, 1)` to
`max(w, EPS)`.

That clamp change is behavior-preserving on real data: `cnt` starts at 1 and only
increases, and nothing deletes rows (GC is not implemented), so `max(cnt, 1)` is
a no-op today. The neutrality proof asserts this explicitly — a scan for zero or
negative counts on `db_prod_copy` — rather than assuming it. Integer and float
inputs give identical results (`3 ** 0.5` and `3.0 ** 0.5` are the same float),
so the neutral path's arithmetic is unchanged bit-for-bit.

*Alternative rejected — a separate blended sampling path.* Two paths means the
neutral one drifts from the live one and the blend gets tested against a sampler
nobody uses.

Scale-invariance makes it safe to hand the sampler probabilities rather than
counts: `(c·p) ** power = c**power · p**power`, and the constant factor
normalizes away in `weighted_index_choice`.

### D5. Diagnostics are computed from the weights the sampler actually used

`pool_diagnostics` moves from raw counts to the blended weights. Entropy must
describe the distribution being sampled, not a distribution that exists only
before the blend.

This is deliberate and is the mechanism by which Phase 3 bears on the Phase 2
decision: Phase 2 was inert because 78.8% of steps had exactly one continuation
and 20.9% were near-uniform. Sublinear compression of the long layer flattens
peaked pools and the short layer adds mass to tokens the long layer has barely
seen, so both modes should move. The Phase 3 report prints the same entropy
histogram Phase 2's verdict printed, over the same 7034-step protocol, so the
two are directly comparable and the keep-or-delete call on Phase 2 rests on
numbers rather than on the argument above.

### D6. Pre-migration rows keep NULL timestamps instead of being stamped with the migration date

`ALTER TABLE ADD COLUMN` with NULL default touches metadata only. Readers treat
NULL `first_seen` as "predates the temporal record" and NULL `s_updated_at` with
`s_value = 0` as an empty short layer.

The audit suggested stamping the migration date. NULL is preferable for the same
reason the audit flagged the risk: a stamped date is indistinguishable from a
real observation two lines later in a report, while NULL cannot be mistaken for
history. It also avoids rewriting every row of four tables.

### D7. Half-life changes reset only the chat whose setting changed

The half-life is a chat-scoped runtime setting, so the reset is scoped the same
way: zero `s_value` and clear `s_updated_at` for that chat, leave `cnt`,
`first_seen` and `last_seen` alone, and answer the command with an explicit note
that fresh-language memory was discarded and rebuilds over roughly the new
half-life. Setting the value it already has is a no-op.

This is the project's first setting that discards stored data, which is why it
gets a requirement in `runtime-knob-validation` rather than only an
implementation.

### D8. The temporal eval fixture replays retained messages into a separate database

`tools/eval/temporal_fixture.py` reads `messages` (chat, `normalized_text`,
`created_at`) from a source snapshot, replays them in timestamp order through the
real learning arithmetic into a new database file, and writes true `first_seen`,
`last_seen`, `s_value`, `s_updated_at`. Deterministic given source and
parameters; the source is opened read-only.

Two honesty constraints, both enforced in the report rather than left to the
reader:

- The fixture's chain is built from the ~1000-message retention window, so it is
  an order of magnitude smaller than the prod copy's chain. **Its C0 is not the
  frozen C0 baseline.** Every Phase 3 delta is computed within the fixture
  (C0_fixture vs C2_fixture), and the report states this.
- The "historical" slice is only as old as the retention window, so the
  historical-meme check draws its n-gram list from the full snapshot's
  `chat_verbatim_ngrams` (which the retention never trims) rather than from the
  fixture's own old rows.

### D9. `phase3_temporal` is registered before the grid runs, and is two-sided

Registered in `eval_thresholds.yaml` in its own commit before any Phase 3 number
exists, mirroring `phase2_entropy`: freshness must rise significantly, the
historical-meme check must hold, copying must not rise significantly, affinity
without copies must not drop significantly, p95 must stay in budget. A one-sided
"freshness went up" gate would pass an arm that simply forgets the chat.

## Risks / Trade-offs

- **The fixture is not live accumulation** → its time span is the retention
  window, its corpus is small, and the fresh/historical split is a construction.
  Mitigation: deltas only within the fixture; report prints span, slice size and
  provenance; the gate verdict is explicitly labelled as measured on a
  reconstructed snapshot. Live re-measurement after an accumulation window stays
  on the roadmap for M2R-215.
- **Historical `first_seen` is unrecoverable** → age-based GC and any
  longitudinal metric start from this migration. Mitigation: NULL rather than a
  fabricated date (D6); stated in the report and in `00_STATUS.md`.
- **Per-step blending costs latency** → an extra pass over each pool with a
  `2**x` per token. Current p50/p95 is 19/29 ms against a 150 ms budget, so
  there is room, but "there is room" is not a measurement. Mitigation: p50/p95
  reported per arm; if the blend path is hot, the decay factor is constant
  within one generation for a fixed `now` and can be hoisted per state.
- **Learn path gains a SELECT** → one extra indexed read per message.
  Mitigation: measured on the prod copy before merge; it is bounded by message
  length and rides inside the existing transaction and lock.
- **Cached pools become time-dependent** → a pool cached at one moment must
  decay when read at another. Mitigation: the cache stores the raw pair and
  `s_eff` is resolved at read against the injected `now`; a property test covers
  cached and uncached reads producing the same output.
- **Blend may erase long memory** → the failure mode the meme check exists to
  catch; it is a gate component, not a note.
- **Neutrality could regress silently** → same mitigation as Phases 1-2:
  `generation_hash` against the frozen baseline plus the characterization suite,
  both at default settings.

## Migration Plan

1. Migration `018_temporal_layer.sql`: sixteen `ALTER TABLE ADD COLUMN`
   statements (four columns × four chain tables), NULL/0 defaults, no data
   rewrite. Cost measured on `db_prod_copy` before merge; the framework runs it
   at startup inside its own transaction.
2. Ship with every α at 0. Generation is byte-identical; only the schema and the
   learn path change on the live bot.
3. The short layer starts filling from the first message learned after the
   restart. Nothing else has to happen for the phase to be correct.
4. **Rollback**: set every α back to 0 via `/set` — no restart, and the
   documented revert check proves it restores baseline output. The columns stay;
   they are inert when unused and dropping them is neither needed nor attempted
   (forward-only migrations, ADR-005).

## Open Questions

- The shipped α profile and compression shape stay at their neutral/TZ-default
  values in this change; the grid here informs, but does not decide, M2R-215.
  Deferring this changes no spec, no task and no interface.
