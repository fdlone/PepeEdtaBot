# Design: markov2r-phase5-seeded

## Context

See proposal.md — Why. Design-relevant state:

- M2R-400 shipped the reverse order-2 lookup (`get_reverse_transitions`) and
  the df aggregate (`get_token_df`, `get_n_docs`); nothing reads them for
  generation yet. The spec `generation-reverse-index` guarantees a reverse
  lookup agrees with the forward chain by construction.
- `MarkovGenerator` already resolves explicit `seed_tokens` to a **start** (a
  stored start 3/2-gram) and walks forward — the L1 hot-ngram channel. That is
  anchor-at-the-start; Phase 5 needs anchor-anywhere, which is what the reverse
  index is for. The two mechanisms are different and coexist.
- Candidate scoring already carries a window-IDF term (`idf_context_relevance`,
  weight 1.6, live since before 2.0R) — this is TZ §9.7 and is **already
  shipped**. This change does not touch it (see D6).
- The generation pipeline's neutrality contract is `generation_hash`: with a
  feature at its neutral setting, output is byte-identical, which depends on
  the RNG being consumed in the same order.
- The eval runner (`tools/eval/run.py`) already populates per-arm state on a DB
  copy before generating (C3 calls `analyze_chat_memes`) — precedent for a
  seeded arm populating what it needs.
- df on `db_prod_copy` is empty (0 rows): it accumulates only after the
  migration-020 prod restart.

## Goals / Non-Goals

**Goals:**

- An anchor chosen by association strength, robust to junk unique tokens.
- A reply assembled in both directions around that anchor.
- Seeded candidates competing on merit, switchable, neutral by default.
- The `phase5_promotion` gate computable, honest about df accumulation.

**Non-Goals:**

- The promotion decision itself (M2R-430) — needs prod-accumulated df.
- Re-pointing candidate-scoring IDF (§9.7) from window to full-history df (D6).
- Any order-3 reverse data (excluded at M2R-400).
- Raising any default. The gate decides that.

## Decisions

### D1. Seed scoring is a pure module over three inputs

`app/core/seed.py`, no SQL/clock/RNG (the pattern of `collocations.py`):

```
seed_score(t) = normalized_idf(t) × support_factor(t) × branching_quality(t)
```

- `normalized_idf(t) = idf(t) / max_idf_in_message`, `idf(t) = log(n_docs /
  (1 + df(t)))`. Normalizing by the message's own max keeps the score
  comparable across messages of different rarity (TZ §9.4). `n_docs == 0` or
  `df` unknown → idf degenerate → the token is not scored (feeds the
  transparent fallback, not a crash).
- `support_factor(t)` — a saturating ramp on the token's total model count, the
  same shape as the collocation support factor: rare-but-real is good, almost
  never seen is bad. Reuse `collocations.support_factor` rather than a second
  implementation.
- `branching_quality(B)` — a **trapezoid**, not a threshold (TZ §9.4):
  0 below `MARKOV_SEED_BRANCH_MIN`, 1 across `[MIN, IDEAL]`, a linear decay to a
  floor by `MARKOV_SEED_BRANCH_MAX`. Applied to `min(forward_branch,
  reverse_branch)` — a seed unusable in either direction is unusable. Bounds are
  knobs, calibrated by eval.

Stopwords and tokens shorter than `MARKOV_SEED_MIN_TOKEN_LEN` are dropped before
scoring (reuse the lexicon stopword set). A degenerate single-token corpus and
any legal counts must yield a finite score — a property test pins it, as for
collocations.

### D2. Bidirectional assembly reuses the forward walk, mirrored

The seed anchor `t` becomes a one-token state; the candidate is
`head_reversed + [t] + tail`:

- **Tail**: an ordinary forward walk seeded at a state containing `t` — the
  existing walk machinery, entropy/temporal rules included, unchanged.
- **Head**: a reverse walk over `get_reverse_transitions` — at each step the
  predecessor pool of the current order-2 state is sampled by the **same**
  weighted-choice + entropy/temporal helpers the forward walk uses (the reverse
  row carries the identical `(cnt, s_value, s_updated_at)` shape, so the temporal
  blend applies verbatim). The head grows leftward until the reverse pool is
  empty or the head's share of the length budget is spent.
- Length: `MARKOV_SEED_HEAD_SHARE` splits the existing reply-length budget
  between head and tail; the total stays inside `max_reply_tokens`.

The reverse walk holds its fetched pools the same copy-on-write way the forward
walk does, so an interleaved learn cannot mutate a pool mid-assembly.

*ponytail:* the reverse walk is a mirror of the forward stepper, not a second
engine — factor the shared step (pool → weighted choice → append) so head and
tail cannot drift apart in sampling behavior.

### D3. Seeded candidates enter the pool by ratio, no priority

`MARKOV_SEEDED_CANDIDATE_RATIO` (0–0.7, default 0) sets how many of the
`candidate_target` slots are filled by seeded assembly instead of ordinary
generation. Seeded candidates are scored by the existing scorer with no bonus
(ADR-008) and compete in the same softmax/argmax selection. The seed token for
each seeded attempt is drawn from the top of the seed-score ranking over the
current message's tokens.

Telemetry: a generation counts as *seeded-present* if at least one seeded
candidate entered the pool, and *seeded-won* if the selected candidate was
seeded — the two denominators of TZ §9.6, `note_seeded(present, won)` on the
generation telemetry.

### D4. The eval C4 arm populates df from the retained window, gate stays honest

The runner, for the C4 arm only, builds df by learning the copy's retained
messages into `markov_token_df` at the fixed evaluation moment (the C3
precedent, mirrored for df). This is **window-approximated df**, not
prod-accumulated — enough to exercise the seeded branch and measure
`seeded_present_rate` / `seeded_win_rate_given_present` and the affinity delta,
but not the real full-history IDF the seed score is designed for.

Therefore `phase5_promotion` reports `insufficient data` — never `pass` —
while the run's df is window-approximated, exactly as Phase 4's gate withholds
on the missing manual rating. The automatic conditions are computed and printed;
the verdict waits for a protocol run over prod-accumulated df (M2R-430). The
report's phase5 row stops being the hardcoded "does not exist" stub.

### D5. Neutrality is a short-circuit before any seeded read

`ratio == 0` returns before the seed ranking is computed, before the df/reverse
reads, before any extra RNG draw — so the RNG-consumption order is identical to
today and `generation_hash` is unchanged by construction. The seeded path draws
its own RNG only inside the `ratio > 0` branch. A test asserts the reply
pipeline issues no reverse/df query at the default, paired with the hash guard.

### D6. Candidate-scoring IDF (§9.7) is left as the shipped window-IDF

§9.7's candidate IDF term already exists as window-IDF (`idf_context_relevance`,
`get_context_idf`) and is live. The df aggregate this phase adds is for the
**seed** score (§9.4). Re-pointing the candidate term to full-history df would
be a separate behavioral change with its own hash impact and its own
justification — it does not ride silently inside the seeded-generation change.
Recorded so it is a decision, not an omission; a future change can add a knob to
switch the candidate IDF source if eval wants it.

## Risks / Trade-offs

- **Empty df until prod accumulates** → the seed score is degenerate on a fresh
  prod and on the eval copy; handled by the transparent fallback (no anchor →
  unseeded) and by the honest gate. Same accumulation-window shape as Phase 3.
- **Reverse walk latency** → seeded candidates cost extra walks; only when
  ratio > 0, and p95 is a gate criterion measured in the run.
- **Two samplers drifting** → mitigated by factoring the shared step (D2); a
  test asserts a reverse walk over a hand-built chain produces the expected
  predecessor distribution.
- **A third phase that could return null** → possible and fine: the gate is
  pre-registered, so a null seeded result is reportable with numbers, and
  ADR-012 already frames the cheap refusal.

## Migration Plan

No schema change — M2R-400 already shipped the structures. Ship with
`ratio = 0`; the feature is inert until a `/set` raises it, and the reverse/df
reads never fire at the default. Rollback: `ratio` back to 0, no restart.

## Open Questions

- The exact head/tail share and branching-band bounds are eval-calibration
  questions, not interface questions — they change no spec and no task boundary.
  A calibration grid (like Phase 2/3's) can answer them once df has accumulated.
