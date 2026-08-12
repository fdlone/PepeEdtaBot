## Why

The bot's sense of "what this chat says" is purely frequency-based: `chat_hot_ngrams`
counts occurrences in a 7-day window and promotes whatever is common. Frequency
cannot tell a meme from a filler — measured on the prod copy, the top of a raw
frequency-style ranking is dominated by one-character tokens, while the pairs
with the highest association strength (normalized PMI 0.91–0.94, lift 472–672)
sit on long content words. TZ §10 fixes this with association measures instead
of counts: PMI, lift and LLR with mandatory support and recency thresholds.

Phase 4 is also the first phase since Phase 1 that is not blocked by the wall
Phases 2 and 3 both hit. Those two re-weighted per-step choices, and this corpus
has almost none to re-weight — 78.8% of steps have exactly one continuation on
the full chain, 99.0% on the temporal fixture. Phase 4 works one level up, on
whole candidates: the generator produces 5–10 of them per reply and the scorer
chooses between them, so there is real material for a scoring signal to act on.

## What Changes

- **M2R-300 — association analyzer**: PMI, lift and LLR over the chat's bigrams,
  with `meme_score = normalized_pmi × support_factor × recency_factor`. Support
  and recency thresholds are mandatory, not optional tuning.
- **Runs inside the existing daily maintenance** (owner decision 2026-08-12),
  not as a script someone has to remember. Measured on the prod copy: **41 ms**
  per pass with the support threshold applied in SQL (88 ms without it —
  88% of bigrams occur exactly once and can never be memes).
- **New table `markov_collocations`** (TZ §13): per-chat, `status`
  candidate|active|retired, capped at `MARKOV_COLLOCATION_MAX_ENTRIES`, visible
  in `/stats`, wiped by `/clear confirm`.
- **M2R-310 — meme-aware hot n-grams**: `meme_score` replaces pure frequency in
  the hot n-gram selection, behind a knob, neutral by default until the gate
  passes.
- **M2R-320 — collocation scoring (ADR-016)**: a candidate containing an active
  collocation intact gets `MARKOV_COLLOCATION_BONUS`; a candidate that breaks
  one gets `MARKOV_COLLOCATION_BREAK_PENALTY`, **but only when the right token
  was statistically available** — punishing the chain for a continuation it
  never had would be punishing it for the corpus. Both default to 0.
- **NOT in scope, deliberately**: gluing collocations into single tokens.
  ADR-016 rules it out — a one-way tokenization change with no rebuild path
  (ADR-005) creates permanently coexisting incompatible representations
  (`KEK_LOL` vs `кек`+`лол`) that no retirement can clean up. Returning to it
  needs its own proposal with a solved `tokenization_version` story.
- **Manual gate**: the top-20 memes by `meme_score` are rated by hand — real
  local meme / merely frequent / junk — and the phase passes at ≥70% real
  (doc 05 §5). This needs human raters and is the phase's operational
  dependency, not a formality.

## Capabilities

### New Capabilities
- `generation-collocations`: what an association-scored meme is, how the
  collocation registry behaves over its lifecycle (candidate → active →
  retired), and how active collocations may and may not influence candidate
  scoring.

### Modified Capabilities
- `generation-eval`: adds the `phase4_memes` gate and the manual-rating
  artifact, including the rule that rated meme lists are chat content and stay
  out of the repository.
- `generation-telemetry`: the collocation registry's size and the bonus/penalty
  application counts become observable, so "configured but never applied" is
  distinguishable from "working".
- `runtime-knob-validation`: bounds for the new knobs on both paths.

## Impact

- **Schema**: new table `markov_collocations` (migration `019`). No change to
  the chain tables.
- **Daily maintenance**: `run_due_maintenance` gains the analyzer pass. It runs
  under the process-wide DB lock, so its cost is a stated budget, not an
  afterthought — 41 ms measured today, growing with the transitions table
  rather than with the number of memes.
- **Scoring**: `app/core/candidate_scorer.py` gains two terms; the collocation
  set is read once per reply, not per candidate.
- **Hot n-grams**: `chat_hot_ngrams_repo.get_hot` gains a meme-score ordering
  behind a knob; the existing frequency path stays as the default.
- **Docs**: `docs/v2/00_STATUS.md`, `GENERATION_PIPELINE.md`, a report under
  `docs/eval_reports/`, and the manual rating sheet (kept local).
- **Riding along, no functional content**: `openspec archive
  markov2r-phase3-temporal-layer-core` as its own commit — the convention since
  Phase 2.
