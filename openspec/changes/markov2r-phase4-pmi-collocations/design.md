## Context

See proposal.md — Why. Design-relevant state:

- `chat_hot_ngrams` holds 7-day-window counts with a 7-day half-life decay;
  `get_hot` selects by count and by the window's share of the all-time count,
  using a correlated subquery per row (a previous whole-chat `GROUP BY` cost
  ~41 ms per reply and was replaced for exactly this reason).
- Maintenance already exists: `run_due_maintenance` fires lazily from the learn
  path once a day and reports failures through `MaintenanceAlert`.
- The whole process shares one SQLite connection behind one lock, so anything
  the maintenance pass does blocks message handling for its duration.
- Migration 008 deliberately dropped secondary indexes on the chain to protect
  write throughput. Adding one back for a daily job would be the wrong trade.
- Phase 3 gave every transition `last_seen`, which is a recency signal the
  analyzer can use without any new bookkeeping.

## Goals / Non-Goals

**Goals:**

- One association ranking that a human can look at and recognize the chat in.
- A daily pass whose cost is a measured budget, re-checkable as the corpus grows.
- Scoring effects that are provably confined to scoring.

**Non-Goals:**

- Gluing collocations into tokens (ADR-016 — see proposal).
- Deleting or trimming chain data. Bounding what the analyzer *reads* is not
  the same as removing what the chat *stored*, and this design never removes
  anything; age-based GC is M2R-220 and is separately gated.
- Raising any default. The gate decides that, and it needs human raters.

## Decisions

### D1. The support threshold is applied in SQL, and it is the volume control

Measured on the prod copy: 24 325 distinct bigrams, of which **88% occur exactly
once**. Filtering at joint ≥ 3 leaves 1 197 pairs (4.9%), and the scoring stage
collapses from 58 ms to 0.8 ms. Total pass: **41 ms** against 88 ms unfiltered.

The threshold is not an optimization bolted on: TZ §10.1 already requires
`MARKOV_MEME_MIN_JOINT_COUNT`, and a pair seen once carries no evidence. Pushing
it into the `HAVING` clause simply stops the long tail from being loaded and
scored before being thrown away.

*What this is not:* it does not delete anything. The excluded pairs stay in the
chain, keep being learned, and re-enter the analysis the moment their joint
count crosses the threshold.

*If the corpus grows enough to matter*, the remaining cost is the two marginal
aggregations over the transitions table (~40 ms of the 41 ms), and the lever is
a recency window over Phase 3's `last_seen` — which is also the `recency_factor`
already present in `meme_score`. Not an index (migration 008), not deletion.

### D2. The analyzer runs inside the existing daily maintenance

Owner decision 2026-08-12. TZ §10.1 calls it an "offline job", which is
satisfied by "not in the per-message hot path"; a job nobody schedules is a
feature that quietly does nothing.

It rides `run_due_maintenance` with the flavor decays, inherits their
once-a-day cadence, their retry-on-failure interval, and their alert path. Cost
is stated (41 ms) and recorded per pass in telemetry, so growth is visible as a
number rather than as a stall.

### D3. `markov_collocations` is a registry, not a cache

Columns per TZ §13: `chat_id, left_token, right_token, joint_count, pmi, status,
updated_at`. A pass rewrites scores and may promote candidate → active, but
statuses are explicit rather than derived, so a collocation can be retired
without the next pass silently resurrecting it.

Capacity is enforced by keeping the top `MARKOV_COLLOCATION_MAX_ENTRIES` by
score; the rest simply do not enter.

### D4. The break penalty asks the chain before it fires

The penalty applies to a candidate that has the collocation's left token
followed by something else — but only if the chain actually held a transition
from that state to the right token. Otherwise the candidate is being punished
for a continuation the corpus never contained.

This is the same lesson as Phase 2's M2R-110: a rule that looks reasonable in
isolation can be measuring the corpus rather than the candidate. The withheld
penalties are counted separately so the guard's necessity is visible in the
numbers rather than asserted here.

Availability is answered from the transition pool the walk already loaded where
possible; the scorer must not issue a query per candidate per collocation.

### D5. Hot n-grams gain an ordering, not a replacement

`get_hot` keeps its existing frequency-and-recency-share path as the default and
gains a meme-score ordering behind a knob. The roadmap says "replace frequency
selection **where it wins on eval**" — so both must be runnable side by side to
find out, and the ablation arm needs the old path intact.

### D6. The manual gate is part of the phase, not a follow-up

Doc 05 §5 fixes the protocol: top-20 by `meme_score`, rated real / merely
frequent / junk, gate at ≥70% real. This change produces the ranking, the rating
sheet template, and the aggregation — and stops. The rating itself needs people
(owner + chat participants), which is an operational dependency to line up
early, not a task the implementation can close.

The rated phrases are private chat content. The sheet stays local; the committed
report carries counts, shares and inter-rater agreement only. This follows the
project's existing snapshot-privacy rule rather than inventing a new one.

## Risks / Trade-offs

- **A daily pass under the global lock** → 41 ms today, linear in the
  transitions table. Mitigation: cost recorded per pass; the recency window of
  D1 is the documented next lever; the alert path already exists if a pass
  starts failing.
- **LLR ranks frequent short tokens first** → observed directly in the
  measurement: the top LLR pairs are one-character tokens while the high-npmi
  pairs are long content words. Mitigation: `meme_score` is built on
  *normalized* PMI with mandatory support and recency factors, and the manual
  gate exists precisely because no automatic ranking is trustworthy here.
- **The manual gate is noisy with few raters** → doc 05 §8 already says so.
  Mitigation: report inter-rater agreement, or state plainly that a single
  rater's verdict has no agreement measure.
- **Scoring terms could leak into tokenization by accident** → the spec forbids
  it and a test asserts that learning and generation tokenize identically with
  collocations active.
- **Third phase in a row that could return a null result** → possible, and the
  gate is pre-registered so a null result is reportable rather than negotiable.
  Unlike Phases 2 and 3, this one is not blocked by the single-continuation
  wall: it acts on whole candidates, of which every reply has 5–10.

## Migration Plan

1. Migration `019_markov_collocations.sql`: one new table. No chain change, so
   no rebuild risk and no forward-compatibility question.
2. Ship with bonus, penalty and meme-aware ordering all at zero/off. The
   analyzer still runs and fills the registry — the chat starts having a meme
   list before anything acts on it, which is what the manual rating needs.
3. **Rollback**: knobs back to zero via `/set`, no restart. The registry can
   stay; an unread registry costs one daily pass and nothing else. Dropping the
   table is a separate migration and is not required to disable the feature.

## Open Questions

- Whether `meme_score`'s recency factor should read Phase 3's `last_seen` or the
  existing `chat_hot_ngrams` window. Both are available; the grid can answer it,
  and the answer changes no interface, no spec and no task boundary.
