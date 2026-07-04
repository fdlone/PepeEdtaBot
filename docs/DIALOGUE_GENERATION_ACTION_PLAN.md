# Dialogue Generation — Action Plan

Companion to [DIALOGUE_GENERATION_AUDIT.md](DIALOGUE_GENERATION_AUDIT.md) (2026-07-02).
This document turns the audit findings into concrete, ordered work items.

## Progress

| Item | Status | Where |
|---|---|---|
| QW1 softmax candidate selection | **done** (2026-07-02), merged via PR #50 | `SELECTION_SCORE_MARGIN=0.5` chosen after eval (margin 1.0 collapsed context overlap 0.38→0.13; 0.5 gives 0.22) |
| QW2 fallback phrase pools | **done** (2026-07-02), merged via PR #50 | 12 phrases per pool, per-chat anti-repeat window of 3 |
| QW3 length-proportional typing | **done** (2026-07-02), merged via PR #50 | `TYPING_PER_CHAR_MS=12`, hard cap 4 s |
| QW4 fuzzy casefold default | **done** (2026-07-02), merged via PR #50 | `normalize_lower=true` migration question still open |
| QW5 reply flavor post-processor | **done** (2026-07-02), merged via PR #50 | ending-punctuation transforms only |
| Baselines regenerated | **done** (2026-07-02) | own commit, diff explained in commit message |
| S1 full-reply anti-repeat | **done** (2026-07-02), merged via PR #51 | exact matches hard-rejected (window 20, trailing punctuation stripped to survive flavor), trigram overlap penalized via `recent_penalty` in `CandidateScore`; `RECENT_REPLY_PENALTY_STRENGTH=0.5` after eval sweep (1.0 collapsed case-preserved context overlap 0.21→0.09; 0.5 keeps 0.14) |
| S3 dynamic target reply length | **done** (2026-07-02), merged via PR #51 | modes short(1–4)/medium(5–14)/long(15–24), `LENGTH_MODE_WEIGHTS=0.25,0.55,0.2`; short-режим ограничивает генерацию 8 raw-токенами (кап 6 давал 11% пустых на синтетике); на синтетическом корпусе длины почти не расходятся (предложения однородны ~8 токенов) — реальный эффект смотреть в живом чате |
| S2 `/pivo` weighted anti-repeat | **done** (2026-07-03), merged via PR #52 | last `PIVO_RECENT_POOL_WINDOW`=5 индексов top/body/bottom на чат хранятся в таблице `pivo_pool_usage` (миграция 010) и исключаются при следующем выборе; выбор идёт через `_pick` (фильтрация пула → `random.choice`), поэтому seeded-вывод и патч-контракт тестов сохранены |
| S4 temporal modifiers | **done** (2026-07-03), merged via PR #52 | тематические bottom-пулы `/pivo` (ночь 00–06 / пятница / понедельник) подменяют нейтральную строку с вероятностью `PIVO_TEMPORAL_FLAVOR_CHANCE`=0.5; late-night добавляет фразы в learning-fallback; `now` инжектится для тестируемости |
| M1 chat mood state | **done** (2026-07-03), merged via PR #53 | `app/core/mood.py` (pure EWMA signals → sleepy/calm/lively/heated), stored per-chat in `RuntimeState.chat_mood` (pruned via forget_chat); `MoodModifiers` scaled by `MOOD_MODULATION_STRENGTH` modulate reply_probability, randomness_strength, length_mode_weights, reply_flavor_strength (in `ResponseGenerator`) and a heated fallback pool; transitions logged at DEBUG. All thresholds are `MOOD_*` registry knobs; `MOOD_ENABLED` master toggle. Eval unaffected (bypasses the handler) |
| M2 reply-probability AI Director | **done** (2026-07-03), merged via PR #54 | `app/core/reply_policy.py` gains pure `conversation_momentum` (rate/mention/reply-thread blend → [0,1]), `burst_factor` (post-reply ×2 for `REPLY_BURST_BOOST_SEC`, then ×0.5 for `REPLY_BURST_SUPPRESS_SEC`), `effective_reply_probability` (momentum → `[REPLY_PROBABILITY_MIN, REPLY_PROBABILITY_MAX]` band × mood × burst), `within_hourly_cap` (`REPLY_MAX_PER_HOUR` safety gate). `should_reply_to_message` takes `hourly_cap_ok`. Handler decouples rhythm tracking from mood so the director's EWMA signals exist even with mood off; `RuntimeState.note_reply_sent` records reply timestamps for cap+burst. Master toggle `REPLY_DIRECTOR_ENABLED` (off → legacy flat `reply_probability`). Mentions always answered, never capped. Carried the Stage 1–3 review fixes (mood clamp, `long`-shadow, `_pick` caveat, typo, docs) |
| M3 emoji channel | **done** (2026-07-03) | new `app/core/emoji.py` (pure `extract_emojis`/`count_emojis`/`sample_emoji`/`append_emoji_flavor`), migration `011_chat_emoji_stats` + `ChatEmojiStatsRepo` keyed by raw `chat_id` (same contour as the word model, wiped in `clear_chat`). Learning handler records this chat's emoji frequencies from raw text before the word gates (captures emoji-only reactions); `ResponseGenerator` appends a frequency-sampled emoji (`count^0.5` flattening) with chance `EMOJI_APPEND_CHANCE`, suppressed after a "?" and boosted in heated mood. Stale rows decay (halve) at init via `decay_chat_emoji_stats`. Eval keeps the channel off |
| M4 topic-drift jumps | **done** (2026-07-03) | re-enabled the mid-generation jump in `app/core/markov.py` (was hard-disabled): on jump it splices a connective (`", кстати"`, `", короче"`, …) + the new learned sentence-start triplet so the drift reads as an aside instead of dropping mid-chain. Threaded a `jump_probability` param through `generate_text`/`generate_text_with_trace`/`_generate_text_once` (default 0.0 keeps old callers exact); knob `MARKOV_JUMP_PROBABILITY` (default 0.04), only fires after 8 tokens at order 3. Baselines unchanged (eval runs with jump off) |
| L1 running jokes / hot n-grams | **done** (2026-07-04), branch `feat/dialogue-gen-stage4-l1` | new pure `app/core/hot_ngrams.py` (`extract_content_ngrams`: adjacent content bigrams/trigrams, stopword-only and punctuation n-grams excluded, casefolded stopword check for the case-preserved profile, position-major cap 24/message), migration `012_chat_hot_ngrams` + `ChatHotNgramsRepo` keyed by raw `chat_id` (bigrams store `w3=''`; wiped in `clear_chat`; decay-halved at init on the M3 cadence). Hot = window count ≥ `HOT_NGRAM_MIN_COUNT` and window/all-time share ≥ `HOT_NGRAM_RECENCY_SHARE` (SQL join to `transitions`/`transitions1`, PK-only lookups, EXPLAIN-guarded by test). Handler records n-grams next to `record_message` and, on unprompted replies only, seeds generation from a hot n-gram with chance `HOT_NGRAM_SEED_CHANCE` (0.05; 0 disables the whole channel; mentions never seeded; n-gram text never logged). Planted-spike e2e test; eval baseline byte-identical (channel off in eval) |

Status (2026-07-03): Stage 1 (PR #50), Stage 2 (PRs #51, #52) and Stage 3 M1
(PR #53) are merged into `main`; the feature branches are deleted. The suite is
green (513 tests, ruff, `mypy app/`). M1 is being observed in the live chat —
M2 starts only after that observation is signed off. Each item below lists what
to do, where, how to verify, and what can go wrong. No LLMs anywhere —
everything stays algorithmic and lightweight.

## Ground rules for all items

- **Measure before/after.** Run `tools/eval_generation.py` and compare against
  `tools/generation_baseline.json` (distinct-1/2, context overlap, empty-result
  rate, latency). Any item that changes generation output must regenerate the
  baseline deliberately, in its own commit, with the diff explained.
- **New knobs go through the registry.** Every new tunable is a `FieldSpec` in
  `app/config/registry.py` so it is env-configurable and `/set`-mutable for free.
  Do not hardcode probabilities in handlers.
- **Determinism in tests.** All new randomness must accept an injected
  `random.Random` (the codebase already follows this pattern everywhere).
- **Privacy contour.** Anything that stores per-user data must go through the
  existing anonymization/PII discipline (`privacy_filter`, anonymized authors,
  `/pivo_privacy`-style disclosure). This mainly affects items M4 and L2.
- **Characterization tests** (`tests/test_markov_generation_characterization.py`)
  will break on sampling changes — update them consciously, never blindly.

---

## Stage 1 — Quick Wins (~1 day total)

### QW1. Stochastic candidate selection (softmax instead of argmax)

**Problem:** `ResponseGenerator.generate_with_result` picks
`max(candidates, key=score.total)` (`app/core/response_generator.py:198`), which
systematically selects the "safest average" candidate and erases the variance the
sampler worked hard to produce.

**Do:**
- Replace argmax with softmax sampling over `score.total` with a temperature knob:
  `weights = [exp(s.total / T) for s in candidates]`, `rng.choices(...)`.
- Add `FieldSpec("candidate_selection_temperature", ..., default "0.7",
  float_in_range(0.0, 3.0))`; `0.0` (or `<= epsilon`) must degrade to argmax so the
  old behavior remains reachable.
- Keep hard filters (verbatim copy, echo of current message) as filters — softmax
  applies only to candidates that already passed the gates.

**Files:** `app/core/response_generator.py`, `app/config/registry.py`,
`tests/test_response_generator.py`.
**Verify:** distinct-1/2 in eval should rise; empty-result rate must not change.
**Risk:** occasionally picks a weaker candidate — that is the point; cap it by
excluding candidates with `total < best_total - margin` (margin knob, default ~1.0).

### QW2. Fallback phrase pools

**Problem:** the two hardcoded fallbacks in `app/handlers/learning.py:169` and
`:250` repeat verbatim forever and are the most visible "I am a bot" tell.

**Do:**
- Create `app/presentation/fallback_phrases.py` with two tuples of 10–15 phrases:
  `NOT_ENOUGH_DATA_PHRASES`, `GENERATION_FAILED_PHRASES` (same register/humor as
  the pivo templates).
- Pick with the injected RNG; remember the last 3 used per chat in `RuntimeState`
  (small deque, same pattern as `recent_short_replies`) and re-roll on collision.

**Files:** new `app/presentation/fallback_phrases.py`, `app/handlers/learning.py`,
`app/config/runtime_state.py`, tests.
**Risk:** none.

### QW3. Typing delay proportional to reply length

**Problem:** `reply_humanized` sleeps 350–1100 ms regardless of text length
(`app/handlers/_helpers.py:17`); a 250-char reply appearing in half a second reads
as mechanical.

**Do:** compute `delay = base + per_char_ms * len(text) + jitter`, clamp to
`[typing_min_ms, typing_max_ms_hard]` (add a hard cap knob, e.g. 4000 ms). Keep the
existing knobs as the base range.

**Files:** `app/handlers/_helpers.py`, `app/config/registry.py`, `tests/test_handlers.py`.
**Risk:** none; keep cap low enough not to feel laggy.

### QW4. Enable `fuzzy_context_casefold` by default

**Problem:** with `normalize_lower=false` (default), "Пиво" and "пиво" are distinct
model states; the casefold matcher that bridges this exists
(`app/core/context_state_matcher.py`) but ships disabled.

**Do:** flip the registry default to `"true"`. Separately evaluate (eval harness,
real chat) whether `normalize_lower=true` + `auto_capitalize_replies=true` is a
better global default — that is a data migration question (existing per-chat models
were learned case-sensitive), so keep it as a follow-up decision, not part of this
change.

**Files:** `app/config/registry.py`, `.env.example`, `tests/test_settings.py`.
**Risk:** low; slightly more DB work on context matching (index is cached per chat).

### QW5. Reply flavor post-processor (form variance)

**Problem:** every reply is a declarative sentence ending with "." —
`finalize_reply_ending` appends it unconditionally (`app/core/markov.py:284`).

**Do:**
- New module `app/core/reply_flavor.py`: pure function
  `apply_flavor(text, rng, config) -> str` applied after candidate selection in
  `ResponseGenerator`.
- Initial transforms (each behind a probability, all knobs in one registry field or
  a few): drop trailing period (p≈0.25 — chat Russian rarely ends with "."),
  `.` → `...` (p≈0.07), `.` → `!` (p≈0.05), duplicate final `?`/`!` (p≈0.04).
- Mutually exclusive: roll once for which transform applies, not independent rolls.
- Never touch texts that end with `?` semantics-changing ways; never modify anything
  but the final punctuation cluster in stage 1.

**Files:** new `app/core/reply_flavor.py`, `app/core/response_generator.py`,
`app/config/registry.py`, new test file.
**Risk:** low; keep transforms trivially reversible and covered by table-driven tests.

---

## Stage 2 — Small Improvements (~2 half-days)

### S1. Anti-repeat for full replies

**Problem:** only ≤3-token replies are deduplicated (`recent_short_replies`);
medium replies can repeat across sessions on small corpora.

**Do:**
- Add `recent_replies: dict[int, deque[str]]` (normalized full texts, maxlen ~20)
  to `RuntimeState`; record every sent generated reply.
- In `ResponseGenerator`, penalize candidates by trigram-overlap with recent
  replies (add a `recent_penalty` component to `CandidateScore` or subtract before
  softmax); hard-reject exact normalized matches.

**Files:** `app/config/runtime_state.py`, `app/core/response_generator.py`,
`app/core/candidate_scorer.py`, tests.
**Verify:** run eval twice with the same seed corpus; inter-reply repetition drops.
**Risk:** on tiny corpora this can starve candidates — penalty, not rejection,
except for exact matches.

### S2. `/pivo` weighted anti-repeat

**Problem:** `PivoMessageGenerator.build` picks top/body/bottom uniformly
(`app/services/pivo_message_builder.py:113`); tops become recognizable quickly.

**Do:**
- Persist the last N (5) used indices per pool per chat — either in the existing
  `pivo_daily_usage` table (new columns/rows) or a small new table.
- Exclude (or heavily down-weight) recently used entries when choosing.

**Files:** `app/services/pivo_message_builder.py`, `app/services/pivo_service.py`,
`app/repositories/pivo_usage_repo.py`, migration, tests.
**Risk:** low; fall back to uniform choice if the store is empty/unavailable.

### S3. Dynamic target reply length

**Problem:** `natural_length` pins everything to 5–14 content tokens
(`app/core/candidate_scorer.py:115`) — length is the most audible monotony.

**Do:**
- Before generating candidates, sample a length mode per reply:
  short (≤4) / medium (5–14) / long (15–24) with weights ~0.25/0.55/0.20.
- Parameterize the scorer: `natural_length(tokens, mode)` peaks at the sampled
  mode's range; pass the mode through `GenerationRequest`.
- Optionally scale `max_tokens` passed to the Markov generator by mode.

**Files:** `app/core/candidate_scorer.py`, `app/core/response_generator.py`, tests.
**Risk:** medium — touches scorer tests; keep the default mode distribution as a
registry knob string (e.g. `"0.25,0.55,0.2"`) or three float fields.

### S4. Temporal modifiers

**Do:** add small time-aware phrase pools: late-night fallbacks, Monday-morning
`/pivo` bottoms, Friday variants. Selection = filter pools by
`datetime.now()` bucket before random choice; always keep a neutral pool as
fallback so no bucket is ever empty.

**Files:** `app/domain/pivo_templates.py`, `app/services/pivo_message_builder.py`,
`app/presentation/fallback_phrases.py`, tests (inject clock).
**Risk:** low. Inject `now` as a parameter for testability.

---

## Stage 3 — Medium Features (1–2 days each)

### M1. Chat mood state

**Goal:** a hidden per-chat variable that makes the bot's *behavior* drift.

**Do:**
- New `app/core/mood.py`: `ChatMood` computed from cheap signals already flowing
  through `on_text_message`: messages/minute (EWMA), share of `!`/`?`/caps in
  recent messages, mention frequency. States: `calm`, `lively`, `heated`, `sleepy`.
- Store per-chat mood + EWMA counters in `RuntimeState` (prune with the existing
  `prune_inactive` mechanism).
- Mood modulates (multipliers, all registry-tunable): `randomness_strength`
  (heated → +0.5), length-mode weights (sleepy → shorter), `reply_probability`
  (lively → ×1.5), flavor probabilities (heated → more `!`), fallback pool choice.
- Log mood transitions at DEBUG with masked chat id.

**Files:** new `app/core/mood.py`, `app/config/runtime_state.py`,
`app/handlers/learning.py`, `app/core/response_generator.py`, registry, tests.
**Verify:** unit tests drive synthetic message timelines through the EWMA and
assert state transitions; eval harness unaffected (mood off by default in eval).

### M2. Reply-probability AI Director

**Goal:** replace the flat `p=0.08` with conversation-momentum dynamics.

**Do:**
- Extend `should_reply_to_message` inputs with a `momentum` score derived from M1's
  counters (or ship standalone if M1 is deferred): recent message rate, whether the
  bot was mentioned in the last K messages, thread depth of reply chains.
- Map momentum → probability in `[reply_probability_min, reply_probability_max]`
  (new knobs, defaults 0.02/0.30, current `reply_probability` becomes the midpoint).
- Add "burst" mechanic: after the bot replies, for the next ~3 minutes probability
  is boosted ×2, then suppressed ×0.5 for the following ~10 minutes (cooldown knob
  stays as the hard floor). This creates the join-then-withdraw rhythm humans have.
- **Safety:** keep `min_cooldown_sec` as an absolute rate limiter; add a per-hour
  reply cap knob (default e.g. 20) so a runaway chat can't make the bot spam.

**Files:** `app/core/reply_policy.py`, `app/config/runtime_state.py`,
`app/handlers/learning.py`, registry, `tests/test_bot_policy.py`.
**Risk:** medium — behavioral, hard to unit-test perception; test the math, then
tune live via `/set`.

**Review fixes to fold into the M2 PR** (from the 2026-07-03 review of Stages
1–3 M1; none are blockers, all ride along with M2):

1. **`MOOD_MODULATION_STRENGTH` > 2.0 produces negative modifiers.** The
   registry allows up to 3.0, but the linear scaling in
   `modifiers_for_mood` (`1.0 + strength * (mult - 1.0)`) goes negative for
   any base multiplier < 1.0 once `strength > 2.0`: sleepy at 3.0 yields
   `length_weight_mult=(2.8, 1.0, -0.5)` and `reply_probability_mult=-0.5`.
   A negative weight reaches `random.Random.choices` in
   `sample_length_mode` (non-monotonic cumulative weights → undefined pick),
   and the negative probability silently disables unprompted replies.
   `randomness_delta` is already guarded by `max(0.0, ...)` in
   `ResponseGenerator`; the multipliers are not.
   **Fix:** clamp inside `modifiers_for_mood` — wrap each scaled multiplier
   in `max(0.0, ...)` so any strength degrades gracefully (weights hitting 0
   just exclude a mode; `sample_length_mode` is safe because registry
   validation guarantees at least one positive base weight). Keep the
   registry range at 3.0. Add a unit test in `tests/test_mood.py` asserting
   all modifier components are ≥ 0 for every mood at strength 3.0.
2. **`_parse_length_mode_weights` shadows the `long` builtin**
   (`app/config/registry.py`). Rename the unpacked variables to
   `short_w, medium_w, long_w`. Cosmetic, no behavior change.
3. **`_pick` recovers the index via `pool.index(chosen)`**
   (`app/services/pivo_message_builder.py`): with duplicate strings in a
   pool it would record the first occurrence's index, slightly skewing
   anti-repeat. Pools are currently duplicate-free, so either leave a
   comment stating the uniqueness assumption or switch to filtering
   `(index, part)` pairs and choosing among them. Low priority.
4. *(already fixed in the working tree, rides with this PR)* Typo in
   `LATE_NIGHT_FALLBACK_PHRASES`: "генератор works" → "генератор работает".
5. *(docs, already done in the working tree, ride with this PR)* Plan/audit
   status refresh (merged-PR notes, stale "not merged yet" paragraph removed)
   and `ARCHITECTURE.md` sync (module lists, migrations 001…010,
   `pivo_pool_usage` table, archive links).

### M3. Emoji channel

**Goal:** the bot uses *this chat's* emojis without polluting the Markov model.

**Do:**
- During learning, extract emojis from the raw text (before `TOKEN_RE` drops them)
  with a proper emoji regex; store per-chat frequency in a new table
  `chat_emoji_stats(chat_id, emoji, count)` (migration 010).
- In `reply_flavor`, with p≈0.15 append 1 (rarely 2) emoji sampled by frequency
  with the same `count^power` flattening used elsewhere; suppress when the reply
  already ends with `?` + question intonation feels off, and in `heated` mood
  optionally boost.
- Respect retention: decay counts (e.g. halve weekly via a maintenance query) so
  dead memes fade.

**Files:** migration, `app/infrastructure/database.py`,
`app/services/learning_service.py`, `app/core/reply_flavor.py`, repo layer, tests.
**Risk:** medium; keep emoji storage within the privacy contour (emojis are not PII,
but the table is per-chat — cover it in `/clear`).

### M4. Re-enable mid-generation jumps (topic drift)

**Problem:** the jump mechanism exists but is disabled
(`jump_probability = 0.0`, `app/core/markov.py:1511`) because a jump replaces the
current state without splicing connective tissue into the output.

**Do:**
- On jump, insert a connective token sequence before continuing: sample from a tiny
  pool (`"кстати"`, `"короче"`, `"а вообще"`, ...) plus a comma, then continue from
  the new start state.
- Set `jump_probability` from a registry knob (default ~0.04), only for replies
  already ≥8 tokens (the guard exists).
- Update `_run_generation_loop` so jump output passes the finalization gates
  (context-heavy checks must not count the connective).

**Files:** `app/core/markov.py`, registry, characterization tests.
**Risk:** medium — this is the deepest core change in the plan; land it behind the
knob defaulting to a small value, with eval before/after.

---

## Stage 4 — Large Features (up to a week each)

### L1. Running jokes / callbacks

**Goal:** the bot occasionally references what the chat was obsessed with recently.

**Do:**
- Track "hot n-grams": per chat, count content bigrams/trigrams over a sliding
  window (e.g. 7 days) in a new table `chat_hot_ngrams`; "hot" = frequency spike
  vs the long-term Markov transition counts (ratio threshold). Exclude stopwords-only
  n-grams (reuse `app/core/lexicon.py`).
- With small probability (p≈0.05, knob) when generating an unprompted reply, use a
  hot n-gram as `seed_tokens` (the seed API already exists and is unused in prod —
  `_pick_seed_start` at `app/core/markov.py:1074`).
- Maintenance: decay/purge on the same cadence as retention cleanup.

**Files:** migration, database, `app/services/learning_service.py`,
`app/core/response_generator.py`, `app/handlers/learning.py`, registry, tests.
**Verify:** synthetic corpus with a planted spike n-gram → assert seeded replies
reference it at roughly the configured rate.

### L2. Per-user quirks (privacy-sensitive)

**Goal:** frequent interlocutors get rare personalized touches.

**Do:**
- Extend `chat_members` with an interaction counter (bot was mentioned by / replied
  to this user). **Store only anonymized ids consistent with migration 003 and the
  existing privacy model; document in the privacy notes; wipe via `/clear`.**
- On mention-triggered replies from a "regular" (counter above threshold), with
  p≈0.1 prepend a vocative from a small pool ("опять ты", first-name address using
  the already-stored member profile, etc.).
- Keep it rare: per-user cooldown on personalization (once/day) so it stays a
  delight, not a gimmick.

**Files:** `app/repositories/chat_members_repo.py`, migration,
`app/handlers/learning.py`, `app/presentation/` pool, registry, tests, privacy docs.
**Risk:** highest in the plan — privacy review is part of the definition of done.

### L3. Rare events & false starts

**Do:**
- `reply_flavor` gains an "event" roll (p≈0.005, knob): one-word verdict reply,
  ALL-CAPS reply, or double-message reply (split candidate at a sentence boundary,
  send two messages with a typing pause between).
- False starts (p≈0.03): send a filler ("ну как бы...", "щас"), wait 1–3 s typing,
  then send the real reply as a second message. (Prefer send-then-send over
  edit-message: simpler, and edits look botty in Telegram clients.)
- Both need handler-level support for multi-message replies —
  extend `reply_humanized` into `reply_humanized_sequence`.

**Files:** `app/core/reply_flavor.py`, `app/handlers/_helpers.py`,
`app/handlers/learning.py`, registry, tests.
**Risk:** medium; cap events per chat per day (counter in `RuntimeState`).

---

## Suggested sequencing & checkpoints

Actual so far: Stage 1 landed as one PR (#50, all QW + baseline), S1 + S3 as
PR #51, S2 + S4 as PR #52, M1 as PR #53, M2 + the Stage 1–3 review fixes as
PR #54, and M3 + M4 together as the next PR. Remaining sequencing:

1. **L1 → L3 → L2** (L2 last: needs the privacy review and benefits from the
   flavor/sequence plumbing built earlier).

After each stage: run the full test suite + `tools/eval_generation.py`, and let the
bot run in the live chat for a few days before starting the next stage — perceived
liveliness is the real metric, and it can only be judged in situ.
