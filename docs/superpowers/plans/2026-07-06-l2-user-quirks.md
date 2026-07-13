# L2 Per-User Quirks Implementation Plan (privacy-light edition)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Frequent interlocutors very rarely get a personalized touch: when a
"regular" (someone who has addressed the bot many times) mentions the bot, the
reply is occasionally preceded by a short vocative message ("опять ты", "а,
снова ты") — per L2 in `docs/DIALOGUE_GENERATION_ACTION_PLAN.md`, redesigned
after the 2026-07-06 analysis (see Design Decisions).

**Architecture:** A new `chat_user_interactions` table stores **only**
`(chat_id, HMAC(user_id), count)` — no names, no usernames, no reversible ids.
The learning handler bumps the counter on every answered mention and, for
users above a threshold, rolls a small chance to send a vocative as a
*separate first message* through the existing `reply_humanized_sequence`
(L3 plumbing). The generated reply text itself is never modified, so all
anti-repeat bookkeeping stays untouched by construction.

**Tech Stack:** Python 3.12+, aiogram 3, unittest, ruff, mypy (strict), bandit. **No new dependencies.**

## Design Decisions (differences from the original plan sketch)

The original L2 sketch ("extend `chat_members` with an interaction counter,
prepend a vocative / first-name address") is **rejected** on three grounds:

1. **`chat_members` presence ≡ `/pivo` opt-in subscription.** `list_members`
   feeds the `/pivo` mention fanout directly, so adding rows for everyone who
   talks to the bot would make `/pivo` ping users who never subscribed — a
   functional *and* consent regression. → Counters live in their own table.
2. **First-name address requires storing/decrypting identity on the learning
   path.** For non-subscribers that is a brand-new PII category collected
   without opt-in; for subscribers the stored name goes stale (updated only on
   `/pivo_on`). → v1 stores no names and addresses nobody by name. A
   name-vocative for `/pivo` subscribers is a possible L2.1 follow-up with its
   own privacy review.
3. **Prefixing the reply text changes it**, which would let an already-sent
   phrase slip past the S1 exact-match anti-repeat window (the exact bug class
   fixed for the M3 emoji flavor). → The vocative is a separate message;
   `normalize_reply_for_repeat` needs no change at all.

Privacy contour after this plan: the only new stored datum is a counter keyed
by the same HMAC (`PIVO_HMAC_SECRET`) already used for `/pivo` — unlinkable
without the secret, wiped by `/clear`, decayed on the flavor-decay cadence.
Rotating `PIVO_HMAC_SECRET` silently resets regular status (same failure mode
as `/pivo` subscriptions; acceptable, documented).

## Global Constraints

- Base branch: create `feat/dialogue-gen-stage4-l2` from `main` (post-PR #78,
  2026-07-13). No stacking. PR targets `main`. This plan doc is the branch's
  first commit.
- Code, comments, tests, commits: English. All user-facing communication:
  Russian. `.env.example` comments: Russian (file convention).
- Commit messages: NO AI attribution; write the message to a file and use
  `git commit -F` (repo convention). Never `git add -A` — stage explicit paths.
- Quality gate per task: `.venv/Scripts/ruff.exe check app/ tests/ tools/ main.py`,
  `.venv/Scripts/mypy.exe app/`, full unittest suite green. `bandit -r app
  tools main.py` must stay 0 medium/high (new non-crypto `random` uses may add
  Low B311 — acceptable).
- Master gate: `user_quirk_chance == 0.0` disables the whole channel — no
  counter writes, no reads (pattern of `emoji_append_chance` /
  `hot_ngram_seed_chance`).
- Quirks fire only on *answered mentions with a generated reply* — never on
  fallback phrases, never on unprompted replies, never in eval (eval bypasses
  the handler). Baselines must stay byte-identical (no Markov change).
- A quirked reply skips the L3 rare-event roll (one shape break per reply);
  the rare-event daily budget is not spent.
- Per-user frequency guard: at most one quirk per (chat, user) per UTC day,
  fixed in code (not a knob) — rarity is the point.

---

### Task 1: Migration + `ChatUserInteractionsRepo`

**Files:**
- Create: `app/migrations/014_chat_user_interactions.sql` (013 was taken by
  `013_drop_transitions1.sql`, PR #70, after this plan was written)
- Create: `app/repositories/chat_user_interactions_repo.py`
- Modify: `app/repositories/__init__.py` (export)
- Test: `tests/test_chat_user_interactions_repo.py` (new; mirror
  `tests/test_chat_emoji_stats_repo.py` harness: real `Database` on a temp file)

**Migration** (runner wraps in BEGIN/COMMIT — do not add them; header comment
explains the privacy design, mirroring 011/012):

```sql
CREATE TABLE IF NOT EXISTS chat_user_interactions (
    chat_id    INTEGER NOT NULL,
    user_hash  TEXT NOT NULL,
    cnt        INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chat_id, user_hash)
);
```

Keying rationale (into the header comment): raw `chat_id` matches the model
tables and rides `clear_chat`'s one-statement wipe; `user_hash` is
HMAC-SHA256 under `PIVO_HMAC_SECRET` (same anonymization as `/pivo`, no
reversible identity, no payload).

**Repo interface** (same `ConnProvider`/lock pattern as `ChatEmojiStatsRepo`):
- `bump(chat_id: int, user_hash: str) -> None` — upsert `cnt = cnt + 1`,
  refresh `updated_at`.
- `get_count(chat_id: int, user_hash: str) -> int` — 0 when absent.
- `decay_stale(cutoff_iso: str) -> int` — halve rows with
  `updated_at < cutoff`, reset their clock, delete rows reaching 0; return the
  number deleted (identical contract to the 011/012 repos).

- [ ] Write failing repo tests: bump accumulates; count 0 for unknown user;
  per-chat isolation; decay halves stale rows / drops zeros / leaves fresh
  rows; `clear_chat` removes rows (added in Task 2 — mark expectedFailure or
  order tests accordingly).
- [ ] Implement migration + repo; export from `app/repositories/__init__.py`.
- [ ] Quality gate; commit (`feat(dialogue): chat_user_interactions table and repository (L2)`), push.

### Task 2: `Database` delegates, decay wiring, `clear_chat`

**Files:**
- Modify: `app/infrastructure/database.py`
- Test: extend `tests/test_chat_user_interactions_repo.py`; extend the lazy-decay
  tests in `tests/test_chat_emoji_stats_repo.py`

**Changes:**
- Constant `CHAT_USER_INTERACTION_DECAY_DAYS = 30` (regular status should fade
  slower than memes; 7-day halving would demote a regular after one vacation).
- Repo field + init/close wiring (pattern of `chat_emoji_stats`).
- Delegates: `record_user_interaction(chat_id, user_hash)`,
  `get_user_interaction_count(chat_id, user_hash) -> int`,
  `decay_chat_user_interactions(*, decay_days=..., now=None) -> int` (same
  cutoff-string construction as the 011/012 decays).
- `decay_flavor_stats_if_due()` additionally runs
  `decay_chat_user_interactions()` (init-time decay too, next to the other two).
- `clear_chat` gains `DELETE FROM chat_user_interactions WHERE chat_id = ?`.

- [ ] Failing tests first (decay window 30d, clear_chat coverage, lazy-decay
  includes the new table).
- [ ] Implement; quality gate; commit (`feat(dialogue): user-interaction delegates, decay and clear wiring (L2)`), push.

### Task 3: `LearningService` passthroughs + hasher injection

**Files:**
- Modify: `app/services/learning_service.py`, `main.py`
- Test: `tests/test_learning_service.py`, smoke in `tests/test_handlers.py`
  fixtures if the constructor signature change breaks them

**Changes:**
- `LearningService.__init__` gains keyword-only
  `user_hasher: Callable[[int], str] | None = None` (None keeps every existing
  test and caller working; the quirk methods raise `RuntimeError` when no
  hasher was injected — they are only reachable with the channel on, and
  `main.py` always injects one).
- Methods:
  - `record_user_interaction(chat_id: int, user_id: int) -> None` —
    `db.record_user_interaction(chat_id, self._user_hasher(user_id))`.
  - `get_user_interaction_count(chat_id: int, user_id: int) -> int`.
- `main.py`: pass `user_hasher=pivo_security.hmac_value` (PivoSecurity is
  already built two lines above `LearningService`). No new dp keys.

- [ ] Failing tests (passthrough + hashing actually applied: two different
  user ids produce independent counters; same user accumulates).
- [ ] Implement; quality gate; commit (`feat(dialogue): interaction passthroughs on LearningService (L2)`), push.

### Task 4: Knobs, vocative pool, per-user daily gate in `RuntimeState`

**Files:**
- Modify: `app/config/registry.py`, `app/config/settings.py`,
  `app/config/runtime_state.py`, `.env.example`,
  `app/presentation/fallback_phrases.py`
- Test: `tests/test_runtime_config.py` (registry roundtrip),
  `tests/test_runtime_state.py` (gate + prune),
  `tests/test_fallback_phrases.py` (pool)

**Registry FieldSpecs** (with the usual explanatory comments):
- `user_quirk_chance` / `USER_QUIRK_CHANCE`, default `"0.1"`,
  `_float_in_range(0.0, 1.0)` — chance to prepend a vocative when a regular's
  mention is answered with a generated reply; 0 disables the whole channel
  (no counting, no reads).
- `user_quirk_min_interactions` / `USER_QUIRK_MIN_INTERACTIONS`, default
  `"25"`, `_int_min(1)` — window count (30-day decayed) that makes a user a
  "regular".

**Vocative pool** in `fallback_phrases.py`: `USER_QUIRK_VOCATIVES` — 8–10
short second-person addressives in the established register, **no
placeholders, no names** ("опять ты", "а, снова ты", "ну кто ж ещё", "как
всегда, ты", "ты, конечно", …). Picked with the injected RNG via a tiny
`next_quirk_vocative(rng)` helper (no anti-repeat window needed at
once-a-day-per-user frequency).

**RuntimeState:**
- Field `last_user_quirk_day: dict[tuple[int, int], str]` (chat_id, user_id →
  ISO UTC day).
- `can_fire_user_quirk(chat_id, user_id, today_iso) -> bool` /
  `note_user_quirk(chat_id, user_id, today_iso)`.
- `forget_chat` sweeps the dict by `key[0] == chat_id` (pattern of
  `last_mention_reply_ts`).

- [ ] Failing tests first; implement; quality gate; commit
  (`feat(dialogue): user-quirk knobs, vocative pool and daily gate (L2)`), push.

### Task 5: Handler wiring in `app/handlers/learning.py`

**Files:**
- Modify: `app/handlers/learning.py`
- Test: `tests/test_handlers.py` (new TestCase next to the L3 rare-event ones,
  reusing `_fake_message` / `_reply_state`)

**Counting** — bump exactly where an answered mention is recorded, i.e. at the
three `note_mention_reply` call sites (not-enough-data fallback, generation-
failed fallback, successful generated reply; learning.py:285/438/456 as of
2026-07-13), gated by `runtime_state.user_quirk_chance > 0.0`. Anonymous
admins (GroupAnonymousBot) cannot feed counters: the handler already returns
early on `message.from_user.is_bot` (learning.py:175) — add a test asserting
this stays true rather than new code. Rationale: "interactions" = times the
bot actually answered this user's address; demoted mentions
(`address_reply=False`) and unprompted replies never count, so regular status
cannot be farmed faster than `mention_cooldown_sec` allows. Fallback answers
count toward the *counter* (the user did interact) but never receive a quirk.

**Quirk roll** — on the successful generated-reply path only, before the L3
rare-event block:

```python
quirked = False
if (
    address_reply
    and runtime_state.user_quirk_chance > 0.0
    and runtime_state.can_fire_user_quirk(message.chat.id, message.from_user.id, today_iso)
    and random.random() < runtime_state.user_quirk_chance
):
    count = await learning_service.get_user_interaction_count(
        message.chat.id, message.from_user.id
    )
    if count >= runtime_state.user_quirk_min_interactions:
        reply_parts = [next_quirk_vocative(random.Random()), reply_text]
        runtime_state.note_user_quirk(message.chat.id, message.from_user.id, today_iso)
        quirked = True
        # DEBUG log: masked chat, no user id, no vocative text needed — log the
        # event kind only (project log-masking policy).
if not quirked and runtime_state.can_fire_rare_event(...):
    ...  # existing L3 block, unchanged
```

Ordering note: the chance is rolled *before* the DB read so the common path
stays read-free; `today_iso` is the existing UTC value computed for L3 —
hoist its assignment above both blocks. Anti-repeat/`remember_recent_reply`
keeps using `reply_text` (unchanged by construction).

**Test matrix (all with chance forced to 1.0 / 0.0):**
- regular + mention → two parts sent (`msg.reply(vocative)`, `msg.answer(reply)`),
  vocative ∈ pool, daily gate stamped;
- below threshold → single plain reply, no gate stamp;
- same user same day → second quirk suppressed;
- unprompted reply → never quirked even for a regular;
- fallback (not-enough-data) → counter bumped, no quirk;
- demoted mention (`mention_cooldown_sec` active) → no bump, no quirk;
- quirked reply → rare-event roll skipped, rare-event budget untouched;
- `user_quirk_chance=0` → no `record_user_interaction` / `get_..._count` calls
  at all (assert on the AsyncMock).

- [ ] Failing tests first; implement; quality gate; commit
  (`feat(dialogue): vocative quirks for regulars on answered mentions (L2)`), push.

### Task 6: Docs, privacy disclosure, audit sync

**Files:**
- Modify: `docs/ARCHITECTURE.md` (module/table rows: `chat_user_interactions`,
  migration 014, repo list), `README.md` (Privacy section + `USER_QUIRK_*`
  knobs + `/clear` scope line), `docs/DIALOGUE_GENERATION_ACTION_PLAN.md`
  (L2 row → done + design deviation note; status block → track complete),
  `docs/DIALOGUE_GENERATION_AUDIT.md` (status note if stale claims remain).
- Privacy disclosure: README Privacy gains one bullet — the bot keeps an
  anonymous per-chat interaction counter (HMAC hash + number, same scheme as
  `/pivo`, no names), decayed after ~30 days of silence, wiped by `/clear`.
  `PIVO_PRIVACY_MESSAGE` is `/pivo`-scoped — leave it; add the disclosure to
  the `/clear` confirmation text instead if it enumerates data classes.

- [ ] Update docs; quality gate (docs don't break gates but run anyway);
  commit (`docs: L2 user-quirks docs, privacy notes and plan sync`), push.
- [ ] Open PR to `main`; differential self-review of the branch diff before
  requesting merge (the standard angles + the L2-specific checks: no
  `chat_members` writes anywhere, no name/username persisted, counters absent
  from logs).

## Definition of Done

- All six tasks merged via one PR to `main`, CI green (3.12/3.13/3.14 + docker).
- **Privacy review checklist** (explicit, in the PR description):
  - [ ] nothing new stored beyond `(chat_id, HMAC(user_id), cnt, updated_at)`;
  - [ ] `chat_members` untouched — `/pivo` fanout provably unaffected (grep +
        existing pivo tests green);
  - [ ] `/clear` wipes the new table (test);
  - [ ] no user id, hash, or vocative text in logs (grep for the new logger
        calls);
  - [ ] README privacy disclosure updated.
- `tools/generation_baseline*.json` byte-identical (no Markov change; assert
  by running the eval once).
- Live-chat observation window before calling the dialogue-gen track complete
  (perceived effect is the real metric — same rule as every prior stage).

## Out of Scope (possible L2.1, separate decision)

- First-name vocatives for `/pivo` subscribers (requires Fernet decrypt on the
  reply path, staleness handling, and its own privacy review).
- Cross-restart per-user *style* adaptation (anything beyond a counter).
- A `/quirks_privacy`-style command — revisit only if the disclosure in README
  proves insufficient in practice.
