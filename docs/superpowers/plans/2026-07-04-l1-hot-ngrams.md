# L1 Running Jokes / Hot N-grams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The bot occasionally opens an unprompted reply from an n-gram the chat has been unusually obsessed with in the last ~7 days ("running joke" callbacks), per L1 in `docs/DIALOGUE_GENERATION_ACTION_PLAN.md`.

**Architecture:** A new per-chat sliding-window table `chat_hot_ngrams` counts content bigrams/trigrams as messages are learned (same privacy contour as the word model: raw `chat_id`, no author, wiped by `/clear`). "Hot" = the window count is a large share of the all-time count from the existing `transitions`/`transitions1` tables (recency-share ratio). On unprompted replies only, with small probability, a hot n-gram is passed as `seed_tokens` into the already-existing (prod-unused) seed API of `MarkovGenerator`. Stale rows decay at startup on the same cadence as `chat_emoji_stats`.

**Tech Stack:** Python 3.12, aiosqlite, aiogram 3, unittest, ruff, mypy (strict), bandit. **No new dependencies.**

## Global Constraints

- Base branch: create `feat/dialogue-gen-stage4-l1` from `chore/audit-followup-fixes`. **Rebase onto `main` after PRs #55/#56 merge, before opening the PR** (user has unpushed local edits for #55 on another machine — do not touch that branch).
- Code, comments, tests, commits: English. All user-facing communication: Russian.
- Commit messages: NO `Co-Authored-By`, NO "Generated with Claude Code", no AI attribution of any kind.
- Tests use **unittest** (not pytest): `python -m unittest tests.test_<name> -v`; full suite `python -m unittest discover tests`.
- Every task ends with the quality gate: `ruff check app/ tests/ tools/ main.py` clean, `mypy app/` clean, relevant tests green — then commit.
- After each task's commit, run the **review checkpoint** (see "Per-task review checkpoints" at the end): `/code-review` skill + `bandit` on changed files; fix findings before the next task.
- SQL: parametrized only; tables keyed by raw `chat_id` (word-model contour); no author/user ids anywhere in this feature.
- Master gate: `hot_ngram_seed_chance == 0.0` disables the whole channel (no recording, no reads) — same pattern as `emoji_append_chance`.
- Eval (`tools/eval_generation.py`) keeps the channel off; baselines must not move.

---

### Task 1: Pure core module `app/core/hot_ngrams.py`

**Files:**
- Create: `app/core/hot_ngrams.py`
- Test: `tests/test_hot_ngrams.py`

**Interfaces:**
- Consumes: `app.core.lexicon.STOPWORDS` (existing `frozenset[str]`).
- Produces: `extract_content_ngrams(tokens: list[str], *, max_per_message: int = 24) -> list[tuple[str, ...]]` — deduped adjacent bigrams+trigrams that contain at least one content token; used by Task 6 (handler) and Task 4 (service typing).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hot_ngrams.py
from __future__ import annotations

import unittest

from app.core.hot_ngrams import extract_content_ngrams


class ExtractContentNgramsTest(unittest.TestCase):
    def test_extracts_bigrams_and_trigrams(self) -> None:
        ngrams = extract_content_ngrams(["крутой", "бобёр", "пришёл"])
        self.assertIn(("крутой", "бобёр"), ngrams)
        self.assertIn(("бобёр", "пришёл"), ngrams)
        self.assertIn(("крутой", "бобёр", "пришёл"), ngrams)

    def test_skips_stopword_only_ngrams(self) -> None:
        # "а", "он", "не" — no content token (stopword or shorter than 3 chars)
        self.assertEqual(extract_content_ngrams(["а", "он", "не"]), [])

    def test_keeps_ngram_with_one_content_token(self) -> None:
        ngrams = extract_content_ngrams(["не", "бобёр"])
        self.assertEqual(ngrams, [("не", "бобёр")])

    def test_skips_punctuation_tokens(self) -> None:
        # tokenize() may emit punctuation tokens; n-grams containing them are noise
        ngrams = extract_content_ngrams(["бобёр", "?", "пришёл"])
        self.assertNotIn(("бобёр", "?"), ngrams)
        self.assertNotIn(("бобёр", "?", "пришёл"), ngrams)

    def test_dedup_and_cap(self) -> None:
        tokens = ["бобёр", "пришёл"] * 40
        ngrams = extract_content_ngrams(tokens, max_per_message=5)
        self.assertEqual(len(ngrams), len(set(ngrams)))
        self.assertLessEqual(len(ngrams), 5)

    def test_short_input_returns_empty(self) -> None:
        self.assertEqual(extract_content_ngrams(["бобёр"]), [])
        self.assertEqual(extract_content_ngrams([]), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_hot_ngrams -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.hot_ngrams'`

- [ ] **Step 3: Write the implementation**

```python
# app/core/hot_ngrams.py
"""Content n-gram extraction for the L1 "running jokes" channel.

Pure functions only (no I/O): the handler feeds learned message tokens in,
gets back the bigrams/trigrams worth counting in ``chat_hot_ngrams``.
A "content" n-gram contains at least one token that is not a stopword and is
long enough to carry meaning; punctuation-only tokens disqualify an n-gram.
"""
from __future__ import annotations

from app.core.lexicon import STOPWORDS

# A token shorter than this cannot be the content anchor of an n-gram
# (filters "да", "ну", "он" — high-frequency filler the stopword list misses).
MIN_CONTENT_TOKEN_LEN = 3
# Hard cap of n-grams recorded per message: bounds the per-message write batch
# (executemany) regardless of message length.
MAX_NGRAMS_PER_MESSAGE = 24


def _is_wordlike(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def _is_content_token(token: str) -> bool:
    return (
        len(token) >= MIN_CONTENT_TOKEN_LEN
        and token not in STOPWORDS
        and _is_wordlike(token)
    )


def extract_content_ngrams(
    tokens: list[str],
    *,
    max_per_message: int = MAX_NGRAMS_PER_MESSAGE,
) -> list[tuple[str, ...]]:
    """Adjacent bigrams and trigrams that contain at least one content token.

    Preserves first-seen order, dedupes within the message, and caps the total
    so a long message cannot flood the hot-ngram table.
    """
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for size in (2, 3):
        for i in range(len(tokens) - size + 1):
            ngram = tuple(tokens[i : i + size])
            if ngram in seen:
                continue
            if not all(_is_wordlike(token) for token in ngram):
                continue
            if not any(_is_content_token(token) for token in ngram):
                continue
            seen.add(ngram)
            result.append(ngram)
            if len(result) >= max_per_message:
                return result
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_hot_ngrams -v`
Expected: PASS (6 tests OK)

- [ ] **Step 5: Quality gate + commit**

Run: `ruff check app/ tests/ tools/ main.py && mypy app/ && python -m unittest tests.test_hot_ngrams`
Expected: all clean.

```bash
git add app/core/hot_ngrams.py tests/test_hot_ngrams.py
git commit -m "feat(dialogue): content n-gram extraction for L1 hot-ngram channel"
```

---

### Task 2: Migration 012 + `ChatHotNgramsRepo`

**Files:**
- Create: `app/migrations/012_chat_hot_ngrams.sql`
- Create: `app/repositories/chat_hot_ngrams_repo.py`
- Modify: `app/repositories/__init__.py` (add export, alphabetical)
- Test: `tests/test_chat_hot_ngrams_repo.py` (mirror `tests/test_chat_emoji_stats_repo.py` fixtures)

**Interfaces:**
- Consumes: `ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]` + `asyncio.Lock` (same constructor contract as `ChatEmojiStatsRepo`); existing tables `transitions` (trigram counts) and `transitions1` (bigram counts).
- Produces (used by Task 3 facade):
  - `ChatHotNgramsRepo(conn_provider, lock)`
  - `async bump(chat_id: int, ngrams: Iterable[tuple[str, ...]]) -> None`
  - `async get_hot(chat_id: int, *, min_count: int, recency_share: float, limit: int = 8) -> list[tuple[str, ...]]`
  - `async decay_stale(cutoff_iso: str) -> int`

- [ ] **Step 1: Write the migration**

```sql
-- app/migrations/012_chat_hot_ngrams.sql
--
-- Dialogue-generation improvement L1 ([DIALOGUE_GENERATION_ACTION_PLAN] Stage 4):
-- running jokes. Counts content bigrams/trigrams per chat over a sliding window
-- (rows decay/expire at startup, see Database.decay_chat_hot_ngrams). An n-gram
-- is "hot" when its window count is a large share of its all-time count in
-- transitions/transitions1 — i.e. the chat started saying it recently. Hot
-- n-grams occasionally seed unprompted replies via the generator's seed API.
--
-- Privacy: keyed by raw chat_id to match the Markov model tables; per-chat
-- aggregate only (no author), built from the same normalized tokens as the
-- word model, wiped together with it in clear_chat. Bigrams store w3 = ''.
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them.

CREATE TABLE IF NOT EXISTS chat_hot_ngrams (
    chat_id    INTEGER NOT NULL,
    w1         TEXT NOT NULL,
    w2         TEXT NOT NULL,
    w3         TEXT NOT NULL DEFAULT '',
    cnt        INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chat_id, w1, w2, w3)
);
```

- [ ] **Step 2: Write the failing repo tests**

Copy the async-fixture scaffolding from `tests/test_chat_emoji_stats_repo.py` (temp DB file + `Database.init()`), then:

```python
# tests/test_chat_hot_ngrams_repo.py  (test bodies; scaffold as in test_chat_emoji_stats_repo.py)

async def _seed_transitions(db, chat_id):
    # Long-term model counts the hotness ratio compares against.
    conn = await db._get_conn()
    await conn.execute(
        "INSERT INTO transitions1(chat_id, w1, w2, cnt) VALUES (?, ?, ?, ?)",
        (chat_id, "крутой", "бобёр", 10),
    )
    await conn.execute(
        "INSERT INTO transitions(chat_id, w1, w2, w3, cnt) VALUES (?, ?, ?, ?, ?)",
        (chat_id, "крутой", "бобёр", "пришёл", 10),
    )
    await conn.commit()


class ChatHotNgramsRepoTest(...):  # same async harness as emoji repo tests
    async def test_bump_accumulates_and_get_hot_filters_by_share(self):
        await _seed_transitions(self.db, 1)
        repo = self.db.chat_hot_ngrams
        # window count 8 of 10 all-time -> share 0.8, hot at threshold 0.5
        for _ in range(8):
            await repo.bump(1, [("крутой", "бобёр"), ("крутой", "бобёр", "пришёл")])
        hot = await repo.get_hot(1, min_count=3, recency_share=0.5)
        self.assertIn(("крутой", "бобёр"), hot)
        self.assertIn(("крутой", "бобёр", "пришёл"), hot)

    async def test_get_hot_excludes_low_share(self):
        await _seed_transitions(self.db, 1)  # all-time 10
        await self.db.chat_hot_ngrams.bump(1, [("крутой", "бобёр")] * 3)  # share 0.3
        hot = await self.db.chat_hot_ngrams.get_hot(1, min_count=3, recency_share=0.5)
        self.assertEqual(hot, [])

    async def test_get_hot_excludes_below_min_count(self):
        await self.db.chat_hot_ngrams.bump(1, [("крутой", "бобёр")] * 2)
        hot = await self.db.chat_hot_ngrams.get_hot(1, min_count=3, recency_share=0.5)
        self.assertEqual(hot, [])

    async def test_get_hot_is_per_chat(self):
        await self.db.chat_hot_ngrams.bump(1, [("крутой", "бобёр")] * 5)
        hot_other = await self.db.chat_hot_ngrams.get_hot(2, min_count=3, recency_share=0.5)
        self.assertEqual(hot_other, [])

    async def test_bump_empty_is_noop(self):
        await self.db.chat_hot_ngrams.bump(1, [])

    async def test_decay_stale_halves_and_purges(self):
        repo = self.db.chat_hot_ngrams
        await repo.bump(1, [("крутой", "бобёр")] * 4, )
        conn = await self.db._get_conn()
        await conn.execute(
            "UPDATE chat_hot_ngrams SET updated_at = '2000-01-01 00:00:00'"
        )
        await conn.commit()
        deleted = await repo.decay_stale("2001-01-01 00:00:00")
        # 4 -> 2, row survives; second decay: 2 -> 1; third: 1 -> 0 -> purged
        self.assertEqual(deleted, 0)
```

(Also add a decay-to-zero purge assertion: set `cnt = 1`, decay, expect `deleted == 1` and table empty.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m unittest tests.test_chat_hot_ngrams_repo -v`
Expected: FAIL — `ImportError` / `AttributeError: chat_hot_ngrams` (facade not wired yet is fine — wire a minimal `Database.chat_hot_ngrams` in Task 3; for now instantiate the repo directly in tests against the migrated DB, matching how `test_chat_emoji_stats_repo.py` does it).

- [ ] **Step 4: Write the repo**

```python
# app/repositories/chat_hot_ngrams_repo.py
from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable

import aiosqlite

ConnProvider = Callable[[], Awaitable[aiosqlite.Connection]]


class ChatHotNgramsRepo:
    """Sliding-window content n-gram counts for the L1 running-jokes channel.

    Keyed by raw ``chat_id`` to match the Markov model tables; per-chat
    aggregate only (no author). Bigrams are stored with ``w3 = ''``. Hotness
    is the window count's share of the all-time count in ``transitions`` /
    ``transitions1``: a spike means the chat picked the phrase up recently.
    """

    def __init__(self, conn_provider: ConnProvider, lock: asyncio.Lock) -> None:
        self._conn_provider = conn_provider
        self._lock = lock

    async def bump(self, chat_id: int, ngrams: Iterable[tuple[str, ...]]) -> None:
        """Add one occurrence per listed n-gram (no-op if empty)."""
        counts = Counter(
            ngram for ngram in ngrams if len(ngram) in (2, 3)
        )
        if not counts:
            return
        rows = [
            (chat_id, ngram[0], ngram[1], ngram[2] if len(ngram) == 3 else "", n)
            for ngram, n in counts.items()
        ]
        async with self._lock:
            db = await self._conn_provider()
            await db.executemany(
                """
                INSERT INTO chat_hot_ngrams(chat_id, w1, w2, w3, cnt)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, w1, w2, w3) DO UPDATE SET
                    cnt = cnt + excluded.cnt,
                    updated_at = datetime('now')
                """,
                rows,
            )
            await db.commit()

    async def get_hot(
        self,
        chat_id: int,
        *,
        min_count: int,
        recency_share: float,
        limit: int = 8,
    ) -> list[tuple[str, ...]]:
        """Top n-grams whose window count is a big share of their all-time count.

        ``recency_share`` in (0..1]: 0.5 means at least half of all recorded
        occurrences happened inside the current window. Missing long-term rows
        fall back to the window count itself (share 1.0) so a brand-new meme
        still qualifies.
        """
        query = """
            SELECT h.w1, h.w2, h.w3, h.cnt
            FROM chat_hot_ngrams h
            LEFT JOIN transitions t
              ON t.chat_id = h.chat_id
             AND t.w1 = h.w1 AND t.w2 = h.w2 AND t.w3 = h.w3
            WHERE h.chat_id = ? AND h.w3 <> '' AND h.cnt >= ?
              AND h.cnt * 1.0 / MAX(COALESCE(t.cnt, h.cnt), h.cnt) >= ?
            UNION ALL
            SELECT h.w1, h.w2, h.w3, h.cnt
            FROM chat_hot_ngrams h
            LEFT JOIN transitions1 t1
              ON t1.chat_id = h.chat_id AND t1.w1 = h.w1 AND t1.w2 = h.w2
            WHERE h.chat_id = ? AND h.w3 = '' AND h.cnt >= ?
              AND h.cnt * 1.0 / MAX(COALESCE(t1.cnt, h.cnt), h.cnt) >= ?
            ORDER BY 4 DESC
            LIMIT ?
        """
        params = (
            chat_id, min_count, recency_share,
            chat_id, min_count, recency_share,
            limit,
        )
        async with self._lock:
            db = await self._conn_provider()
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
        result: list[tuple[str, ...]] = []
        for w1, w2, w3, _cnt in rows:
            if w3:
                result.append((str(w1), str(w2), str(w3)))
            else:
                result.append((str(w1), str(w2)))
        return result

    async def decay_stale(self, cutoff_iso: str) -> int:
        """Halve counts of rows not bumped since ``cutoff_iso``; purge zeros.

        Same contract as ``ChatEmojiStatsRepo.decay_stale``: halved rows get a
        fresh clock so they will not re-decay for another window; returns the
        number of purged rows.
        """
        async with self._lock:
            db = await self._conn_provider()
            await db.execute(
                """
                UPDATE chat_hot_ngrams
                SET cnt = cnt / 2, updated_at = datetime('now')
                WHERE updated_at < ?
                """,
                (cutoff_iso,),
            )
            cursor = await db.execute("DELETE FROM chat_hot_ngrams WHERE cnt <= 0")
            await db.commit()
            return max(0, cursor.rowcount)
```

Add to `app/repositories/__init__.py`: import + `__all__` entry `ChatHotNgramsRepo` (alphabetical, next to `ChatEmojiStatsRepo`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m unittest tests.test_chat_hot_ngrams_repo tests.test_migrator -v`
Expected: PASS (migration 012 picked up by the migrator test that scans `app/migrations/`).

- [ ] **Step 6: Quality gate + commit**

Run: `ruff check app/ tests/ tools/ main.py && mypy app/ && python -m unittest tests.test_chat_hot_ngrams_repo tests.test_migrator`

```bash
git add app/migrations/012_chat_hot_ngrams.sql app/repositories/chat_hot_ngrams_repo.py app/repositories/__init__.py tests/test_chat_hot_ngrams_repo.py
git commit -m "feat(dialogue): chat_hot_ngrams table and repository (L1)"
```

---

### Task 3: `Database` facade wiring + `LearningService` methods

**Files:**
- Modify: `app/infrastructure/database.py` (init/close wiring at lines 56–111, facade methods next to the emoji ones at ~390, `clear_chat` at ~475, decay constant at ~28)
- Modify: `app/services/learning_service.py` (methods next to `record_emojis`)
- Test: extend `tests/test_db_logic.py`, `tests/test_learning_service.py`

**Interfaces:**
- Consumes: `ChatHotNgramsRepo` from Task 2.
- Produces (used by Task 6 handler):
  - `Database.record_chat_hot_ngrams(chat_id: int, ngrams: Iterable[tuple[str, ...]]) -> None`
  - `Database.get_hot_chat_ngrams(chat_id: int, *, min_count: int, recency_share: float) -> list[tuple[str, ...]]`
  - `Database.decay_chat_hot_ngrams(*, decay_days: int = CHAT_HOT_NGRAM_DECAY_DAYS) -> int` (called in `init()`)
  - `LearningService.record_hot_ngrams(chat_id, ngrams)` / `LearningService.get_hot_ngrams(chat_id, *, min_count, recency_share)` — thin passthroughs like `record_emojis`/`get_emoji_stats`.
  - `clear_chat` additionally deletes from `chat_hot_ngrams`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_db_logic.py` (reuse its existing temp-DB harness):

```python
async def test_clear_chat_wipes_hot_ngrams(self):
    await self.db.record_chat_hot_ngrams(1, [("крутой", "бобёр")])
    await self.db.clear_chat(1)
    hot = await self.db.get_hot_chat_ngrams(1, min_count=1, recency_share=0.0)
    self.assertEqual(hot, [])

async def test_decay_runs_on_init(self):
    # mirrors the existing decay_chat_emoji_stats-on-init assertion:
    # plant a stale row, close, re-init, assert it was halved/purged.
    ...
```

In `tests/test_learning_service.py`: assert `record_hot_ngrams`/`get_hot_ngrams` delegate to the `Database` stub (extend the existing fake-db pattern used for `record_emojis`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_db_logic tests.test_learning_service -v`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'record_chat_hot_ngrams'`

- [ ] **Step 3: Implement wiring**

In `app/infrastructure/database.py`:

```python
CHAT_HOT_NGRAM_DECAY_DAYS = 7  # next to CHAT_EMOJI_DECAY_DAYS

# __init__: self.chat_hot_ngrams: Optional[ChatHotNgramsRepo] = None
# init(): self.chat_hot_ngrams = ChatHotNgramsRepo(self._get_conn, self._lock)
#         await self.decay_chat_hot_ngrams()   # after decay_chat_emoji_stats()
# close(): self.chat_hot_ngrams = None

async def record_chat_hot_ngrams(
    self, chat_id: int, ngrams: Iterable[tuple[str, ...]]
) -> None:
    await self._require(self.chat_hot_ngrams).bump(chat_id, ngrams)

async def get_hot_chat_ngrams(
    self, chat_id: int, *, min_count: int, recency_share: float
) -> list[tuple[str, ...]]:
    return await self._require(self.chat_hot_ngrams).get_hot(
        chat_id, min_count=min_count, recency_share=recency_share
    )

async def decay_chat_hot_ngrams(
    self, *, decay_days: int = CHAT_HOT_NGRAM_DECAY_DAYS
) -> int:
    cutoff = (datetime.now(UTC) - timedelta(days=decay_days)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )  # match the exact cutoff formatting used by decay_chat_emoji_stats
    return await self._require(self.chat_hot_ngrams).decay_stale(cutoff)
```

In `clear_chat` (inside the existing lock/transaction block, next to the other DELETEs):

```python
await db.execute("DELETE FROM chat_hot_ngrams WHERE chat_id = ?", (chat_id,))
```

In `app/services/learning_service.py` (next to `record_emojis`, add `Iterable` to imports):

```python
async def record_hot_ngrams(
    self, chat_id: int, ngrams: Iterable[tuple[str, ...]]
) -> None:
    """Fold a learned message's content n-grams into the hot-ngram window (L1)."""
    await self._db.record_chat_hot_ngrams(chat_id, ngrams)

async def get_hot_ngrams(
    self, chat_id: int, *, min_count: int, recency_share: float
) -> list[tuple[str, ...]]:
    """Currently-hot n-grams for unprompted-reply seeding (L1)."""
    return await self._db.get_hot_chat_ngrams(
        chat_id, min_count=min_count, recency_share=recency_share
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_db_logic tests.test_learning_service tests.test_chat_hot_ngrams_repo -v`
Expected: PASS

- [ ] **Step 5: Quality gate + commit**

Run: `ruff check app/ tests/ tools/ main.py && mypy app/ && python -m unittest discover tests`

```bash
git add app/infrastructure/database.py app/services/learning_service.py tests/test_db_logic.py tests/test_learning_service.py
git commit -m "feat(dialogue): wire hot-ngram repo through Database and LearningService (L1)"
```

---

### Task 4: Registry knobs + Settings + RuntimeState

**Files:**
- Modify: `app/config/registry.py` (after `markov_jump_probability`, line ~141)
- Modify: `app/config/settings.py` (dataclass field block, near `markov_jump_probability` line ~45)
- Modify: `app/config/runtime_state.py` (same, line ~33)
- Modify: `.env.example` (document the new knobs)
- Test: extend `tests/test_registry.py`, `tests/test_settings.py`, `tests/test_runtime_state.py` (wherever field lists/defaults are asserted — search for `markov_jump_probability` in each and mirror)

**Interfaces:**
- Produces (used by Task 6): `runtime_state.hot_ngram_seed_chance: float` (default 0.05, 0 disables the channel), `runtime_state.hot_ngram_min_count: int` (default 3), `runtime_state.hot_ngram_recency_share: float` (default 0.5). All runtime-mutable via `/set`.

- [ ] **Step 1: Write the failing tests**

Mirror the existing per-field assertions in `tests/test_registry.py` (find the block asserting `markov_jump_probability` defaults/parsers and add the three new fields: default values `"0.05"`, `"3"`, `"0.5"`; range validation `_float_in_range(0.0, 1.0)` rejects `1.5`, `_int_min(1)` rejects `0`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_registry tests.test_settings tests.test_runtime_state -v`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Implement**

`app/config/registry.py`, after the `markov_jump_probability` FieldSpec:

```python
    # L1 running jokes: chance to seed an *unprompted* reply from a currently
    # hot n-gram (a phrase the chat picked up in the last ~7 days). 0 disables
    # the whole channel (no recording, no reads) — same gate pattern as
    # emoji_append_chance. Mention replies are never seeded.
    FieldSpec("hot_ngram_seed_chance", "HOT_NGRAM_SEED_CHANCE", "0.05",
              _float_in_range(0.0, 1.0)),
    # Minimum window occurrences before an n-gram can be considered hot.
    FieldSpec("hot_ngram_min_count", "HOT_NGRAM_MIN_COUNT", "3", _int_min(1)),
    # Hot = window count / all-time count >= this share; 0.5 means at least
    # half of all recorded occurrences happened inside the decay window.
    FieldSpec("hot_ngram_recency_share", "HOT_NGRAM_RECENCY_SHARE", "0.5",
              _float_in_range(0.0, 1.0)),
```

`app/config/settings.py` and `app/config/runtime_state.py`: add the three fields to the dataclasses in the same relative position:

```python
    hot_ngram_seed_chance: float
    hot_ngram_min_count: int
    hot_ngram_recency_share: float
```

`.env.example`: add the three variables with defaults and one-line comments (English), next to `MARKOV_JUMP_PROBABILITY`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_registry tests.test_settings tests.test_runtime_state tests.test_runtime_config -v`
Expected: PASS (`/set`-path parsing comes free via the registry).

- [ ] **Step 5: Quality gate + commit**

Run: `ruff check app/ tests/ tools/ main.py && mypy app/ && python -m unittest discover tests`

```bash
git add app/config/registry.py app/config/settings.py app/config/runtime_state.py .env.example tests/test_registry.py tests/test_settings.py tests/test_runtime_state.py
git commit -m "feat(dialogue): hot-ngram runtime knobs (L1)"
```

---

### Task 5: Handler integration — record on learn, seed unprompted replies

**Files:**
- Modify: `app/handlers/learning.py` (two spots: the learn path in the `finally` block at ~443–448, and the `GenerationRequest` construction at ~389–397)
- Test: extend `tests/test_handlers.py`

**Interfaces:**
- Consumes: `extract_content_ngrams` (Task 1), `LearningService.record_hot_ngrams` / `get_hot_ngrams` (Task 3), the three knobs (Task 4), existing locals `address_reply`, `tokens`, `learnable`, `learning_service`, `runtime_state`.
- Produces: `GenerationRequest.seed` is a hot n-gram `list[str]` on a successful roll, else `None` (unchanged). Note: `ResponseGenerator` already drops the seed after attempt 1 (`seed = None` in the retry loop) and `_pick_seed_start` falls back gracefully when the n-gram has no start continuation — no generator changes needed.

- [ ] **Step 1: Write the failing tests**

In `tests/test_handlers.py`, following its existing fake-service/message harness:

```python
async def test_learnable_message_records_hot_ngrams(self):
    # hot_ngram_seed_chance > 0, learnable text -> record_hot_ngrams called
    # with the extract_content_ngrams() of the learned tokens.

async def test_hot_ngram_recording_disabled_at_zero_chance(self):
    # hot_ngram_seed_chance = 0.0 -> record_hot_ngrams never called.

async def test_unprompted_reply_seeded_on_roll(self):
    # unprompted path (no mention), patch random.random -> 0.0 (roll wins),
    # get_hot_ngrams returns [("крутой", "бобёр")]; assert the captured
    # GenerationRequest.seed == ["крутой", "бобёр"].

async def test_mention_reply_never_seeded(self):
    # address_reply path -> get_hot_ngrams never called, seed stays None.

async def test_no_hot_ngrams_means_no_seed(self):
    # roll wins but get_hot_ngrams returns [] -> seed is None.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest tests.test_handlers -v`
Expected: FAIL on the five new tests.

- [ ] **Step 3: Implement**

Import at top of `app/handlers/learning.py`: `from app.core.hot_ngrams import extract_content_ngrams`.

Learn path — inside the existing `finally: if learnable:` block, right after `record_message` (so the window only counts messages the model itself learned):

```python
            # L1 running jokes: count this message's content n-grams in the
            # sliding hot-ngram window. Gated on the channel knob so a zero
            # chance keeps the learn path write-free (same pattern as M3).
            if runtime_state.hot_ngram_seed_chance > 0.0:
                content_ngrams = extract_content_ngrams(tokens)
                if content_ngrams:
                    await learning_service.record_hot_ngrams(
                        message.chat.id, content_ngrams
                    )
```

Reply path — replace `seed=None` in the `GenerationRequest` with a locally computed `seed`; insert right before `response_generator.generate(...)`:

```python
        # L1 running jokes: occasionally open an unprompted reply from a
        # phrase the chat has been hot on lately. Mention replies are never
        # seeded — a direct address should answer the человек, not the meme.
        seed: list[str] | None = None
        if (
            not address_reply
            and runtime_state.hot_ngram_seed_chance > 0.0
            and random.random() < runtime_state.hot_ngram_seed_chance
        ):
            hot = await learning_service.get_hot_ngrams(
                message.chat.id,
                min_count=runtime_state.hot_ngram_min_count,
                recency_share=runtime_state.hot_ngram_recency_share,
            )
            if hot:
                seed = list(random.choice(hot))
                logger.debug(
                    "Hot-ngram seed picked: chat=%s ngram_len=%s",
                    mask_chat_id(message.chat.id),
                    len(seed),
                )
```

(Note: the debug log deliberately logs only the length, never the n-gram text — chat content stays out of logs, consistent with the project's log-masking policy.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest tests.test_handlers -v`
Expected: PASS.

- [ ] **Step 5: Quality gate + commit**

Run: `ruff check app/ tests/ tools/ main.py && mypy app/ && python -m unittest discover tests`

```bash
git add app/handlers/learning.py tests/test_handlers.py
git commit -m "feat(dialogue): seed unprompted replies from hot n-grams (L1)"
```

---

### Task 6: Planted-spike verification, eval regression, docs, audit sync

**Files:**
- Test: add the spike test to `tests/test_chat_hot_ngrams_repo.py` (end-to-end through `Database`)
- Modify: `README.md` (knob table + one line in the privacy/`/clear` notes: hot n-grams are part of the model contour and wiped by `/clear`)
- Modify: `docs/ARCHITECTURE.md` (L1 data flow paragraph next to the M3 one)
- Modify: `docs/DIALOGUE_GENERATION_ACTION_PLAN.md` (L1 → done row in the status table, same format as M1–M4)
- Modify: `docs/audits/2026-07-04-followup-review.md` (+ session update section) and `docs/audits/README.md` if statuses change

**Interfaces:** none new — verification and documentation only.

- [ ] **Step 1: Planted-spike test (the L1 "Verify" criterion from the action plan)**

```python
async def test_planted_spike_becomes_hot_and_seeds(self):
    # Feed a synthetic corpus through Database.save_message_and_update_model:
    # 30 varied filler sentences + the planted phrase "крутой бобёр пришёл"
    # repeated 6 times. Record hot ngrams for every message via
    # extract_content_ngrams (as the handler does). Then:
    hot = await self.db.get_hot_chat_ngrams(1, min_count=3, recency_share=0.5)
    self.assertIn(("крутой", "бобёр"), hot)
    # And the seed API actually starts a reply from it:
    text = await generator.generate_text(
        chat_id=1, max_chars=280, max_tokens=45,
        seed_tokens=["крутой", "бобёр"],
    )
    self.assertTrue(text and text.lower().startswith("крутой бобёр"))
```

- [ ] **Step 2: Eval regression**

Run: `python tools/eval_generation.py` (same invocation as previous stages — check the file header for exact flags).
Expected: metrics match the Stage 3 baseline (the eval path never sets `hot_ngram_seed_chance`, and `GenerationRequest.seed` stays `None` there). Record the numbers in the audit session note.

- [ ] **Step 3: Docs**

- `README.md`: three new env vars in the settings table; one sentence in the `/clear` description ("also wipes the hot-ngram window"); confirm the `/clear` confirmation text in `app/presentation/bot_messages.py` still truthfully describes what is wiped — hot n-grams fall under "model data", so no text change is expected, but verify.
- `docs/ARCHITECTURE.md`: L1 paragraph (table, decay cadence, seeding path) next to the M3/M4 notes.
- `docs/DIALOGUE_GENERATION_ACTION_PLAN.md`: L1 status row → **done** with date + summary, matching the M1–M4 row format.

- [ ] **Step 4: Full quality gate**

Run: `ruff check app/ tests/ tools/ main.py && mypy app/ && python -m unittest discover tests`
Expected: all clean, full suite green (569 + new tests).

- [ ] **Step 5: Audit sync + commit**

Add the session update to `docs/audits/2026-07-04-followup-review.md` (or a new dated file if the session is substantial): completed, changed files, checks run, limitations. Refresh `docs/audits/README.md` if statuses changed.

```bash
git add README.md docs/ARCHITECTURE.md docs/DIALOGUE_GENERATION_ACTION_PLAN.md docs/audits/ tests/test_chat_hot_ngrams_repo.py
git commit -m "docs: L1 hot-ngram channel docs, planted-spike verification, audit sync"
```

---

## Per-task review checkpoints (user requirement)

After **every** task's commit, before starting the next task:

1. **Code review:** run the `/code-review` skill (code-review plugin) at medium effort on the branch diff; apply or consciously reject findings (rejections noted in the audit session update).
2. **Security:** `bandit -r app tools main.py` — must stay at 0 medium/high (the known 15× Low B311 baseline is accepted; the new `random.choice`/`random.random` uses in Task 5 are non-crypto by design and may add Low B311 entries — acceptable, note them).
3. **Perf sanity (Tasks 2, 3, 5):** confirm the hot path budget — one `executemany` batch (≤24 rows) per *learnable* message; one indexed read only on a won seed roll (≤5% of unprompted replies); decay once at startup. All lookups ride the PKs of `chat_hot_ngrams`/`transitions`/`transitions1` — no new indexes needed; verify with `EXPLAIN QUERY PLAN` on `get_hot` in Task 2 (expect `SEARCH ... USING ... PRIMARY KEY` / covering index, no SCAN of transitions).

At the **end of the whole plan** (after Task 6):

4. **Final security pass:** run the `/security-review` skill on the full branch diff; plus `pip-audit -r requirements.lock` (with `PYTHONUTF8=1` on Windows) — no dependency changes are expected, so this is a formality.
5. **Final full-suite gate:** ruff + mypy + `python -m unittest discover tests` + eval regression already done in Task 6.
6. PR: rebase onto `main` once #55/#56 have merged, then open `feat/dialogue-gen-stage4-l1` → `main`. Live-chat observation for a few days before starting L3 (per the action plan's checkpoint rule).

## Risks / notes

- **Seed quality:** `_pick_seed_start` needs the n-gram to have a usable continuation; when it doesn't, attempt 1 may come out empty and attempts 2+ run unseeded — the reply silently degrades to a normal one. This is the designed fallback, not an error.
- **Ratio semantics:** window counts are a subset of all-time counts (both bumped on learn), so recency-share is naturally in (0..1]; after a decay halving the share drops, which is exactly the "joke is getting old" behaviour we want.
- **`clear_chat` contract:** hot n-grams are model-contour data; Task 3 wires the wipe and Task 6 verifies the user-facing `/clear` text still tells the truth.
- **Privacy:** no author ids, no raw text in logs (n-gram text never logged), same normalized-token source as the word model. No new PII surface.
