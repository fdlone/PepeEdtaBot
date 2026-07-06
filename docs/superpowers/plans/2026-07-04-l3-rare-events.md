# L3 Rare Events & False Starts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The bot very rarely breaks its usual reply shape — a one-word verdict, an ALL-CAPS reply, a double message split at a sentence boundary, or a "false start" (filler → typing pause → the real reply) — per L3 in `docs/DIALOGUE_GENERATION_ACTION_PLAN.md`.

**Architecture:** A pure event roll + transform in `app/core/reply_flavor.py` (the module that already owns cosmetic reply surface changes) maps a generated reply into a *sequence* of messages. `app/handlers/_helpers.py` gains `reply_humanized_sequence` (send-then-send with a typing pause per part; edits look botty in Telegram — never edit). The handler rolls an event only for generated replies, guarded by a per-chat daily cap kept in `RuntimeState`. Anti-repeat bookkeeping keeps using the original candidate text — events are surface-only.

**Tech Stack:** Python 3.12+, aiogram 3, unittest, ruff, mypy (strict), bandit. **No new dependencies.**

## Global Constraints

- Base branch: create `feat/dialogue-gen-stage4-l3` from `feat/dialogue-gen-stage4-l1` (both touch `app/handlers/learning.py`; stacking avoids conflicts). PR later targets the L1 branch; rebase chain collapses after #55/#56/#57 merge.
- Code, comments, tests, commits: English. All user-facing communication: Russian. `.env.example` comments: Russian (file convention).
- Commit messages: NO AI attribution of any kind. Commit **and push** after every task once its checks pass (user requirement).
- Tests: unittest (`python -m unittest tests.test_<name> -v`; full: `discover tests`). Quality gate per task: `.venv/Scripts/ruff.exe check app/ tests/ tools/ main.py`, `.venv/Scripts/mypy.exe app/`, tests green.
- Review checkpoint after every task's push: diff review across the standard angles (line-by-line, removed-behavior, cross-file, reuse, simplification, efficiency, altitude, conventions) + `bandit -r app tools main.py` must stay 0 medium/high (15× Low B311 baseline; new non-crypto `random` uses may add Low B311 — acceptable).
- Master gates: `rare_event_chance == 0.0` and `false_start_chance == 0.0` disable the respective rolls entirely (pattern of `emoji_append_chance`); daily cap `rare_event_daily_cap` bounds both.
- Events fire only on *generated* replies (never on fallback phrases, never in eval — eval bypasses the handler).
- Multi-message sends must be send-then-send (`message.reply` for the first part, `message.answer` for follow-ups); never edit-message.

---

### Task 1: Event roll + transforms in `app/core/reply_flavor.py`

**Files:**
- Modify: `app/core/reply_flavor.py` (append below `apply_reply_flavor`)
- Test: `tests/test_reply_flavor.py` (append a new TestCase)

**Interfaces:**
- Consumes: nothing new (stdlib `random.Random`).
- Produces (used by Task 4):
  - `RARE_EVENT_KINDS = ("verdict", "caps", "double")`
  - `roll_rare_event(rng: random.Random, *, event_chance: float, false_start_chance: float) -> str | None` — returns one of `"verdict" | "caps" | "double" | "false_start" | None`; the two chances are independent gates rolled in that order.
  - `apply_rare_event(kind: str, text: str, rng: random.Random) -> list[str]` — maps a reply into the message sequence to send; unknown kind or non-applicable text returns `[text]` unchanged.
  - Pools: `VERDICT_WORDS` (one-word verdicts), `FALSE_START_FILLERS` (short fillers).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_reply_flavor.py
class RareEventsTest(unittest.TestCase):
    def test_roll_disabled_at_zero_chances(self) -> None:
        rng = random.Random(1)
        for _ in range(200):
            self.assertIsNone(
                roll_rare_event(rng, event_chance=0.0, false_start_chance=0.0)
            )

    def test_roll_certain_event_returns_event_kind(self) -> None:
        rng = random.Random(1)
        kinds = {
            roll_rare_event(rng, event_chance=1.0, false_start_chance=0.0)
            for _ in range(60)
        }
        self.assertEqual(kinds, set(RARE_EVENT_KINDS))

    def test_roll_certain_false_start(self) -> None:
        rng = random.Random(1)
        self.assertEqual(
            roll_rare_event(rng, event_chance=0.0, false_start_chance=1.0),
            "false_start",
        )

    def test_verdict_is_single_word_from_pool(self) -> None:
        out = apply_rare_event("verdict", "какой-то длинный ответ", random.Random(1))
        self.assertEqual(len(out), 1)
        self.assertIn(out[0], VERDICT_WORDS)

    def test_caps_uppercases_reply(self) -> None:
        out = apply_rare_event("caps", "ну это интересно", random.Random(1))
        self.assertEqual(out, ["НУ ЭТО ИНТЕРЕСНО"])

    def test_double_splits_at_sentence_boundary(self) -> None:
        out = apply_rare_event(
            "double", "Первая мысль. Вторая мысль подлиннее", random.Random(1)
        )
        self.assertEqual(out, ["Первая мысль.", "Вторая мысль подлиннее"])

    def test_double_without_boundary_is_noop(self) -> None:
        out = apply_rare_event("double", "одно предложение без точки", random.Random(1))
        self.assertEqual(out, ["одно предложение без точки"])

    def test_false_start_prepends_filler(self) -> None:
        out = apply_rare_event("false_start", "настоящий ответ", random.Random(1))
        self.assertEqual(len(out), 2)
        self.assertIn(out[0], FALSE_START_FILLERS)
        self.assertEqual(out[1], "настоящий ответ")

    def test_unknown_kind_is_noop(self) -> None:
        out = apply_rare_event("nope", "текст", random.Random(1))
        self.assertEqual(out, ["текст"])
```

Add to the existing imports in `tests/test_reply_flavor.py`:
`from app.core.reply_flavor import (FALSE_START_FILLERS, RARE_EVENT_KINDS, VERDICT_WORDS, apply_rare_event, roll_rare_event, ...)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_reply_flavor -v`
Expected: FAIL — `ImportError: cannot import name 'roll_rare_event'`

- [ ] **Step 3: Implement**

```python
# append to app/core/reply_flavor.py

# --- L3 rare events & false starts -----------------------------------------
# Surface-only shape breaks for generated replies. The words of the reply are
# only ever uppercased or split, never rewritten; "verdict" replaces the reply
# with a one-word reaction from a fixed pool (rare enough to stay a delight).

RARE_EVENT_KINDS = ("verdict", "caps", "double")

VERDICT_WORDS = (
    "база",
    "жиза",
    "классика",
    "мощно",
    "сильно",
    "именно",
)

FALSE_START_FILLERS = (
    "ну как бы...",
    "щас",
    "эм",
    "короче",
    "погоди",
)

# A sentence boundary for the "double" split: terminal punctuation followed by
# whitespace and a word character (so "т.е." style dots don't split).
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?…])\s+(?=\S)")


def roll_rare_event(
    rng: random.Random,
    *,
    event_chance: float,
    false_start_chance: float,
) -> str | None:
    """Decide whether this reply becomes a rare event.

    The event roll (verdict/caps/double, uniform among them) is checked first,
    then the false-start roll; both gates are independent knobs and 0 disables
    the respective roll entirely.
    """
    if event_chance > 0.0 and rng.random() < event_chance:
        return rng.choice(RARE_EVENT_KINDS)
    if false_start_chance > 0.0 and rng.random() < false_start_chance:
        return "false_start"
    return None


def apply_rare_event(kind: str, text: str, rng: random.Random) -> list[str]:
    """Map a reply into the message sequence for the rolled event.

    Non-applicable input (no sentence boundary for "double", unknown kind,
    empty text) degrades to the unchanged single-message sequence.
    """
    if not text:
        return [text]
    if kind == "verdict":
        return [rng.choice(VERDICT_WORDS)]
    if kind == "caps":
        return [text.upper()]
    if kind == "double":
        parts = _SENTENCE_BOUNDARY_RE.split(text, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return [parts[0].strip(), parts[1].strip()]
        return [text]
    if kind == "false_start":
        return [rng.choice(FALSE_START_FILLERS), text]
    return [text]
```

Add `import re` to the module imports.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.test_reply_flavor -v`
Expected: PASS.

- [ ] **Step 5: Quality gate + commit + push**

Run: `.venv/Scripts/ruff.exe check app/ tests/ tools/ main.py && .venv/Scripts/mypy.exe app/ && .venv/Scripts/python.exe -m unittest tests.test_reply_flavor`

```bash
git add app/core/reply_flavor.py tests/test_reply_flavor.py
git commit -m "feat(dialogue): rare-event roll and transforms (L3)"
git push -u origin feat/dialogue-gen-stage4-l3
```

Then run the review checkpoint (Global Constraints).

---

### Task 2: `reply_humanized_sequence` in `app/handlers/_helpers.py`

**Files:**
- Modify: `app/handlers/_helpers.py` (append below `reply_humanized`)
- Test: `tests/test_handlers.py` (the chat-action tests live there; append to that TestCase's module)

**Interfaces:**
- Consumes: existing `compute_typing_delay_ms`, `TYPING_HARD_CAP_MS`.
- Produces (used by Task 4): `reply_humanized_sequence(message: Message, texts: Sequence[str], typing_min_ms: int, typing_max_ms: int, *, typing_per_char_ms: int = 0, rng: random.Random | None = None) -> None` — first part via `message.reply`, follow-ups via `message.answer`, a typing action + pause before every part; a chat-action failure never blocks sending; empty parts are skipped; an empty sequence is a no-op.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_handlers.py (uses the existing _fake_message helper)
class TestReplyHumanizedSequence(unittest.IsolatedAsyncioTestCase):
    async def test_single_part_replies_once(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message(text="x")
        with patch("app.handlers._helpers.asyncio.sleep", new=AsyncMock()):
            await reply_humanized_sequence(msg, ["один"], 0, 0)
        msg.reply.assert_awaited_once_with("один")
        msg.answer.assert_not_awaited()

    async def test_two_parts_reply_then_answer_with_two_pauses(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message(text="x")
        sleep = AsyncMock()
        with patch("app.handlers._helpers.asyncio.sleep", new=sleep):
            await reply_humanized_sequence(msg, ["раз", "два"], 0, 0)
        msg.reply.assert_awaited_once_with("раз")
        msg.answer.assert_awaited_once_with("два")
        self.assertEqual(sleep.await_count, 2)

    async def test_chat_action_failure_does_not_block_parts(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message(text="x")
        msg.bot.send_chat_action = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("app.handlers._helpers.asyncio.sleep", new=AsyncMock()):
            await reply_humanized_sequence(msg, ["раз", "два"], 0, 0)
        msg.reply.assert_awaited_once_with("раз")
        msg.answer.assert_awaited_once_with("два")

    async def test_empty_parts_are_skipped(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message(text="x")
        with patch("app.handlers._helpers.asyncio.sleep", new=AsyncMock()):
            await reply_humanized_sequence(msg, ["", "два"], 0, 0)
        msg.reply.assert_awaited_once_with("два")
        msg.answer.assert_not_awaited()
```

(If `_fake_message` does not preset `msg.answer`, it is a `MagicMock` attribute and `AsyncMock` by construction — verify while writing; adjust the helper only if `answer` is missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_handlers.TestReplyHumanizedSequence -v`
Expected: FAIL — `ImportError: cannot import name 'reply_humanized_sequence'`

- [ ] **Step 3: Implement**

```python
# append to app/handlers/_helpers.py

async def reply_humanized_sequence(
    message: Message,
    texts: Sequence[str],
    typing_min_ms: int,
    typing_max_ms: int,
    *,
    typing_per_char_ms: int = 0,
    rng: random.Random | None = None,
) -> None:
    """Send a sequence of messages with a humanized typing pause before each.

    The first non-empty part replies to the triggering message; follow-ups go
    as plain chat messages (send-then-send — edits look botty in Telegram
    clients). Chat-action/pause failures never block sending.
    """
    first = True
    for text in texts:
        if not text:
            continue
        try:
            if message.bot is not None:
                await message.bot.send_chat_action(
                    chat_id=message.chat.id, action=ChatAction.TYPING
                )
            delay_ms = compute_typing_delay_ms(
                len(text),
                typing_min_ms,
                typing_max_ms,
                typing_per_char_ms,
                rng=rng,
            )
            await asyncio.sleep(delay_ms / 1000)
        except Exception as exc:
            logger.debug("send_chat_action/typing delay failed: %s", exc)
        if first:
            await message.reply(text)
            first = False
        else:
            await message.answer(text)
```

Add `from collections.abc import Sequence` to the module imports. Refactor `reply_humanized` to delegate: `await reply_humanized_sequence(message, [text], ...)` — one code path, no drift.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.test_handlers -v`
Expected: PASS including the pre-existing chat-action tests (the delegation must not change `reply_humanized` behaviour).

- [ ] **Step 5: Quality gate + commit + push**

Run: `.venv/Scripts/ruff.exe check app/ tests/ tools/ main.py && .venv/Scripts/mypy.exe app/ && .venv/Scripts/python.exe -m unittest tests.test_handlers`

```bash
git add app/handlers/_helpers.py tests/test_handlers.py
git commit -m "feat(dialogue): reply_humanized_sequence for multi-message replies (L3)"
git push
```

Then run the review checkpoint.

---

### Task 3: Knobs + daily cap state in `RuntimeState`

**Files:**
- Modify: `app/config/registry.py` (after the `hot_ngram_recency_share` FieldSpec)
- Modify: `app/config/settings.py`, `app/config/runtime_state.py` (dataclass fields after `hot_ngram_recency_share`; cap state + methods on `RuntimeState`)
- Modify: `.env.example` (after the HOT_NGRAM block, comments in Russian)
- Modify: fixtures: `tests/test_runtime_state.py`, `tests/test_runtime_config.py`, `tests/test_bot_messages.py`, `tests/test_fallback_phrases.py` (add the three values after `hot_ngram_recency_share`), `tests/test_handlers.py` `_fake_state` defaults (`rare_event_chance = 0.0`, `false_start_chance = 0.0`, `rare_event_daily_cap = 3`, `rare_events_today = {}`)
- Test: `tests/test_runtime_state.py` (cap-method tests)

**Interfaces:**
- Produces (used by Task 4):
  - Knobs: `runtime_state.rare_event_chance: float` (default 0.005), `runtime_state.false_start_chance: float` (default 0.03), `runtime_state.rare_event_daily_cap: int` (default 3), all runtime-mutable via `/set`.
  - State: `RuntimeState.rare_events_today: dict[int, tuple[str, int]]` (chat_id → (ISO day, fired count)), pruned in `forget_chat`.
  - `RuntimeState.can_fire_rare_event(chat_id: int, today_iso: str) -> bool`
  - `RuntimeState.note_rare_event(chat_id: int, today_iso: str) -> None` (resets the counter on day change).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_runtime_state.py (its TestCase builds RuntimeState from the field dict)
def test_rare_event_cap_counts_per_day(self) -> None:
    state = self._make_state()  # helper constructing RuntimeState from the dict
    self.assertTrue(state.can_fire_rare_event(1, "2026-07-04"))
    for _ in range(state.rare_event_daily_cap):
        state.note_rare_event(1, "2026-07-04")
    self.assertFalse(state.can_fire_rare_event(1, "2026-07-04"))
    # New day resets the counter.
    self.assertTrue(state.can_fire_rare_event(1, "2026-07-05"))
    state.note_rare_event(1, "2026-07-05")
    self.assertEqual(state.rare_events_today[1], ("2026-07-05", 1))

def test_forget_chat_drops_rare_event_state(self) -> None:
    state = self._make_state()
    state.note_rare_event(1, "2026-07-04")
    state.forget_chat(1)
    self.assertNotIn(1, state.rare_events_today)
```

(Adapt the construction call to however that file builds `RuntimeState` from its field dictionary — it already exists for the earlier knobs.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_runtime_state -v`
Expected: FAIL — unknown field / missing method.

- [ ] **Step 3: Implement**

`app/config/registry.py` (after `hot_ngram_recency_share`):

```python
    # L3 rare events: chance that a generated reply becomes a "shape break" —
    # one-word verdict, ALL-CAPS, or a double message. Uniform among the three.
    # 0 disables the roll. Capped per chat per day by rare_event_daily_cap.
    FieldSpec("rare_event_chance", "RARE_EVENT_CHANCE", "0.005",
              _float_in_range(0.0, 1.0)),
    # L3 false starts: chance to send a short filler, keep "typing", then the
    # real reply as a second message. 0 disables. Shares the daily cap.
    FieldSpec("false_start_chance", "FALSE_START_CHANCE", "0.03",
              _float_in_range(0.0, 1.0)),
    # Combined per-chat daily budget for rare events + false starts.
    FieldSpec("rare_event_daily_cap", "RARE_EVENT_DAILY_CAP", "3", _int_min(0)),
```

`settings.py` / `runtime_state.py` dataclass fields (after `hot_ngram_recency_share`):

```python
    rare_event_chance: float
    false_start_chance: float
    rare_event_daily_cap: int
```

`RuntimeState` state + methods:

```python
    # L3: (ISO day, fired count) per chat for the combined daily event budget.
    rare_events_today: dict[int, tuple[str, int]] = field(default_factory=dict)

    def can_fire_rare_event(self, chat_id: int, today_iso: str) -> bool:
        day, count = self.rare_events_today.get(chat_id, (today_iso, 0))
        if day != today_iso:
            return self.rare_event_daily_cap > 0
        return count < self.rare_event_daily_cap

    def note_rare_event(self, chat_id: int, today_iso: str) -> None:
        day, count = self.rare_events_today.get(chat_id, (today_iso, 0))
        if day != today_iso:
            day, count = today_iso, 0
        self.rare_events_today[chat_id] = (day, count + 1)
```

`forget_chat`: add `self.rare_events_today.pop(chat_id, None)`.

`.env.example` (after the HOT_NGRAM block):

```
# Редкие события (L3): шанс, что сгенерированный ответ станет «сломом формы» —
# односложный вердикт, КАПС или двойное сообщение. 0 — выключено.
RARE_EVENT_CHANCE=0.005

# Фальстарты (L3): шанс отправить короткий филлер («ну как бы...», «щас»),
# подержать «печатает…» и прислать настоящий ответ вторым сообщением. 0 — выключено.
FALSE_START_CHANCE=0.03

# Общий суточный бюджет редких событий и фальстартов на чат
RARE_EVENT_DAILY_CAP=3
```

Fixtures: add the three values after `hot_ngram_recency_share` in the four fixture files (`0.005`, `0.03`, `3`); in `tests/test_handlers.py` `_fake_state` defaults set `rare_event_chance = 0.0`, `false_start_chance = 0.0`, `rare_event_daily_cap = 3`, `rare_events_today = {}` so existing handler tests stay deterministic.

- [ ] **Step 4: Run the full suite**

Run: `.venv/Scripts/python.exe -m unittest discover tests`
Expected: OK.

- [ ] **Step 5: Quality gate + commit + push**

Run: `.venv/Scripts/ruff.exe check app/ tests/ tools/ main.py && .venv/Scripts/mypy.exe app/`

```bash
git add app/config/registry.py app/config/settings.py app/config/runtime_state.py .env.example tests/test_runtime_state.py tests/test_runtime_config.py tests/test_bot_messages.py tests/test_fallback_phrases.py tests/test_handlers.py
git commit -m "feat(dialogue): rare-event knobs and daily cap state (L3)"
git push
```

Then run the review checkpoint.

---

### Task 4: Handler integration

**Files:**
- Modify: `app/handlers/learning.py` (the generated-reply send site: the `reply_humanized(message, reply_text, ...)` call around line 430, after L1's seed block; imports)
- Test: `tests/test_handlers.py`

**Interfaces:**
- Consumes: `roll_rare_event`/`apply_rare_event` (Task 1), `reply_humanized_sequence` (Task 2), knobs + cap methods (Task 3), existing locals `reply_text`, `runtime_state`, `message`.
- Produces: generated replies occasionally go out as event sequences; fallback-phrase paths and eval are untouched; anti-repeat bookkeeping (`remember_recent_reply`, `remember_short_reply`) keeps using the original `reply_text`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_handlers.py TestLearningHandler
async def test_rare_event_false_start_sends_filler_then_reply(self) -> None:
    from app.core.reply_flavor import FALSE_START_FILLERS
    from app.handlers.learning import on_text_message

    msg = _fake_message(text="pepe ответь развёрнуто")
    learning_service = AsyncMock()
    learning_service.get_token_volume = AsyncMock(return_value=100)
    learning_service.record_message = AsyncMock(return_value=100)
    learning_service.is_verbatim_copy = AsyncMock(return_value=False)
    generator = AsyncMock()
    generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
    state = self._reply_state(
        recent_replies={},
        false_start_chance=1.0,
        rare_events_today={},
    )

    with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
        await on_text_message(
            msg, learning_service, generator, state,
            "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
        )

    msg.reply.assert_awaited_once()
    self.assertIn(msg.reply.await_args.args[0], FALSE_START_FILLERS)
    msg.answer.assert_awaited_once_with("настоящий ответ бота")
    self.assertEqual(state.rare_events_today[msg.chat.id][1], 1)

async def test_rare_event_respects_daily_cap(self) -> None:
    from app.handlers.learning import on_text_message

    msg = _fake_message(text="pepe ответь развёрнуто")
    learning_service = AsyncMock()
    learning_service.get_token_volume = AsyncMock(return_value=100)
    learning_service.record_message = AsyncMock(return_value=100)
    learning_service.is_verbatim_copy = AsyncMock(return_value=False)
    generator = AsyncMock()
    generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
    from datetime import date
    state = self._reply_state(
        recent_replies={},
        false_start_chance=1.0,
        rare_events_today={msg.chat.id: (date.today().isoformat(), 3)},
        rare_event_daily_cap=3,
    )

    with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
        await on_text_message(
            msg, learning_service, generator, state,
            "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
        )

    msg.reply.assert_awaited_once_with("настоящий ответ бота")
    msg.answer.assert_not_awaited()

async def test_zero_chances_send_plain_reply(self) -> None:
    from app.handlers.learning import on_text_message

    msg = _fake_message(text="pepe ответь развёрнуто")
    learning_service = AsyncMock()
    learning_service.get_token_volume = AsyncMock(return_value=100)
    learning_service.record_message = AsyncMock(return_value=100)
    learning_service.is_verbatim_copy = AsyncMock(return_value=False)
    generator = AsyncMock()
    generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
    state = self._reply_state(recent_replies={})  # both chances 0.0 by default

    with patch("app.handlers.learning.mask_chat_id", return_value="chat"):
        await on_text_message(
            msg, learning_service, generator, state,
            "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
        )

    msg.reply.assert_awaited_once_with("настоящий ответ бота")
    msg.answer.assert_not_awaited()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_handlers -v`
Expected: the three new tests FAIL (no event path yet; `msg.answer` never called / cap ignored).

- [ ] **Step 3: Implement**

Imports in `app/handlers/learning.py`:
- extend the `_helpers` import: `from app.handlers._helpers import is_group_message, reply_humanized, reply_humanized_sequence`
- add `from app.core.reply_flavor import apply_rare_event, roll_rare_event`
- `date` comes via the existing `datetime` import (`from datetime import datetime` → add `date`).

Replace the generated-reply send block (currently `await reply_humanized(message, reply_text, ...)` followed by the short-reply/anti-repeat bookkeeping) with:

```python
        # L3 rare events: a generated reply may break shape (verdict/CAPS/
        # double message) or false-start (filler → typing → real reply).
        # Bounded by a per-chat daily budget; fallback phrases never event.
        reply_parts = [reply_text]
        today_iso = date.today().isoformat()
        if runtime_state.can_fire_rare_event(message.chat.id, today_iso):
            event_kind = roll_rare_event(
                random.Random(),
                event_chance=runtime_state.rare_event_chance,
                false_start_chance=runtime_state.false_start_chance,
            )
            if event_kind is not None:
                event_parts = apply_rare_event(
                    event_kind, reply_text, random.Random()
                )
                if event_parts != [reply_text]:
                    reply_parts = event_parts
                    runtime_state.note_rare_event(message.chat.id, today_iso)
                    logger.debug(
                        "Rare event fired: chat=%s kind=%s parts=%s",
                        mask_chat_id(message.chat.id),
                        event_kind,
                        len(reply_parts),
                    )

        await reply_humanized_sequence(
            message,
            reply_parts,
            runtime_state.typing_min_ms,
            runtime_state.typing_max_ms,
            typing_per_char_ms=runtime_state.typing_per_char_ms,
        )
```

Keep the existing bookkeeping lines (`is_short_generated_reply(...)` → `remember_short_reply`, `remember_recent_reply`) operating on the original `reply_text` — events are surface-only; the anti-repeat memory must keep matching future candidates (a verdict word or caps variant would never be generated verbatim anyway).

Note on `roll_rare_event(random.Random(), ...)`: pass `random.Random()` instances (module-level `random.random` is already patched in several tests; a fresh `Random` keeps the event roll independent of the reply-probability roll and testable via the chance knobs alone).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.test_handlers -v`
Expected: PASS — the three new tests and all pre-existing reply tests (zero chances keep the old single-reply behaviour byte-for-byte).

- [ ] **Step 5: Full gate + commit + push**

Run: `.venv/Scripts/python.exe -m unittest discover tests && .venv/Scripts/ruff.exe check app/ tests/ tools/ main.py && .venv/Scripts/mypy.exe app/`

```bash
git add app/handlers/learning.py tests/test_handlers.py
git commit -m "feat(dialogue): rare events and false starts on generated replies (L3)"
git push
```

Then run the review checkpoint.

---

### Task 5: Docs, audit sync, final security pass, PR

**Files:**
- Modify: `README.md` (config bullet next to `HOT_NGRAM_SEED_CHANCE`)
- Modify: `docs/ARCHITECTURE.md` (reply_flavor module row mentions L3; `_helpers.py` row mentions `reply_humanized_sequence`)
- Modify: `docs/DIALOGUE_GENERATION_ACTION_PLAN.md` (L3 → done row, same format as L1)
- Modify: `docs/audits/2026-07-04-followup-review.md` (session update)

**Interfaces:** none new.

- [ ] **Step 1: Docs**

- `README.md`, after the HOT_NGRAM bullet:

```markdown
- `RARE_EVENT_CHANCE` / `FALSE_START_CHANCE` — редкие «сломы формы» (L3):
  односложный вердикт, КАПС или двойное сообщение (`RARE_EVENT_CHANCE`) и
  фальстарты «филлер → печатает… → ответ» (`FALSE_START_CHANCE`); общий
  суточный бюджет на чат — `RARE_EVENT_DAILY_CAP`; 0 отключает соответствующий
  канал.
```

- `docs/ARCHITECTURE.md`: `app/core/reply_flavor.py` row → append "; редкие события и фальстарты (L3): ролл и трансформации ответа в последовательность сообщений". `handlers/` row → `_helpers.py` — `reply_humanized`, `reply_humanized_sequence`.
- `docs/DIALOGUE_GENERATION_ACTION_PLAN.md`: add the L3 **done** row after L1, listing knobs, cap, sequence helper, and that anti-repeat memory stays on the original candidate.

- [ ] **Step 2: Audit session update**

Append to `docs/audits/2026-07-04-followup-review.md`: a `## Session update — 2026-07-04 (L3 rare events & false starts)` section with Completed / Changed files / Tests run / Remaining (same shape as the L1 entry).

- [ ] **Step 3: Final gate**

Run: `.venv/Scripts/python.exe -m unittest discover tests && .venv/Scripts/ruff.exe check app/ tests/ tools/ main.py && .venv/Scripts/mypy.exe app/ && .venv/Scripts/bandit.exe -r app tools main.py` (0 medium/high) and the `/security-review` skill over the branch diff.

- [ ] **Step 4: Commit + push + PR**

```bash
git add README.md docs/ARCHITECTURE.md docs/DIALOGUE_GENERATION_ACTION_PLAN.md docs/audits/2026-07-04-followup-review.md docs/superpowers/plans/2026-07-04-l3-rare-events.md
git commit -m "docs: L3 rare events docs and audit sync"
git push
gh pr create --base feat/dialogue-gen-stage4-l1 --title "feat(dialogue): rare events & false starts (Stage 4 L3)" --body "<summary + verification + merge-order note (stacked on #57)>"
```

Confirm CI is green on the PR (`gh pr checks`).

## Risks / notes

- **Typing-pause length for false starts:** the spec asks for 1–3 s before the real reply; `compute_typing_delay_ms` with the default `TYPING_MIN/MAX` (350–1100 ms base + per-char) lands in that ballpark for typical replies — no extra knob (YAGNI).
- **Anti-repeat memory intentionally ignores the event surface** (records the original candidate): verdict/caps variants can't be re-generated verbatim, and the double-split halves are covered by the trigram overlap penalty.
- **`message.answer` availability:** aiogram's `Message.answer` sends without reply-to; used for follow-up parts so only the first part visually anchors to the trigger.
- **Events + L1 seeding can co-occur** (seeded reply turned into caps) — harmless and rare (≈0.05 × 0.005).
