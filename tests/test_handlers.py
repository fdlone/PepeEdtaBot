"""Happy-path tests for all routers.

Each handler is called directly with faked Message / service objects.
No aiogram dispatcher is started; we only verify that handlers call
the right service methods and reply to the message.
"""
from __future__ import annotations

import random
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

from app.services.pivo_service import PivoCallMessage

# Один генератор-мок на весь сьют: копия этого хелпера уже расходилась с
# оригиналом (у неё не было телеметрии), а рукописные зеркала проект уже
# однажды оплатил падениями фикстур (O6).
from tests.test_response_generator import _traced_generator


def _real_runtime_state() -> object:
    """A genuine RuntimeState — overlay tests need real behaviour, not a mock."""
    from tests.test_runtime_state import make_runtime_state

    return make_runtime_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pivo_stub() -> AsyncMock:
    """PivoService mock: on_text_message refreshes the sender's /pivo profile."""
    service = AsyncMock()
    service.refresh_member = AsyncMock()
    return service


def _fake_message(
    *,
    text: str = "",
    chat_type: str = "supergroup",
    user_id: int = 1,
    chat_id: int = 100,
    is_bot: bool = False,
    reply_to: MagicMock | None = None,
) -> MagicMock:
    msg = MagicMock()
    msg.text = text
    msg.chat.id = chat_id
    msg.chat.type = chat_type
    msg.from_user.id = user_id
    msg.from_user.is_bot = is_bot
    msg.reply_to_message = reply_to
    msg.reply = AsyncMock()
    msg.answer = AsyncMock()
    msg.bot = AsyncMock()
    msg.bot.send_chat_action = AsyncMock()
    return msg


def _fake_state(**kwargs: object) -> MagicMock:
    s = MagicMock()
    s.typing_min_ms = 0
    s.typing_max_ms = 0
    s.typing_per_char_ms = 0
    # Реальная зона, не Mock: datetime.now(tz) требует настоящий tzinfo.
    s.chat_timezone = ZoneInfo("UTC")
    s.recent_fallbacks = {}
    s.recent_replies = {}
    # Hourly-cap guard neutral by default: an empty history never trips the cap.
    s.reply_max_per_hour = 20
    s.recent_reply_times = {}
    s.recent_reply_penalty_strength = 1.0
    s.verbatim_penalty_strength = 0.0
    s.length_mode_weights = (0.25, 0.55, 0.2)
    s.intonation_profile_strength = 0.0
    s.length_context_adaptation = 0.0
    # Deterministic selection and untouched reply text so handler tests can
    # assert generated candidates verbatim.
    s.candidate_selection_temperature = 0.0
    s.reply_flavor_strength = 0.0
    # Emoji channel off by default so generation tests assert reply text verbatim
    # and don't hit the learning service's emoji-stats read.
    s.emoji_append_chance = 0.0
    s.pivo_recent_pool_window = 5
    s.pivo_temporal_flavor_chance = 0.5
    # M4 jump off by default so generation tests stay deterministic; the markov
    # characterization tests drive jump_probability directly.
    s.markov_jump_probability = 0.0
    s.context_jump_boost = 1.0
    s.verbatim_extension_share = 0.0
    s.order_mix_probability = 0.0
    s.slot_mutation_probability = 0.0
    # Phase 2 knobs neutral by default: gain 0 is the 1.x sampler and a
    # degenerate bound of 0 leaves the candidate target fixed, so handler tests
    # keep asserting the pre-Phase-2 candidate flow.
    s.markov_entropy_temp_gain = 0.0
    s.markov_entropy_pivot = 0.5
    s.markov_entropy_temp_min = 0.5
    s.markov_entropy_temp_max = 12.0
    s.markov_branching_degenerate_max = 0.0
    s.markov_branching_candidate_floor = 2
    # Phase 4 collocation weights neutral: non-zero (or a bare MagicMock, which
    # does not support ordering comparisons) would send the pipeline to the
    # collocation registry on a mock.
    s.markov_collocation_bonus = 0.0
    s.markov_collocation_break_penalty = 0.0
    s.markov_hot_ngram_meme_ordering = False
    s.markov_seeded_candidate_ratio = 0.0
    s.markov_seed_branch_min = 2.0
    s.markov_seed_branch_ideal = 6.0
    s.markov_seed_branch_max = 50.0
    s.markov_seed_min_support = 5.0
    s.markov_seed_min_score = 0.1
    s.markov_seed_min_token_len = 3
    s.markov_seed_head_share = 0.4
    # L1 hot-ngram channel off by default so learn/reply tests stay
    # deterministic; dedicated hot-ngram tests enable it explicitly.
    s.hot_ngram_seed_chance = 0.0
    s.hot_ngram_min_count = 3
    s.hot_ngram_recency_share = 0.5
    # L3 rare events off by default so reply tests assert a single message;
    # dedicated rare-event tests enable the chances explicitly.
    s.rare_event_chance = 0.0
    s.false_start_chance = 0.0
    s.rare_event_daily_cap = 3
    s.user_quirk_chance = 0.0
    s.user_quirk_min_interactions = 25
    s.user_quirk_name_share = 0.0
    s.rare_events_today = {}
    # Bind the real cap methods so handler tests exercise actual budget logic.
    from app.config.runtime_state import RuntimeState

    s.can_fire_rare_event = (
        lambda chat_id, today: RuntimeState.can_fire_rare_event(s, chat_id, today)
    )
    s.note_rare_event = (
        lambda chat_id, today: RuntimeState.note_rare_event(s, chat_id, today)
    )
    # L2 quirk gate: real once-a-day logic for the same reason as the caps.
    s.last_user_quirk_day = {}
    s.can_fire_user_quirk = (
        lambda chat_id, user_id, today: RuntimeState.can_fire_user_quirk(
            s, chat_id, user_id, today
        )
    )
    s.note_user_quirk = (
        lambda chat_id, user_id, today: RuntimeState.note_user_quirk(
            s, chat_id, user_id, today
        )
    )
    # Mood off by default so the existing behavioural assertions see the
    # unmodulated path; dedicated mood tests enable it explicitly.
    s.mood_enabled = False
    # Числовое значение обязательно: счётчик темпа (O14) сравнивает с ним
    # rate_ewma, а голый MagicMock не поддерживает сравнение и даёт TypeError,
    # из которого не видно ни причины, ни того, что виноват реестр.
    #
    # Это уже вторая ручка, добавленная в реестр и уронившая тесты здесь по
    # той же причине (первая — на M2R-900, 71 падение). Фикстура остаётся
    # рукописным зеркалом RUNTIME_FIELDS поверх MagicMock — единственным из
    # оставшихся: три других зеркала сняты по O6 и собираются из реестра.
    # Пока это не сделано и здесь, каждая новая ручка будет ломать этот файл
    # заново. Заведено ревью 2026-08-26 как A-9/D-3.
    s.mood_lively_rate_per_min = 12.0
    # M2 director off by default so the flat reply_probability path is exercised
    # by the existing tests; dedicated director tests enable it explicitly.
    s.reply_director_enabled = False
    # Off by default so generated reply text is asserted verbatim; a bare
    # MagicMock attribute would be truthy and trigger reply capitalization.
    s.auto_capitalize_replies = False
    # Mention anti-flood gate off by default so existing mention-driven tests
    # keep their guaranteed-reply behaviour; dedicated tests enable it.
    s.mention_cooldown_sec = 0
    s.last_mention_reply_ts = {}
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------

class TestCommonHandlers(unittest.IsolatedAsyncioTestCase):
    async def test_ping_replies_pong(self) -> None:
        from app.handlers.common import cmd_ping
        msg = _fake_message()
        await cmd_ping(msg)
        msg.reply.assert_awaited_once_with("pong")

    async def test_help_replies(self) -> None:
        from app.handlers.common import cmd_help
        msg = _fake_message()
        state = _fake_state()
        await cmd_help(msg, state)
        msg.reply.assert_awaited_once()

    async def test_stats_replies_with_model_volume(self) -> None:
        from app.core.markov import MarkovGenerator
        from app.handlers.common import cmd_stats
        msg = _fake_message()
        db = AsyncMock()
        db.get_chat_token_volume = AsyncMock(return_value=250)
        db.collocations.count_by_status = AsyncMock(return_value={})
        state = _fake_state(min_tokens_for_model=50)
        generator = MarkovGenerator(db=AsyncMock())

        await cmd_stats(msg, db, state, generator)

        msg.reply.assert_awaited_once()
        self.assertIn("250", msg.reply.await_args.args[0])
        # Полный срез по чату — пять COUNT-ов, из которых команда не печатает
        # ни одного; читается только то, что показывается.
        db.get_stats.assert_not_called()


# ---------------------------------------------------------------------------
# admin.py
# ---------------------------------------------------------------------------

class TestAdminHandlers(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        from app.filters import admin_or_owner

        admin_or_owner._admin_cache.clear()

    @staticmethod
    def _config_deps(*, admin_ids: tuple[int, ...] = ()) -> tuple[AsyncMock, MagicMock]:
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(
            return_value=[MagicMock(user=MagicMock(id=uid)) for uid in admin_ids]
        )
        return bot, MagicMock(owner_id=None)

    async def test_config_replies(self) -> None:
        from app.handlers.admin import cmd_config
        msg = _fake_message(text="/config")
        state = _fake_state()
        base = _real_runtime_state()
        bot, settings = self._config_deps()
        with patch("app.handlers.admin.format_config_message", return_value="cfg"):
            await cmd_config(msg, state, base, bot, settings)
        msg.reply.assert_awaited_once()
        # Краткая форма открыта всем: права даже не запрашиваются.
        bot.get_chat_administrators.assert_not_called()

    async def test_config_marks_chat_overrides(self) -> None:
        from app.handlers.admin import cmd_config

        msg = _fake_message(text="/config")
        base = _real_runtime_state()
        base.set_override(msg.chat.id, "reply_probability", 0.5)
        bot, settings = self._config_deps()
        await cmd_config(msg, base.effective(msg.chat.id), base, bot, settings)

        text = msg.reply.await_args.args[0]
        self.assertIn("Переопределено для этого чата", text)
        self.assertIn("reply_probability", text)

    async def test_config_full_is_denied_to_plain_member(self) -> None:
        """Полная форма — карта модерации, а не общая справка.

        По ней видно, какие пороги надо обойти, чтобы бот отвечал чаще, и что
        именно переопределено в этом чате. Читать её логично тому же кругу,
        который может её менять через /set.
        """
        from app.handlers.admin import cmd_config

        msg = _fake_message(text="/config full")
        base = _real_runtime_state()
        bot, settings = self._config_deps()

        await cmd_config(msg, base.effective(msg.chat.id), base, bot, settings)

        text = msg.reply.await_args.args[0]
        self.assertIn("админам этого чата", text)
        # Отказ, а не тихая подмена краткой формой.
        self.assertNotIn("max_reply_chars", text)

    async def test_config_full_is_allowed_to_chat_admin(self) -> None:
        from app.handlers.admin import cmd_config

        msg = _fake_message(text="/config full")
        base = _real_runtime_state()
        bot, settings = self._config_deps(admin_ids=(msg.from_user.id,))

        await cmd_config(msg, base.effective(msg.chat.id), base, bot, settings)

        text = msg.reply.await_args.args[0]
        self.assertIn("max_reply_chars", text)

    async def test_config_full_is_allowed_to_owner(self) -> None:
        from app.handlers.admin import cmd_config

        msg = _fake_message(text="/config full")
        base = _real_runtime_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=msg.from_user.id)

        await cmd_config(msg, base.effective(msg.chat.id), base, bot, settings)

        text = msg.reply.await_args.args[0]
        self.assertIn("max_reply_chars", text)

    async def test_config_short_form_is_allowed_to_plain_member(self) -> None:
        from app.handlers.admin import cmd_config

        msg = _fake_message(text="/config")
        base = _real_runtime_state()
        base.set_override(msg.chat.id, "reply_probability", 0.5)
        bot, settings = self._config_deps()

        await cmd_config(msg, base.effective(msg.chat.id), base, bot, settings)

        text = msg.reply.await_args.args[0]
        self.assertIn("шанс ответа", text)
        self.assertNotIn("max_reply_chars", text)
        # Пометка о переопределениях остаётся и в краткой форме.
        self.assertIn("Переопределено для этого чата", text)

    async def test_set_no_args_shows_usage(self) -> None:
        from app.handlers.admin import cmd_set
        msg = _fake_message(text="/set")
        state = _fake_state()
        settings = MagicMock()
        await cmd_set(msg, state, settings, _real_runtime_state(), AsyncMock())
        msg.reply.assert_awaited_once()
        assert "Использование" in msg.reply.call_args[0][0]

    async def test_set_writes_override_for_this_chat_only(self) -> None:
        from app.handlers.admin import cmd_set

        msg = _fake_message(text="/set reply_probability 0.5", chat_id=100)
        base = _real_runtime_state()
        base.reply_probability = 0.1

        await cmd_set(msg, base.effective(100), MagicMock(), base, AsyncMock())

        # Written into the overlay, not the global value.
        self.assertEqual(base.effective(100).reply_probability, 0.5)
        self.assertEqual(base.reply_probability, 0.1)
        self.assertEqual(base.effective(200).reply_probability, 0.1)
        self.assertIn("в этом чате", msg.reply.await_args.args[0])

    async def test_set_global_form_changes_the_global_value(self) -> None:
        from app.handlers.admin import cmd_set

        msg = _fake_message(
            text="/set global reply_probability 0.5", user_id=42, chat_id=100
        )
        base = _real_runtime_state()
        base.reply_probability = 0.1

        # The global form is the owner's; a chat admin gets refused (covered
        # separately below).
        await cmd_set(msg, base.effective(100), MagicMock(owner_id=42), base, AsyncMock())

        self.assertEqual(base.reply_probability, 0.5)
        self.assertEqual(base.chat_overrides, {})
        self.assertIn("глобально", msg.reply.await_args.args[0])

    async def test_set_reset_returns_chat_to_the_global_value(self) -> None:
        from app.handlers.admin import cmd_set

        msg = _fake_message(text="/set reset reply_probability", chat_id=100)
        base = _real_runtime_state()
        base.reply_probability = 0.1
        base.set_override(100, "reply_probability", 0.5)

        await cmd_set(msg, base.effective(100), MagicMock(), base, AsyncMock())

        self.assertEqual(base.effective(100).reply_probability, 0.1)

    async def test_set_survives_to_the_next_update(self) -> None:
        # The trap this design has: writing into the per-chat *view* would
        # report success and vanish. Read it back the way a later update does.
        from app.handlers.admin import cmd_set

        msg = _fake_message(text="/set reply_probability 0.5", chat_id=100)
        base = _real_runtime_state()

        await cmd_set(msg, base.effective(100), MagicMock(), base, AsyncMock())

        self.assertEqual(base.effective(100).reply_probability, 0.5)

    async def test_chat_admin_can_tune_their_own_chat(self) -> None:
        from app.handlers.admin import cmd_set

        msg = _fake_message(text="/set reply_probability 0.5", user_id=7, chat_id=100)
        base = _real_runtime_state()
        base.reply_probability = 0.1
        settings = MagicMock(owner_id=42)  # caller is an admin, not the owner

        await cmd_set(msg, base.effective(100), settings, base, AsyncMock())

        self.assertEqual(base.effective(100).reply_probability, 0.5)
        self.assertEqual(base.reply_probability, 0.1)

    async def test_chat_admin_cannot_change_the_global_value(self) -> None:
        from app.handlers.admin import cmd_set

        msg = _fake_message(
            text="/set global reply_probability 0.5", user_id=7, chat_id=100
        )
        base = _real_runtime_state()
        base.reply_probability = 0.1
        settings = MagicMock(owner_id=42)

        await cmd_set(msg, base.effective(100), settings, base, AsyncMock())

        self.assertEqual(base.reply_probability, 0.1)
        # And it must not quietly fall back to scoping the change to this
        # chat: the caller asked for "everywhere" and has to learn it failed.
        self.assertEqual(base.chat_overrides, {})
        self.assertIn("только OWNER_ID", msg.reply.await_args.args[0])

    async def test_chat_admin_cannot_change_global_probability(self) -> None:
        from app.handlers.admin import cmd_setprob

        msg = _fake_message(text="/setprob global 0.5", user_id=7, chat_id=100)
        base = _real_runtime_state()
        base.reply_probability = 0.1
        settings = MagicMock(owner_id=42)

        await cmd_setprob(msg, base.effective(100), settings, base)

        self.assertEqual(base.reply_probability, 0.1)
        self.assertEqual(base.chat_overrides, {})

    async def test_owner_can_still_use_both_forms(self) -> None:
        from app.handlers.admin import cmd_set

        base = _real_runtime_state()
        base.reply_probability = 0.1
        settings = MagicMock(owner_id=42)

        scoped = _fake_message(
            text="/set reply_probability 0.5", user_id=42, chat_id=100
        )
        await cmd_set(scoped, base.effective(100), settings, base, AsyncMock())
        glob = _fake_message(
            text="/set global reply_probability 0.3", user_id=42, chat_id=100
        )
        await cmd_set(glob, base.effective(100), settings, base, AsyncMock())

        self.assertEqual(base.reply_probability, 0.3)
        self.assertEqual(base.effective(100).reply_probability, 0.5)

    async def test_chat_admin_can_reset_their_chat(self) -> None:
        from app.handlers.admin import cmd_set

        msg = _fake_message(text="/set reset reply_probability", user_id=7, chat_id=100)
        base = _real_runtime_state()
        base.reply_probability = 0.1
        base.set_override(100, "reply_probability", 0.5)

        await cmd_set(msg, base.effective(100), MagicMock(owner_id=42), base, AsyncMock())

        self.assertEqual(base.effective(100).reply_probability, 0.1)

    async def test_set_reset_half_life_wipes_the_chat_short_layer(self) -> None:
        # M2R-210: dropping the override changes the effective half-life, and
        # the short layer only means something against the half-life it
        # accumulated under.
        from app.handlers.admin import cmd_set

        msg = _fake_message(
            text="/set reset markov_short_half_life_days", chat_id=100
        )
        base = _real_runtime_state()
        base.set_override(100, "markov_short_half_life_days", 7.0)
        learning_service = AsyncMock()

        await cmd_set(msg, base.effective(100), MagicMock(), base, learning_service)

        learning_service.reset_short_layer.assert_awaited_once_with(100)
        self.assertIn("короткий слой", msg.reply.await_args.args[0])

    async def test_set_reset_half_life_equal_to_global_keeps_the_layer(self) -> None:
        # The override equals the global value: the effective half-life does
        # not change, so wiping the layer would throw data away for nothing.
        from app.handlers.admin import cmd_set

        msg = _fake_message(
            text="/set reset markov_short_half_life_days", chat_id=100
        )
        base = _real_runtime_state()
        base.set_override(100, "markov_short_half_life_days", 3.0)
        learning_service = AsyncMock()

        await cmd_set(msg, base.effective(100), MagicMock(), base, learning_service)

        learning_service.reset_short_layer.assert_not_awaited()

    async def test_set_global_half_life_compares_against_the_base(self) -> None:
        # The invoking chat's override (7.0) equals the new value, but the
        # BASE changes 3.0 -> 7.0 — every other chat's short layer must go.
        from app.handlers.admin import cmd_set

        msg = _fake_message(
            text="/set global markov_short_half_life_days 7", user_id=42, chat_id=100
        )
        base = _real_runtime_state()
        base.set_override(100, "markov_short_half_life_days", 7.0)
        learning_service = AsyncMock()

        await cmd_set(
            msg, base.effective(100), MagicMock(owner_id=42), base, learning_service
        )

        self.assertEqual(base.markov_short_half_life_days, 7.0)
        learning_service.reset_short_layer.assert_awaited_once_with(None)

    async def test_set_global_half_life_unchanged_base_keeps_the_layers(self) -> None:
        # The base stays 3.0; the invoking chat's differing override must not
        # fake a change and trigger a global wipe.
        from app.handlers.admin import cmd_set

        msg = _fake_message(
            text="/set global markov_short_half_life_days 3", user_id=42, chat_id=100
        )
        base = _real_runtime_state()
        base.set_override(100, "markov_short_half_life_days", 7.0)
        learning_service = AsyncMock()

        await cmd_set(
            msg, base.effective(100), MagicMock(owner_id=42), base, learning_service
        )

        learning_service.reset_short_layer.assert_not_awaited()

    async def test_set_global_conflicting_with_a_chat_override_is_refused(self) -> None:
        # Globally valid, but chat 200 overrode typing_max_ms below the new
        # min — the refusal must name the conflicting chat.
        from app.handlers.admin import cmd_set

        msg = _fake_message(
            text="/set global typing_min_ms 900", user_id=42, chat_id=100
        )
        base = _real_runtime_state()
        base.set_override(200, "typing_max_ms", 500)

        await cmd_set(
            msg, base.effective(100), MagicMock(owner_id=42), base, AsyncMock()
        )

        self.assertEqual(base.typing_min_ms, 350)
        text = msg.reply.await_args.args[0]
        self.assertIn("Некорректное значение", text)
        self.assertIn("200", text)

    async def test_set_reset_breaking_the_chat_view_is_refused(self) -> None:
        # Chat lowered both typing bounds; resetting only the min would pair
        # the global min with the overridden max — an inverted band.
        from app.handlers.admin import cmd_set

        msg = _fake_message(text="/set reset typing_min_ms", chat_id=100)
        base = _real_runtime_state()
        base.set_override(100, "typing_min_ms", 100)
        base.set_override(100, "typing_max_ms", 200)

        await cmd_set(msg, base.effective(100), MagicMock(), base, AsyncMock())

        self.assertEqual(base.effective(100).typing_min_ms, 100)
        self.assertIn("Сброс не применён", msg.reply.await_args.args[0])

    def test_set_and_setprob_are_registered_for_chat_admins(self) -> None:
        from app.filters import AdminOrOwner
        from app.handlers.admin import cmd_set, cmd_setprob, router

        for callback in (cmd_set, cmd_setprob):
            with self.subTest(command=callback.__name__):
                handler = next(
                    h for h in router.message.handlers if h.callback is callback
                )
                filter_types = {type(f.callback) for f in handler.filters or ()}
                self.assertIn(AdminOrOwner, filter_types)

    async def test_quirk_stats_reports_numbers_for_the_chat(self) -> None:
        from app.handlers.admin import cmd_quirk_stats

        msg = _fake_message(text="/quirk_stats", chat_id=100)
        state = _fake_state()
        state.user_quirk_min_interactions = 25
        learning_service = AsyncMock()
        learning_service.get_user_interaction_stats = AsyncMock(
            return_value=(4, 31, 1, ((1, 6, 2), (6, 12, 1), (12, 18, 0), (18, 25, 0), (25, 0, 1)))
        )

        await cmd_quirk_stats(msg, state, learning_service)

        text = msg.reply.await_args.args[0]
        for expected in ("25", "4", "31", "1"):
            self.assertIn(expected, text)

    async def test_quirk_stats_says_when_the_threshold_is_unreached(self) -> None:
        # The point of the command: a feature that never fires must be
        # distinguishable from a broken one.
        from app.handlers.admin import cmd_quirk_stats

        msg = _fake_message(text="/quirk_stats")
        state = _fake_state()
        state.user_quirk_min_interactions = 25
        learning_service = AsyncMock()
        learning_service.get_user_interaction_stats = AsyncMock(
            return_value=(3, 6, 0, ((1, 6, 2), (6, 12, 1), (12, 18, 0), (18, 25, 0), (25, 0, 0)))
        )

        await cmd_quirk_stats(msg, state, learning_service)

        self.assertIn("Порог пока не взял никто", msg.reply.await_args.args[0])

    async def test_quirk_stats_handles_an_empty_chat(self) -> None:
        from app.handlers.admin import cmd_quirk_stats

        msg = _fake_message(text="/quirk_stats")
        state = _fake_state()
        learning_service = AsyncMock()
        learning_service.get_user_interaction_stats = AsyncMock(
            return_value=(0, 0, 0, ())
        )

        await cmd_quirk_stats(msg, state, learning_service)

        text = msg.reply.await_args.args[0]
        self.assertIn("не накоплено", text)

    async def test_quirk_stats_reveals_no_identifiers(self) -> None:
        from app.handlers.admin import cmd_quirk_stats

        msg = _fake_message(text="/quirk_stats", user_id=777, chat_id=100)
        state = _fake_state()
        state.user_quirk_min_interactions = 25
        learning_service = AsyncMock()
        learning_service.get_user_interaction_stats = AsyncMock(
            return_value=(4, 31, 1, ((1, 6, 2), (6, 12, 1), (12, 18, 0), (18, 25, 0), (25, 0, 1)))
        )

        await cmd_quirk_stats(msg, state, learning_service)

        text = msg.reply.await_args.args[0]
        # Counters are anonymous by construction; diagnostics must not undo it.
        self.assertNotIn("777", text)
        self.assertNotIn("100", text)
        # 4.4: строка распределения — только диапазоны и числа, без имён и
        # хэшей; форма проверяется целиком, а не отсутствием одной подстроки.
        distribution = next(
            line for line in text.splitlines() if line.startswith("распределение")
        )
        self.assertRegex(distribution, r"^распределение: [0-9+–:, ]+$")
        # And the service is asked for an aggregate, not for anyone's count.
        learning_service.get_user_interaction_count.assert_not_called()

    async def test_quirk_stats_shows_the_distribution_as_ranges(self) -> None:
        """4.3: форма распределения, а не значение на человека."""
        from app.handlers.admin import cmd_quirk_stats

        msg = _fake_message(text="/quirk_stats", chat_id=100)
        state = _fake_state()
        state.user_quirk_min_interactions = 25
        learning_service = AsyncMock()
        learning_service.get_user_interaction_stats = AsyncMock(
            return_value=(
                5,
                31,
                1,
                ((1, 6, 2), (6, 12, 1), (12, 18, 1), (18, 25, 0), (25, 0, 1)),
            )
        )

        await cmd_quirk_stats(msg, state, learning_service)

        text = msg.reply.await_args.args[0]
        self.assertIn(
            "распределение: 1–5: 2, 6–11: 1, 12–17: 1, 18–24: 0, 25+: 1", text
        )

    async def test_quirk_stats_changes_nothing(self) -> None:
        """4.5: команда только читает — счётчики после вызова прежние."""
        from app.handlers.admin import cmd_quirk_stats

        msg = _fake_message(text="/quirk_stats", chat_id=100)
        state = _fake_state()
        state.user_quirk_min_interactions = 25
        learning_service = AsyncMock()
        learning_service.get_user_interaction_stats = AsyncMock(
            return_value=(4, 31, 1, ((1, 6, 3), (6, 12, 0), (12, 18, 0), (18, 25, 0), (25, 0, 1)))
        )

        await cmd_quirk_stats(msg, state, learning_service)

        learning_service.record_user_interaction.assert_not_called()
        learning_service.get_user_interaction_stats.assert_awaited_once_with(100, 25)

    async def test_quirk_stats_denied_for_non_owner(self) -> None:
        from app.handlers.admin import cmd_quirk_stats_denied

        msg = _fake_message(text="/quirk_stats")
        state = _fake_state()

        await cmd_quirk_stats_denied(msg, state)

        text = msg.reply.await_args.args[0]
        self.assertIn("OWNER_ID", text)
        self.assertNotIn("порог", text.lower())

    def test_quirk_stats_is_owner_only(self) -> None:
        from app.filters import GROUP_ONLY, OwnerOnly
        from app.handlers.admin import cmd_quirk_stats, router

        handler = next(
            h for h in router.message.handlers if h.callback is cmd_quirk_stats
        )
        filter_types = {type(f.callback) for f in handler.filters or ()}
        self.assertIn(OwnerOnly, filter_types)
        self.assertTrue(
            any(f.magic is GROUP_ONLY for f in handler.filters or ())
        )

    def test_db_snapshot_is_private_and_owner_only(self) -> None:
        # Спека db-snapshot-delivery: у документа с корпусом сообщений не
        # должно быть пути в группу. Проверено мутацией (2026-09-01): снятие
        # PRIVATE_ONLY с регистрации роняет этот тест.
        from app.filters import GROUP_ONLY, PRIVATE_ONLY, OwnerOnly
        from app.handlers.admin import cmd_db_snapshot, router

        handler = next(
            h for h in router.message.handlers if h.callback is cmd_db_snapshot
        )
        filters = handler.filters or ()
        self.assertTrue(any(f.magic is PRIVATE_ONLY for f in filters))
        self.assertIn(OwnerOnly, {type(f.callback) for f in filters})
        self.assertFalse(any(f.magic is GROUP_ONLY for f in filters))

    async def test_db_snapshot_sends_a_document_with_the_database(self) -> None:
        import os
        import sqlite3
        import tempfile

        from app.handlers.admin import cmd_db_snapshot

        tmp = tempfile.mkdtemp(prefix="test_db_snapshot_")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        db_path = os.path.join(tmp, "markov.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE t (chat_id INTEGER)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        conn.close()

        msg = _fake_message(text="/db_snapshot", chat_type="private")
        settings = MagicMock(db_path=db_path)
        bot = AsyncMock()

        await cmd_db_snapshot(msg, settings, bot)

        bot.send_document.assert_awaited_once()
        msg.reply.assert_not_awaited()

    async def test_db_snapshot_without_db_file_replies_and_sends_nothing(
        self,
    ) -> None:
        from app.handlers.admin import cmd_db_snapshot

        msg = _fake_message(text="/db_snapshot", chat_type="private")
        settings = MagicMock(db_path="no_such_dir/no_such.db")
        bot = AsyncMock()

        await cmd_db_snapshot(msg, settings, bot)

        bot.send_document.assert_not_awaited()
        msg.reply.assert_awaited_once()

    async def test_db_snapshot_failure_answers_the_owner_not_raises(self) -> None:
        from app.handlers.admin import cmd_db_snapshot

        msg = _fake_message(text="/db_snapshot", chat_type="private")
        bot = AsyncMock()
        with patch(
            "app.handlers.admin.make_snapshot",
            side_effect=RuntimeError("disk is gone"),
        ):
            with self.assertLogs("chat_markov", level="WARNING"):
                await cmd_db_snapshot(msg, MagicMock(), bot)

        bot.send_document.assert_not_awaited()
        msg.reply.assert_awaited_once()

    async def test_setprob_valid_value(self) -> None:
        from app.handlers.admin import cmd_setprob
        msg = _fake_message(text="/setprob 0.3", chat_id=100)
        base = _real_runtime_state()
        base.reply_probability = 0.1
        await cmd_setprob(msg, base.effective(100), MagicMock(), base)
        self.assertEqual(base.effective(100).reply_probability, 0.3)
        self.assertEqual(base.reply_probability, 0.1)
        msg.reply.assert_awaited_once()

    async def test_setprob_invalid_value_replies_error(self) -> None:
        from app.handlers.admin import cmd_setprob
        msg = _fake_message(text="/setprob abc")
        state = _fake_state()
        settings = MagicMock()
        await cmd_setprob(msg, state, settings, _real_runtime_state())
        msg.reply.assert_awaited_once()
        assert "диапазон" in msg.reply.call_args[0][0]

    async def test_clear_without_confirm_asks_confirmation(self) -> None:
        from app.handlers.admin import cmd_clear
        msg = _fake_message(text="/clear")
        db = AsyncMock()
        state = _fake_state()
        generator = MagicMock()
        settings = MagicMock()
        pivo_service = AsyncMock()
        learning_service = MagicMock()
        with patch("app.handlers.admin.format_clear_confirmation_message", return_value="confirm?"):
            await cmd_clear(
                msg, db, state, generator, settings, pivo_service, learning_service
            )
        db.clear_chat.assert_not_called()
        pivo_service.clear_chat_data.assert_not_called()
        learning_service.forget_chat.assert_not_called()
        msg.reply.assert_awaited_once()

    async def test_clear_with_confirm_clears_chat(self) -> None:
        from app.handlers.admin import cmd_clear
        msg = _fake_message(text="/clear confirm")
        db = AsyncMock()
        state = _fake_state()
        generator = MagicMock()
        settings = MagicMock()
        pivo_service = AsyncMock()
        learning_service = MagicMock()
        await cmd_clear(
            msg, db, state, generator, settings, pivo_service, learning_service
        )
        db.clear_chat.assert_awaited_once_with(msg.chat.id)
        pivo_service.clear_chat_data.assert_awaited_once_with(msg.chat.id)
        # Кэши обучения переживали очистку: гейт дословного повтора продолжал
        # считать цитатой сообщения, которых в базе уже нет.
        learning_service.forget_chat.assert_called_once_with(msg.chat.id)
        msg.reply.assert_awaited_once()

    # --- fallback handlers для unauthorized админ-команд ---

    async def test_set_denied_replies_with_explanation(self) -> None:
        # The scoped form is open to this chat's admins again (the O5 vector
        # is closed by scope, not by rights), so the denial names both — a
        # refusal that hides who does qualify is a worse refusal.
        from app.handlers.admin import cmd_set_denied
        msg = _fake_message(text="/set foo bar")
        state = _fake_state()
        await cmd_set_denied(msg, state)
        msg.reply.assert_awaited_once()
        text = msg.reply.call_args[0][0]
        assert "OWNER_ID" in text
        assert "админ" in text.lower()

    async def test_setprob_denied_replies_with_explanation(self) -> None:
        from app.handlers.admin import cmd_setprob_denied
        msg = _fake_message(text="/setprob 0.5")
        state = _fake_state()
        await cmd_setprob_denied(msg, state)
        msg.reply.assert_awaited_once()
        assert "OWNER_ID" in msg.reply.call_args[0][0]

    async def test_clear_denied_replies_with_explanation(self) -> None:
        from app.handlers.admin import cmd_clear_denied
        msg = _fake_message(text="/clear confirm")
        state = _fake_state()
        await cmd_clear_denied(msg, state)
        msg.reply.assert_awaited_once()
        text = msg.reply.call_args[0][0]
        assert "Недостаточно прав" in text
        # /clear имеет специфичный текст, отличный от общего «доступна OWNER_ID...»
        assert "OWNER_ID" in text or "админ" in text.lower()

    def test_admin_router_handler_registration_order(self) -> None:
        """
        Защита от случайной перестановки fallback-handlers.

        Aiogram перебирает handlers Router'а в порядке регистрации
        (см. aiogram/dispatcher/event/telegram.py:111-130: for handler
        in self.handlers с return на первом match). Если cmd_set_denied
        окажется зарегистрирован раньше cmd_set, fallback всегда будет
        выигрывать — admin-команда никогда не выполнится для реальных
        админов, и баг тихо пройдёт CI.
        """
        from app.handlers.admin import router

        callbacks = [h.callback for h in router.message.handlers]
        names = [cb.__name__ for cb in callbacks]

        pairs = [
            ("cmd_set", "cmd_set_denied"),
            ("cmd_setprob", "cmd_setprob_denied"),
            ("cmd_clear", "cmd_clear_denied"),
        ]
        for protected, fallback in pairs:
            assert protected in names, f"{protected} not registered"
            assert fallback in names, f"{fallback} not registered"
            assert names.index(protected) < names.index(fallback), (
                f"{fallback} зарегистрирован раньше {protected}: "
                f"fallback всегда будет выигрывать, защищённый handler "
                f"никогда не вызовется. Поменяйте порядок в admin.py."
            )


# ---------------------------------------------------------------------------
# pivo.py
# ---------------------------------------------------------------------------

class TestPivoHandlers(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        from app.filters import admin_or_owner
        from app.handlers import pivo as pivo_handlers

        # The admin-id cache is process-global; reset it so the shared chat_id
        # across these tests does not leak admin sets between cases.
        admin_or_owner._admin_cache.clear()
        # Same for the quota-notice throttle: the shared (chat, user) pair would
        # otherwise silence the notice in every case after the first.
        pivo_handlers.reset_quota_notice_state()

    def test_pivo_router_handlers_are_group_only(self) -> None:
        from app.filters import GROUP_ONLY
        from app.handlers.pivo import router

        expected = {
            "cmd_pivo",
            "cmd_pivo_on",
            "cmd_pivo_off",
            "cmd_pivo_privacy",
        }

        handlers = {
            handler.callback.__name__: handler
            for handler in router.message.handlers
            if handler.callback.__name__ in expected
        }

        self.assertEqual(set(handlers), expected)
        for name, handler in handlers.items():
            self.assertTrue(
                any(
                    filter_object.magic is GROUP_ONLY
                    for filter_object in handler.filters
                ),
                f"{name} must be registered with GROUP_ONLY",
            )

    async def test_pivo_calls_build_and_replies(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage("Выходи пить!", 2, {}, {})
        )
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.build_call_message.assert_awaited_once_with(
            chat_id=msg.chat.id,
            caller_user_id=msg.from_user.id,
            planned_time=None,
            target=None,
            explicit_mentions=(),
            recent_pool_window=5,
            temporal_flavor_chance=0.5,
            now=ANY,
            mention_by_id=ANY,
        )
        pivo_service.consume_daily_call_quota.assert_awaited_once_with(
            chat_id=msg.chat.id,
            user_id=msg.from_user.id,
            is_admin_or_owner=False,
        )
        msg.reply.assert_awaited_once()
        pivo_service.record_pool_usage.assert_awaited_once_with(
            msg.chat.id, {}, recent_pool_window=5
        )

    async def test_pivo_now_is_taken_in_the_chat_timezone(self) -> None:
        # O12: временны́е ветки /pivo (ночь/пятница/сезон) считаются по зоне
        # CHAT_TIMEZONE, а не по часам контейнера.
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage("Выходи пить!", 2, {}, {})
        )
        state = _fake_state()
        state.chat_timezone = ZoneInfo("Europe/Moscow")
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        now = pivo_service.build_call_message.await_args.kwargs["now"]
        self.assertIs(now.tzinfo, state.chat_timezone)

    async def test_pivo_passes_parsed_arguments_to_service(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message(text="/pivo 20:00 watch movie @friend")
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage("Выходи пить!", 1, {}, {})
        )
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_awaited_once()
        pivo_service.build_call_message.assert_awaited_once_with(
            chat_id=msg.chat.id,
            caller_user_id=msg.from_user.id,
            planned_time="20:00",
            target="watch movie",
            explicit_mentions=("@friend",),
            recent_pool_window=5,
            temporal_flavor_chance=0.5,
            now=ANY,
            mention_by_id=ANY,
        )
        msg.reply.assert_awaited_once()

    async def test_pivo_rejects_over_limit_mentions_without_spending_quota(self) -> None:
        from app.handlers.pivo import cmd_pivo
        from app.services.pivo_service import PivoCallLimitError

        msg = _fake_message(text="/pivo @one @two @three")
        pivo_service = AsyncMock()
        pivo_service.ensure_explicit_mentions_allowed = AsyncMock(
            side_effect=PivoCallLimitError(
                "В /pivo можно указывать не больше 2 явных упоминаний за раз."
            )
        )
        state = _fake_state()
        bot = AsyncMock()
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        # Отказ по явным упоминаниям проверяется до квоты и до базы: список
        # пришёл из текста команды, и он не должен стоить суточной квоты.
        pivo_service.ensure_explicit_mentions_allowed.assert_awaited_once()
        pivo_service.consume_daily_call_quota.assert_not_called()
        pivo_service.build_call_message.assert_not_called()
        msg.reply.assert_awaited_once()
        assert "не больше 2" in msg.reply.call_args[0][0]

    async def test_pivo_rejects_when_daily_quota_is_exhausted(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(
            allowed=False,
            limit=3,
            usage_day="2026-05-12",
        )
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage("Выходи пить!", 2, {}, {})
        )
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_awaited_once()
        # Квота списывается до сборки: отклонённый вызов не должен оплачивать
        # чтение подписчиков и расшифровку их данных.
        pivo_service.build_call_message.assert_not_called()
        # N3: a quota-rejected call must not rotate the anti-repeat pools.
        pivo_service.record_pool_usage.assert_not_called()
        msg.reply.assert_awaited_once()
        assert "Лимит /pivo" in msg.reply.call_args[0][0]

    async def test_pivo_over_quota_explains_once_per_window(self) -> None:
        """Отказ по квоте не превращается в поток сообщений.

        /pivo выведен из-под общего ограничения частоты, а квота ограничивала
        только успешные вызовы: отказ отвечался на каждый, и участник,
        повторяющий команду, получал неограниченный поток ответов бота.
        """
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=False, limit=3, usage_day="2026-08-06")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        for _ in range(3):
            await cmd_pivo(msg, pivo_service, state, bot, settings)

        self.assertEqual(pivo_service.consume_daily_call_quota.await_count, 3)
        self.assertEqual(msg.reply.await_count, 1)

    async def test_pivo_over_quota_explains_again_after_window(self) -> None:
        from app.handlers import pivo as pivo_handlers
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=False, limit=3, usage_day="2026-08-06")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)
        # Сдвигаем отметку в прошлое на всё окно — как если бы оно истекло.
        for key in list(pivo_handlers._quota_notice_sent):
            pivo_handlers._quota_notice_sent[key] -= (
                pivo_handlers.QUOTA_NOTICE_WINDOW_SEC + 1.0
            )
        await cmd_pivo(msg, pivo_service, state, bot, settings)

        self.assertEqual(msg.reply.await_count, 2)

    async def test_pivo_over_quota_failed_notice_does_not_consume_window(self) -> None:
        # Окно отмечается после успешной отправки: упавший ответ не должен
        # оставить повтор команды без объяснения на всё окно.
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        msg.reply = AsyncMock(side_effect=[RuntimeError("telegram down"), None])
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=False, limit=3, usage_day="2026-08-06")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        with self.assertRaises(RuntimeError):
            await cmd_pivo(msg, pivo_service, state, bot, settings)
        await cmd_pivo(msg, pivo_service, state, bot, settings)

        self.assertEqual(msg.reply.await_count, 2)

    async def test_pivo_over_quota_notice_is_per_user(self) -> None:
        from app.handlers.pivo import cmd_pivo
        first = _fake_message(user_id=111)
        second = _fake_message(user_id=222)
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=False, limit=3, usage_day="2026-08-06")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        await cmd_pivo(first, pivo_service, state, bot, settings)
        await cmd_pivo(second, pivo_service, state, bot, settings)

        first.reply.assert_awaited_once()
        second.reply.assert_awaited_once()

    async def test_pivo_refunds_quota_when_build_fails(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-08-06")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            side_effect=RuntimeError("db is gone")
        )
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        with self.assertRaises(RuntimeError):
            await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.refund_daily_call_quota.assert_awaited_once_with(
            chat_id=msg.chat.id,
            user_id=msg.from_user.id,
            usage_day=quota.usage_day,
        )

    async def test_pivo_post_processing_failure_does_not_fail_the_call(self) -> None:
        """Сбой после доставки не отменяет состоявшийся вызов."""
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-08-06")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage("Выходи пить!", 2, {}, {})
        )
        pivo_service.record_pool_usage = AsyncMock(
            side_effect=RuntimeError("disk is gone")
        )
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        with self.assertLogs("chat_markov", level="ERROR") as captured:
            await cmd_pivo(msg, pivo_service, state, bot, settings)

        msg.reply.assert_awaited_once()
        pivo_service.refund_daily_call_quota.assert_not_called()
        self.assertTrue(
            any("post-processing failed" in line for line in captured.output)
        )

    async def test_pivo_uses_admin_daily_quota_path(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage("Выходи пить!", 2, {}, {})
        )
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(
            return_value=[MagicMock(user=MagicMock(id=msg.from_user.id))]
        )
        settings = MagicMock(owner_id=None)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_awaited_once_with(
            chat_id=msg.chat.id,
            user_id=msg.from_user.id,
            is_admin_or_owner=True,
        )
        pivo_service.build_call_message.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_pivo_owner_bypasses_daily_quota(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message(user_id=42)
        pivo_service = AsyncMock()
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage("Выходи пить!", 2, {}, {})
        )
        state = _fake_state()
        bot = AsyncMock()
        settings = MagicMock(owner_id=42)

        await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_not_called()
        bot.get_chat_administrators.assert_not_called()
        pivo_service.build_call_message.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_pivo_refunds_quota_when_reply_fails(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        msg.reply = AsyncMock(side_effect=RuntimeError("telegram failed"))
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage("Выходи пить!", 2, {}, {})
        )
        pivo_service.refund_daily_call_quota = AsyncMock()
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        with self.assertRaises(RuntimeError):
            await cmd_pivo(msg, pivo_service, state, bot, settings)

        pivo_service.consume_daily_call_quota.assert_awaited_once()
        pivo_service.refund_daily_call_quota.assert_awaited_once_with(
            chat_id=msg.chat.id,
            user_id=msg.from_user.id,
            usage_day=quota.usage_day,
        )

    async def test_pivo_logs_mention_aggregate_without_personal_data(self) -> None:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        sent = MagicMock()
        sent.entities = [
            SimpleNamespace(type="text_mention"),
            SimpleNamespace(type="text_mention"),
            SimpleNamespace(type="bold"),
        ]
        msg.reply = AsyncMock(return_value=sent)
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-05-12")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage(
                "Выходи пить! @PepeUser",
                2,
                {},
                {"by_id": 2, "by_username": 0, "skipped": 1},
            )
        )
        state = _fake_state()
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        settings = MagicMock(owner_id=None)

        # Маскирование включено, как в живом процессе, — иначе тест проверял бы
        # только аварийную заглушку.
        from app import log_masking

        log_masking.init_masking("test-secret-for-masking")
        self.addCleanup(log_masking.reset_masking)

        with self.assertLogs("chat_markov", level="INFO") as captured:
            await cmd_pivo(msg, pivo_service, state, bot, settings)

        aggregate = [line for line in captured.output if "pivo mentions" in line]
        self.assertEqual(len(aggregate), 1)
        record = aggregate[0]
        # Числа из entity сервера — это и есть машинный сигнал по O2.
        self.assertIn("text_mention=2", record)
        self.assertIn("mention=0", record)
        self.assertIn("'by_id': 2", record)
        # Ни ников, ни сырого chat_id в записи быть не должно.
        self.assertNotIn("PepeUser", record)
        self.assertNotIn(str(msg.chat.id), record)

    async def _run_pivo_with_report(
        self,
        *,
        owner_id: int | None = 42,
        report_enabled: bool = True,
        send_side_effect: Exception | None = None,
    ) -> tuple[MagicMock, AsyncMock, AsyncMock]:
        from app.handlers.pivo import cmd_pivo
        msg = _fake_message()
        msg.chat.title = "Пивной чат"
        sent = MagicMock()
        sent.entities = [SimpleNamespace(type="text_mention")]
        msg.reply = AsyncMock(return_value=sent)
        pivo_service = AsyncMock()
        quota = MagicMock(allowed=True, usage_day="2026-08-05")
        pivo_service.consume_daily_call_quota = AsyncMock(return_value=quota)
        pivo_service.build_call_message = AsyncMock(
            return_value=PivoCallMessage(
                "Выходи пить! @PepeUser",
                2,
                {},
                {"by_id": 2, "by_username": 0, "skipped": 1},
            )
        )
        state = _fake_state()
        state.pivo_report_to_owner = report_enabled
        bot = AsyncMock()
        bot.get_chat_administrators = AsyncMock(return_value=[])
        bot.send_message = AsyncMock(side_effect=send_side_effect)
        settings = MagicMock(owner_id=owner_id)

        await cmd_pivo(msg, pivo_service, state, bot, settings)
        return msg, bot, pivo_service

    async def test_pivo_reports_aggregate_to_owner(self) -> None:
        _msg, bot, _svc = await self._run_pivo_with_report()

        bot.send_message.assert_awaited_once()
        target, text = bot.send_message.await_args.args[:2]
        self.assertEqual(target, 42)
        # Владелец логов не читает — числа должны быть в самом сообщении.
        self.assertIn("Пивной чат", text)
        self.assertIn("text_mention=1", text)
        self.assertIn("2", text)
        # Персональных данных подписчиков в отчёте быть не должно.
        self.assertNotIn("PepeUser", text)

    async def test_pivo_does_not_report_without_owner(self) -> None:
        _msg, bot, _svc = await self._run_pivo_with_report(owner_id=None)

        bot.send_message.assert_not_called()

    async def test_pivo_does_not_report_when_knob_is_off(self) -> None:
        _msg, bot, _svc = await self._run_pivo_with_report(report_enabled=False)

        bot.send_message.assert_not_called()

    async def test_failed_owner_report_does_not_break_pivo(self) -> None:
        # Сообщение в чат уже доставлено: отчёт не может отменить вызов.
        _msg, bot, pivo_service = await self._run_pivo_with_report(
            send_side_effect=RuntimeError("bot was blocked by the user")
        )

        bot.send_message.assert_awaited_once()
        pivo_service.record_pool_usage.assert_awaited_once()
        pivo_service.refund_daily_call_quota.assert_not_called()

    async def test_pivo_check_lists_paths_for_owner(self) -> None:
        from app.domain.pivo import PivoMentionResult
        from app.handlers.pivo import cmd_pivo_check
        msg = _fake_message()
        pivo_service = AsyncMock()
        pivo_service.describe_mention_paths = AsyncMock(
            return_value=[
                PivoMentionResult(
                    text='<a href="tg://user?id=1">@PepeUser</a>',
                    path="by_id",
                    reason="ссылка по user_id",
                    label="@PepeUser",
                ),
                PivoMentionResult(
                    text="", path="skipped", reason="нет user_id и ника", label=""
                ),
            ]
        )
        state = _fake_state()

        await cmd_pivo_check(msg, pivo_service, state)

        reply_text = msg.reply.await_args.args[0]
        self.assertIn("by_id", reply_text)
        self.assertIn("skipped", reply_text)
        self.assertIn("PepeUser", reply_text)
        # Диагностика не должна пинговать чат: ни разметки упоминания,
        # ни «@» перед ником в ответе быть не должно.
        self.assertNotIn("tg://user", reply_text)
        self.assertNotIn("@PepeUser", reply_text)

    async def test_pivo_check_reports_empty_subscriber_list(self) -> None:
        from app.handlers.pivo import cmd_pivo_check
        msg = _fake_message()
        pivo_service = AsyncMock()
        pivo_service.describe_mention_paths = AsyncMock(return_value=[])
        state = _fake_state()

        await cmd_pivo_check(msg, pivo_service, state)

        self.assertIn("нет подписчиков", msg.reply.await_args.args[0])

    async def test_pivo_check_denied_for_non_owner(self) -> None:
        from app.handlers.pivo import cmd_pivo_check_denied
        msg = _fake_message()
        state = _fake_state()

        await cmd_pivo_check_denied(msg, state)

        reply_text = msg.reply.await_args.args[0]
        self.assertIn("владельцу", reply_text)
        # Отказ не должен раскрывать состав подписчиков.
        self.assertNotIn("by_id", reply_text)

    async def test_pivo_check_is_owner_only(self) -> None:
        from app.filters import GROUP_ONLY, OwnerOnly
        from app.handlers.pivo import cmd_pivo_check, router

        handler = next(
            h for h in router.message.handlers if h.callback is cmd_pivo_check
        )
        filter_types = {type(f.callback) for f in handler.filters or ()}
        self.assertIn(OwnerOnly, filter_types)
        self.assertTrue(
            any(f.magic is GROUP_ONLY for f in handler.filters or ())
        )

    async def test_pivo_on_subscribes_user(self) -> None:
        from app.handlers.pivo import cmd_pivo_on
        msg = _fake_message(is_bot=False)
        pivo_service = AsyncMock()
        state = _fake_state()
        await cmd_pivo_on(msg, pivo_service, state)
        pivo_service.subscribe.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_pivo_on_ignores_bots(self) -> None:
        from app.handlers.pivo import cmd_pivo_on
        msg = _fake_message(is_bot=True)
        pivo_service = AsyncMock()
        state = _fake_state()
        await cmd_pivo_on(msg, pivo_service, state)
        pivo_service.subscribe.assert_not_called()
        msg.reply.assert_awaited_once()

    async def test_pivo_off_unsubscribes_user(self) -> None:
        from app.handlers.pivo import cmd_pivo_off
        msg = _fake_message()
        pivo_service = AsyncMock()
        state = _fake_state()
        await cmd_pivo_off(msg, pivo_service, state)
        pivo_service.unsubscribe.assert_awaited_once()
        msg.reply.assert_awaited_once()

    async def test_pivo_privacy_replies(self) -> None:
        from app.handlers.pivo import cmd_pivo_privacy
        msg = _fake_message()
        state = _fake_state()
        await cmd_pivo_privacy(msg, state)
        msg.reply.assert_awaited_once()


# ---------------------------------------------------------------------------
# reply_pipeline.py — extract_context_tokens
# ---------------------------------------------------------------------------

class TestExtractContextTokens(unittest.TestCase):
    def _call(self, **kwargs: object) -> list[str]:
        from app.services.reply_pipeline import extract_context_tokens
        defaults: dict[str, object] = dict(
            current_text="hello world",
            reply_context_text=None,
            is_reply=False,
            normalize_lower=False,
            max_tokens=10,
            only_for_replies=False,
            include_current_message=True,
        )
        defaults.update(kwargs)
        return extract_context_tokens(**defaults)  # type: ignore[arg-type]

    def test_returns_tokens_from_current_text(self) -> None:
        tokens = self._call(current_text="один два три")
        self.assertGreater(len(tokens), 0)

    def test_only_for_replies_returns_empty_when_no_reply(self) -> None:
        tokens = self._call(is_reply=False, only_for_replies=True)
        self.assertEqual(tokens, [])

    def test_max_tokens_respected(self) -> None:
        tokens = self._call(
            current_text="один два три четыре пять шесть семь восемь девять десять одиннадцать",
            max_tokens=3,
        )
        self.assertLessEqual(len(tokens), 3)

    def test_reply_to_bot_own_message_is_dropped_from_context(self) -> None:
        # Human replies to the bot: the handler passes no reply context for the
        # bot's own line (self-echo cause); the human current message still
        # anchors the reply.
        tokens = self._call(
            is_reply=True,
            reply_context_text=None,
            current_text="урон по жопе демида",
            only_for_replies=True,
        )
        self.assertEqual(tokens, ["урон", "по", "жопе", "демида"])

    def test_reply_to_human_message_is_kept_in_context(self) -> None:
        tokens = self._call(
            is_reply=True,
            reply_context_text="люблю кофе",
            current_text="а я утром",
            only_for_replies=True,
        )
        self.assertEqual(tokens, ["люблю", "кофе", "а", "я", "утром"])

    def test_reply_to_bot_without_current_yields_empty(self) -> None:
        # Non-default: include_current off + reply is the bot's own -> no
        # context at all (documented degradation for that config).
        tokens = self._call(
            is_reply=True,
            reply_context_text=None,
            current_text="ответ человека",
            only_for_replies=True,
            include_current_message=False,
        )
        self.assertEqual(tokens, [])


class TestIncomingMessageFacts(unittest.TestCase):
    """Фильтр «ответ на собственное сообщение бота» живёт в обработчике.

    Слова бота построены по корпусу, поэтому их n-граммы совпадают с
    сохранёнными стартовыми состояниями: вернув их в контекст, бот открывает
    новый ответ дословным повтором предыдущего.
    """

    def _incoming(self, reply_to: object) -> object:
        from app.handlers.learning import _incoming_message

        message = SimpleNamespace(
            chat=SimpleNamespace(id=1),
            from_user=SimpleNamespace(id=5, first_name="Аня"),
            text="урон по жопе демида",
            reply_to_message=reply_to,
        )
        with patch(
            "app.handlers.learning.bot_is_mentioned", return_value=False
        ):
            return _incoming_message(
                message,  # type: ignore[arg-type]
                bot_username="PepeEdtaBot",
                bot_id=999,
                bot_text_aliases=frozenset({"пепе"}),
                monotonic_now=0.0,
            )

    def test_reply_to_bot_message_yields_no_reply_context(self) -> None:
        incoming = self._incoming(
            SimpleNamespace(
                text="например кая скейлила урон",
                from_user=SimpleNamespace(id=999),
            )
        )
        self.assertTrue(incoming.is_reply)  # type: ignore[attr-defined]
        self.assertIsNone(incoming.reply_context_text)  # type: ignore[attr-defined]

    def test_reply_to_human_message_keeps_reply_context(self) -> None:
        incoming = self._incoming(
            SimpleNamespace(
                text="люблю кофе",
                from_user=SimpleNamespace(id=111),
            )
        )
        self.assertEqual(
            incoming.reply_context_text,  # type: ignore[attr-defined]
            "люблю кофе",
        )


class TestReplyHumanizedResilience(unittest.IsolatedAsyncioTestCase):
    """`reply_humanized_sequence` is the single point that calls Telegram's chat-action
    API. A 5xx or network blip there must not block the actual reply — the
    helper catches the exception and proceeds to `message.reply`."""

    async def test_send_chat_action_failure_does_not_block_reply(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message()
        msg.bot.send_chat_action = AsyncMock(
            side_effect=RuntimeError("telegram chat-action 5xx")
        )

        await reply_humanized_sequence(msg, ["ответ"], 0, 0)

        msg.reply.assert_awaited_once_with("ответ")

    async def test_send_chat_action_called_when_bot_present(self) -> None:
        from app.handlers._helpers import reply_humanized_sequence

        msg = _fake_message()
        await reply_humanized_sequence(msg, ["ответ"], 0, 0)
        msg.bot.send_chat_action.assert_awaited_once()
        msg.reply.assert_awaited_once_with("ответ")

class TestComputeTypingDelay(unittest.TestCase):
    def test_zero_per_char_keeps_base_range(self) -> None:
        from app.handlers._helpers import compute_typing_delay_ms

        rng = random.Random(1)
        for _ in range(20):
            delay = compute_typing_delay_ms(500, 350, 1100, 0, rng=rng)
            self.assertGreaterEqual(delay, 350)
            self.assertLessEqual(delay, 1100)

    def test_longer_text_waits_longer(self) -> None:
        from app.handlers._helpers import compute_typing_delay_ms

        short = compute_typing_delay_ms(10, 350, 350, 12, rng=random.Random(1))
        long = compute_typing_delay_ms(200, 350, 350, 12, rng=random.Random(1))
        self.assertEqual(short, 350 + 10 * 12)
        self.assertEqual(long, 350 + 200 * 12)

    def test_delay_is_capped(self) -> None:
        from app.handlers._helpers import (
            TYPING_HARD_CAP_MS,
            compute_typing_delay_ms,
        )

        delay = compute_typing_delay_ms(4000, 350, 1100, 200, rng=random.Random(1))
        self.assertEqual(delay, TYPING_HARD_CAP_MS)

    def test_cap_never_cuts_below_configured_max(self) -> None:
        from app.handlers._helpers import compute_typing_delay_ms

        delay = compute_typing_delay_ms(0, 5000, 5000, 0, rng=random.Random(1))
        self.assertEqual(delay, 5000)


class TestPivoChatActionResilience(unittest.IsolatedAsyncioTestCase):
    async def test_pivo_on_still_subscribes_even_when_chat_action_fails(self) -> None:
        from app.handlers.pivo import cmd_pivo_on

        msg = _fake_message(is_bot=False)
        msg.bot.send_chat_action = AsyncMock(
            side_effect=RuntimeError("transient telegram error")
        )
        pivo_service = AsyncMock()
        state = _fake_state()

        await cmd_pivo_on(msg, pivo_service, state)

        pivo_service.subscribe.assert_awaited_once()
        msg.reply.assert_awaited_once()


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


class TestLearningMessageLength(unittest.TestCase):
    def test_learning_message_length_boundaries(self) -> None:
        from app.services.reply_pipeline import (
            MAX_LEARN_MESSAGE_CHARS,
            is_learnable_message_length,
        )

        self.assertTrue(is_learnable_message_length("x" * MAX_LEARN_MESSAGE_CHARS))
        self.assertFalse(is_learnable_message_length("x" * (MAX_LEARN_MESSAGE_CHARS + 1)))

    def test_learning_token_boundaries(self) -> None:
        from app.services.reply_pipeline import has_enough_tokens_for_learning

        self.assertFalse(has_enough_tokens_for_learning(["один"]))
        self.assertTrue(has_enough_tokens_for_learning(["один", "два"]))


class TestStripLeadingBotVocative(unittest.TestCase):
    aliases = frozenset({"pepe", "пепе"})

    def test_strips_leading_alias_with_separators(self) -> None:
        from app.services.reply_pipeline import strip_leading_bot_vocative

        for text, expected in (
            ("Пепе, расскажи анекдот", "расскажи анекдот"),
            ("pepe: what is up", "what is up"),
            ("Пепе — скажи что-нибудь", "скажи что-нибудь"),
            ("  пепе,  привет всем", "привет всем"),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    strip_leading_bot_vocative(text, self.aliases), expected
                )

    def test_strips_leading_alias_without_separator(self) -> None:
        # SIM-8: a bare "<alias> ..." address is as common as the comma form;
        # unstripped it taught the corpus the bot's own name.
        from app.services.reply_pipeline import strip_leading_bot_vocative

        for text, expected in (
            ("Пепе хороший бот", "хороший бот"),
            ("пепе кто гнойный пидор", "кто гнойный пидор"),
            ("  pepe   what is up", "what is up"),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    strip_leading_bot_vocative(text, self.aliases), expected
                )

    def test_preserves_bare_alias_with_no_content(self) -> None:
        from app.services.reply_pipeline import strip_leading_bot_vocative

        for text in ("Пепе", "пепе  ", " pepe"):
            with self.subTest(text=text):
                self.assertEqual(strip_leading_bot_vocative(text, self.aliases), text)

    def test_preserves_mid_sentence_alias_and_other_vocatives(self) -> None:
        from app.services.reply_pipeline import strip_leading_bot_vocative

        for text in ("Ребята, привет", "скажи пепе, что делать", "Москва, я люблю"):
            with self.subTest(text=text):
                self.assertEqual(strip_leading_bot_vocative(text, self.aliases), text)

    def test_empty_aliases_is_noop(self) -> None:
        from app.services.reply_pipeline import strip_leading_bot_vocative

        text = "Пепе, привет"
        self.assertEqual(strip_leading_bot_vocative(text, frozenset()), text)


class TestLearningHandler(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        mask_patcher = patch(
            "app.core.response_generator.mask_chat_id",
            return_value="chat",
        )
        mask_patcher.start()
        self.addCleanup(mask_patcher.stop)

    def _reply_state(self, **overrides: object) -> MagicMock:
        values: dict[str, object] = {
            "normalize_lower": False,
            "learned_messages": {},
            "recent_short_replies": {},
            "min_tokens_for_model": 10,
            "min_cooldown_sec": 0,
            "last_reply_ts": {},
            "reply_probability": 0.0,
            "use_reply_context": False,
            "fuzzy_context_casefold": False,
            "reply_context_bias": 1.8,
            "reply_context_start_bias": 2.2,
            "context_start_affinity": 3.0,
            "max_reply_chars": 280,
            "max_reply_tokens": 45,
            "randomness_strength": 0.0,
            "repetition_penalty_strength": 1.0,
            "recent_reply_penalty_strength": 1.0,
            "length_mode_weights": (0.25, 0.55, 0.2),
            "intonation_profile_strength": 0.0,
            "length_context_adaptation": 0.0,
            "markov_order": 3,
            "enable_backoff": True,
        }
        values.update(overrides)
        return _fake_state(**values)

    async def test_single_token_message_is_not_learned(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="hello")
        learning_service = AsyncMock()
        generator = _traced_generator()
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.record_message.assert_not_called()
        msg.reply.assert_not_awaited()

    async def test_vocative_stripped_and_pii_redacted_before_learning(self) -> None:
        from app.handlers.learning import on_text_message

        # Leading bot vocative + a phone number: the vocative is stripped from the
        # text fed to record_message, and the phone is redacted out of the model
        # tokens (via sanitize_text), keeping the corpus clean.
        msg = _fake_message(text="pepe, мой телефон +380441234567 ладно")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state()

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.record_message.assert_awaited_once()
        kwargs = learning_service.record_message.await_args.kwargs
        self.assertEqual(kwargs["raw_text"], "мой телефон +380441234567 ладно")
        self.assertEqual(kwargs["tokens"], ["мой", "телефон", "ладно"])

    async def test_threshold_crossing_message_is_learned_without_replying_to_itself(
        self,
    ) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe threshold crossing")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=9)
        learning_service.record_message = AsyncMock(return_value=11)
        generator = _traced_generator()
        state = self._reply_state()

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.get_token_volume.assert_awaited_once_with(msg.chat.id)
        learning_service.record_message.assert_awaited_once()
        generator.generate_text.assert_not_awaited()
        from app.presentation.fallback_phrases import (
            LATE_NIGHT_FALLBACK_PHRASES,
            NOT_ENOUGH_DATA_PHRASES,
        )

        msg.reply.assert_awaited_once()
        # The handler injects the wall clock, so in the small hours the late-night
        # pool is a valid extension (S4). Accept either so the assertion does not
        # depend on the time the CI run happens to execute.
        self.assertIn(
            msg.reply.await_args.args[0],
            set(NOT_ENOUGH_DATA_PHRASES) | set(LATE_NIGHT_FALLBACK_PHRASES),
        )

    async def test_current_incoming_message_copy_is_rejected(self) -> None:
        from app.handlers.learning import on_text_message

        # SIM-8: the leading alias is stripped before learning, so the echo
        # gate compares against the stripped text ("ответь мне").
        msg = _fake_message(text="pepe ответь мне")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=["ответь мне", "Нормально"] + [""] * 8
        )
        state = self._reply_state()

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        self.assertEqual(generator.generate_text.await_count, 10)
        learning_service.record_message.assert_awaited_once()
        msg.reply.assert_awaited_once_with("Нормально")

    async def test_profile_refresh_failure_does_not_abort_learning(self) -> None:
        # Косметика /pivo-профиля не должна стоить обучения на сообщении.
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="обычное сообщение из чата")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        generator = _traced_generator()
        state = self._reply_state()
        pivo_service = _pivo_stub()
        pivo_service.refresh_member.side_effect = RuntimeError("db busy")

        with (
            patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"),
            self.assertLogs("chat_markov", level="WARNING") as captured,
        ):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                pivo_service,
            )

        learning_service.record_message.assert_awaited_once()
        self.assertTrue(
            any("profile refresh failed" in line for line in captured.output)
        )

    async def test_learn_failure_does_not_mask_the_respond_error(self) -> None:
        # Ошибка respond() должна долететь до обработчика ошибок, а не быть
        # подменённой сбоем pipeline.learn из finally; сбой learn — в логе.
        from app.handlers import learning as learning_handlers
        from app.handlers.learning import on_text_message

        pipeline = AsyncMock()
        pipeline.observe.return_value = MagicMock()
        pipeline.respond.side_effect = RuntimeError("respond down")
        pipeline.learn.side_effect = RuntimeError("learn down")
        pipeline.run_due_maintenance.return_value = None

        with (
            patch.object(learning_handlers, "ReplyPipeline", return_value=pipeline),
            self.assertLogs("chat_markov", level="ERROR") as captured,
            self.assertRaises(RuntimeError) as ctx,
        ):
            await on_text_message(
                _fake_message(text="обычное сообщение из чата"),
                AsyncMock(),
                _traced_generator(),
                self._reply_state(),
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        self.assertEqual(str(ctx.exception), "respond down")
        self.assertTrue(
            any("learning failed" in line for line in captured.output)
        )
        pipeline.run_due_maintenance.assert_awaited_once()

    async def test_message_is_persisted_when_cooldown_blocks_reply(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="обычное сообщение")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        generator = _traced_generator()
        state = self._reply_state(
            last_reply_ts={msg.chat.id: 10**20},
            min_cooldown_sec=60,
            reply_probability=1.0,
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.record_message.assert_awaited_once()
        generator.generate_text.assert_not_awaited()
        msg.reply.assert_not_awaited()

    async def test_message_is_persisted_when_generation_fails(self) -> None:
        from app.core.response_generator import GENERATION_ATTEMPT_BUDGET
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe generate something")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=103)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state()

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.record_message.assert_awaited_once()
        self.assertEqual(
            generator.generate_text.await_count,
            GENERATION_ATTEMPT_BUDGET,
        )
        self.assertTrue(
            all(
                call.kwargs["attempt_budget"] == 1
                for call in generator.generate_text.await_args_list
            )
        )
        generation_rngs = [
            call.kwargs["rng"] for call in generator.generate_text.await_args_list
        ]
        self.assertTrue(all(rng is generation_rngs[0] for rng in generation_rngs))
        from app.presentation.fallback_phrases import (
            GENERATION_FAILED_PHRASES,
            LATE_NIGHT_FALLBACK_PHRASES,
        )

        msg.reply.assert_awaited_once()
        # See note above: the wall clock makes the late-night pool a valid
        # extension in the small hours, so accept either pool time-independently.
        self.assertIn(
            msg.reply.await_args.args[0],
            set(GENERATION_FAILED_PHRASES) | set(LATE_NIGHT_FALLBACK_PHRASES),
        )

    async def test_prefix_rejection_gets_extra_retry_budget(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь нормально")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(
            side_effect=[True, True, True, True, False]
        )
        generator = _traced_generator()
        # Empty strings after each verbatim candidate make the extension pass
        # fail, preserving the reject-and-retry flow this test exercises.
        generator.generate_text = AsyncMock(
            side_effect=[
                "один два три четыре",
                "",
                "один два три четыре",
                "",
                "один два три четыре",
                "",
                "один два три четыре",
                "",
                "новый ответ теперь",
            ]
            + [""] * 6
        )
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            fuzzy_context_casefold=False,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            context_start_affinity=3.0,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            recent_reply_penalty_strength=1.0,
            length_mode_weights=(0.25, 0.55, 0.2),
            intonation_profile_strength=0.0,
            length_context_adaptation=0.0,
            markov_order=3,
            enable_backoff=True,
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        # 10 attempt walks + 4 failed extension walks after each verbatim hit.
        self.assertEqual(generator.generate_text.await_count, 14)
        msg.reply.assert_awaited_once_with("новый ответ теперь")

    async def test_short_reply_skips_training_prefix_filter(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=True)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="Привет")
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            recent_short_replies={},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            fuzzy_context_casefold=False,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            context_start_affinity=3.0,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            recent_reply_penalty_strength=1.0,
            length_mode_weights=(0.25, 0.55, 0.2),
            intonation_profile_strength=0.0,
            length_context_adaptation=0.0,
            markov_order=3,
            enable_backoff=True,
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.is_verbatim_copy.assert_not_awaited()
        msg.reply.assert_awaited_once_with("Привет")
        self.assertEqual(list(state.recent_short_replies[msg.chat.id]), ["привет"])

    async def test_recent_short_reply_is_retried_instead_of_sent(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=["Привет", "Нормально"] + [""] * 8
        )
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            recent_short_replies={msg.chat.id: deque(["привет"], maxlen=5)},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            fuzzy_context_casefold=False,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            context_start_affinity=3.0,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.0,
            repetition_penalty_strength=1.0,
            recent_reply_penalty_strength=1.0,
            length_mode_weights=(0.25, 0.55, 0.2),
            intonation_profile_strength=0.0,
            length_context_adaptation=0.0,
            markov_order=3,
            enable_backoff=True,
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        self.assertEqual(generator.generate_text.await_count, 10)
        msg.reply.assert_awaited_once_with("Нормально")
        self.assertEqual(
            list(state.recent_short_replies[msg.chat.id]),
            ["привет", "нормально"],
        )

    async def test_sent_reply_is_recorded_for_full_anti_repeat(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            return_value="новый развёрнутый ответ бота."
        )
        state = self._reply_state(recent_replies={})

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("новый развёрнутый ответ бота.")
        self.assertEqual(
            list(state.recent_replies[msg.chat.id]),
            ["новый развёрнутый ответ бота"],
        )

    async def test_mood_enabled_tracks_state_and_passes_modifiers(self) -> None:
        from app.core.mood import MoodConfig, modifiers_for_mood
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        state = self._reply_state(recent_replies={})
        state.mood_enabled = True
        state.chat_mood = {}
        state.mood_modulation_strength = 1.0
        state.mood_config = MagicMock(
            return_value=MoodConfig(
                ewma_alpha=0.3,
                lively_rate_per_min=12.0,
                sleepy_rate_per_min=2.0,
                heated_intensity=0.4,
                max_rate_per_min=120.0,
            )
        )

        with (
            patch("app.services.reply_pipeline.ResponseGenerator") as response_gen_cls,
            patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"),
        ):
            response_gen_cls.return_value.generate = AsyncMock(return_value="ответ")
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        # Mood was tracked for the chat...
        self.assertIn(msg.chat.id, state.chat_mood)
        tracked = state.chat_mood[msg.chat.id]
        self.assertIn(tracked.mood, ("sleepy", "calm", "lively", "heated"))
        # ...and the matching modifiers were handed to the generator.
        _, kwargs = response_gen_cls.call_args
        self.assertEqual(
            kwargs["mood_modifiers"], modifiers_for_mood(tracked.mood, 1.0)
        )

    def _director_state(self, **overrides: object) -> MagicMock:
        from app.core.mood import MoodConfig

        values: dict[str, object] = {
            "reply_director_enabled": True,
            "mood_enabled": False,
            "chat_mood": {},
            "recent_reply_times": {},
            "mood_modulation_strength": 1.0,
            "mood_lively_rate_per_min": 12.0,
            "reply_probability_min": 0.02,
            "reply_probability_max": 0.30,
            "reply_burst_boost_sec": 180,
            "reply_burst_boost_mult": 2.0,
            "reply_burst_suppress_sec": 600,
            "reply_burst_suppress_mult": 0.5,
            "reply_max_per_hour": 20,
            "min_cooldown_sec": 0,
            "recent_replies": {},
        }
        values.update(overrides)
        state = self._reply_state(**values)
        state.mood_config = lambda: MoodConfig(
            ewma_alpha=0.3,
            lively_rate_per_min=12.0,
            sleepy_rate_per_min=2.0,
            heated_intensity=0.4,
            max_rate_per_min=120.0,
        )
        return state

    async def test_director_hourly_cap_blocks_unprompted_reply(self) -> None:
        from collections import deque

        from app.handlers.learning import on_text_message

        # Unprompted message (no alias, no reply-to) in a chat that already hit
        # its per-hour reply budget: the director must gate the reply, but the
        # message is still learned.
        msg = _fake_message(text="просто болтаю тут в чате")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        generator = _traced_generator()
        state = self._director_state(
            reply_max_per_hour=20,
            recent_reply_times={msg.chat.id: deque([10000.0] * 20)},
        )

        with (
            patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"),
            patch("app.handlers.learning.time.monotonic", return_value=10000.0),
            patch("app.services.reply_pipeline.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        generator.generate_text.assert_not_awaited()
        msg.reply.assert_not_awaited()
        learning_service.record_message.assert_awaited_once()

    async def test_director_allows_reply_when_momentum_clears_probability(self) -> None:
        from app.handlers.learning import on_text_message

        # random forced to 0.0 → any positive momentum-derived probability fires.
        msg = _fake_message(text="просто болтаю тут в чате")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._director_state(recent_reply_times={})

        with (
            patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"),
            patch("app.handlers.learning.time.monotonic", return_value=10000.0),
            patch("app.services.reply_pipeline.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("ответ бота готов")

    async def test_emojis_recorded_from_message(self) -> None:
        from app.handlers.learning import on_text_message

        # A message carrying emojis feeds the per-chat emoji stats even though the
        # word model drops them. Recorded regardless of whether a reply fires.
        msg = _fake_message(text="хаха 😂 огонь 🔥")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state(emoji_append_chance=0.15)

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.record_emojis.assert_awaited_once_with(
            msg.chat.id, {"😂": 1, "🔥": 1}
        )

    async def test_reply_appends_chat_emoji(self) -> None:
        from app.handlers.learning import on_text_message

        # With the channel certain (chance 1.0) and stats present, the reply ends
        # on a frequency-sampled emoji from this chat.
        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_emoji_stats = AsyncMock(return_value={"🍺": 5})
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._reply_state(recent_replies={}, emoji_append_chance=1.0)

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("ответ бота готов 🍺")

    async def test_learnable_message_records_hot_ngrams(self) -> None:
        from app.core.hot_ngrams import extract_content_ngrams
        from app.core.markov import tokenize
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="крутой бобёр пришёл")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state(hot_ngram_seed_chance=0.05)

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        expected = extract_content_ngrams(
            tokenize("крутой бобёр пришёл", normalize_lower=False)
        )
        learning_service.record_hot_ngrams.assert_awaited_once_with(
            msg.chat.id, expected
        )

    async def test_hot_ngram_recording_disabled_at_zero_chance(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="крутой бобёр пришёл")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="")
        state = self._reply_state()  # hot_ngram_seed_chance = 0.0 default

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.record_hot_ngrams.assert_not_awaited()

    async def test_unprompted_reply_seeded_on_roll(self) -> None:
        from app.handlers.learning import on_text_message

        # No mention; reply_probability 1.0 and the patched roll 0.0 win both
        # the reply gate and the seed gate.
        msg = _fake_message(text="обычное сообщение в чате")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_hot_ngrams = AsyncMock(
            return_value=[("крутой", "бобёр")]
        )
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._reply_state(
            reply_probability=1.0,
            hot_ngram_seed_chance=1.0,
            recent_replies={},
        )

        with (
            patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"),
            patch("app.services.reply_pipeline.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.get_hot_ngrams.assert_awaited_once_with(
            msg.chat.id,
            min_count=state.hot_ngram_min_count,
            recency_share=state.hot_ngram_recency_share,
            meme_ordering=state.markov_hot_ngram_meme_ordering,
        )
        first_call = generator.generate_text.await_args_list[0]
        self.assertEqual(first_call.kwargs["seed_tokens"], ["крутой", "бобёр"])
        msg.reply.assert_awaited_once()

    async def test_mention_reply_never_seeded(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._reply_state(
            hot_ngram_seed_chance=1.0,
            recent_replies={},
        )

        with (
            patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"),
            patch("app.services.reply_pipeline.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        learning_service.get_hot_ngrams.assert_not_awaited()
        first_call = generator.generate_text.await_args_list[0]
        self.assertIsNone(first_call.kwargs["seed_tokens"])
        msg.reply.assert_awaited_once()

    async def test_no_hot_ngrams_means_no_seed(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="обычное сообщение в чате")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=101)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_hot_ngrams = AsyncMock(return_value=[])
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="ответ бота готов")
        state = self._reply_state(
            reply_probability=1.0,
            hot_ngram_seed_chance=1.0,
            recent_replies={},
        )

        with (
            patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"),
            patch("app.services.reply_pipeline.random.random", return_value=0.0),
        ):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        first_call = generator.generate_text.await_args_list[0]
        self.assertIsNone(first_call.kwargs["seed_tokens"])
        msg.reply.assert_awaited_once()

    async def test_rare_event_false_start_sends_filler_then_reply(self) -> None:
        from app.core.reply_flavor import FALSE_START_FILLERS
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
        state = self._reply_state(
            recent_replies={},
            false_start_chance=1.0,
            rare_events_today={},
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once()
        self.assertIn(msg.reply.await_args.args[0], FALSE_START_FILLERS)
        msg.answer.assert_awaited_once_with("настоящий ответ бота")
        self.assertEqual(state.rare_events_today[msg.chat.id][1], 1)

    async def test_rare_event_respects_daily_cap(self) -> None:
        from datetime import UTC, datetime

        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
        state = self._reply_state(
            recent_replies={},
            false_start_chance=1.0,
            # UTC, как в хендлере: локальная дата около полуночи дала бы флаки.
            rare_events_today={
                msg.chat.id: (datetime.now(UTC).date().isoformat(), 3)
            },
            rare_event_daily_cap=3,
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
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
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="настоящий ответ бота")
        state = self._reply_state(recent_replies={})  # both chances 0.0 default

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg, learning_service, generator, state,
                "PepeEdtaBot", 777, frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("настоящий ответ бота")
        msg.answer.assert_not_awaited()

    async def test_recent_full_reply_is_retried_instead_of_sent(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь развёрнуто")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(
            side_effect=[
                "недавно отправленный полный ответ.",
                "совсем свежий полный ответ",
            ]
            + [""] * 8
        )
        state = self._reply_state(
            recent_replies={
                msg.chat.id: deque(
                    ["недавно отправленный полный ответ"], maxlen=20
                )
            },
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        msg.reply.assert_awaited_once_with("совсем свежий полный ответ")

    async def test_handler_increases_randomness_on_generation_retries(self) -> None:
        from app.handlers.learning import on_text_message

        msg = _fake_message(text="pepe ответь нормально")
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=100)
        learning_service.is_verbatim_copy = AsyncMock(
            side_effect=[True, True, False]
        )
        generator = _traced_generator()
        # Empty strings after each verbatim candidate make the extension pass
        # fail, preserving the reject-and-retry escalation this test measures.
        generator.generate_text = AsyncMock(
            side_effect=[
                "первый ответ заметно длиннее",
                "",
                "второй ответ заметно длиннее",
                "",
                "третий ответ заметно длиннее",
            ]
            + [""] * 7
        )
        state = _fake_state(
            normalize_lower=False,
            learned_messages={},
            min_tokens_for_model=10,
            min_cooldown_sec=999,
            last_reply_ts={},
            reply_probability=0.0,
            use_reply_context=False,
            fuzzy_context_casefold=False,
            reply_context_bias=1.8,
            reply_context_start_bias=2.2,
            context_start_affinity=3.0,
            max_reply_chars=280,
            max_reply_tokens=45,
            randomness_strength=0.5,
            repetition_penalty_strength=1.0,
            recent_reply_penalty_strength=1.0,
            length_mode_weights=(0.25, 0.55, 0.2),
            intonation_profile_strength=0.0,
            length_context_adaptation=0.0,
            markov_order=3,
            enable_backoff=True,
        )

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

        strengths = [
            call.kwargs["randomness_strength"]
            for call in generator.generate_text.await_args_list
            # Extension walks (max_tokens=10) run at base strength; only the
            # attempt walks carry the escalation this test measures.
            if call.kwargs.get("max_tokens") != 10
        ]
        self.assertEqual(strengths, sorted(strengths))
        self.assertEqual(strengths[0], 0.5)
        self.assertGreater(strengths[-1], strengths[0])
        msg.reply.assert_awaited_once_with("третий ответ заметно длиннее")


# ---------------------------------------------------------------------------
# learning.py — mention anti-flood gate (N1)
# ---------------------------------------------------------------------------

class TestMentionCooldownGate(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        mask_patcher = patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        )
        mask_patcher.start()
        self.addCleanup(mask_patcher.stop)

    def _state(self, **overrides: object) -> MagicMock:
        values: dict[str, object] = {
            "normalize_lower": False,
            "learned_messages": {},
            "recent_short_replies": {},
            "min_tokens_for_model": 10,
            "min_cooldown_sec": 0,
            "last_reply_ts": {},
            "reply_probability": 0.0,
            "use_reply_context": False,
            "fuzzy_context_casefold": False,
            "reply_context_bias": 1.8,
            "reply_context_start_bias": 2.2,
            "context_start_affinity": 3.0,
            "max_reply_chars": 280,
            "max_reply_tokens": 45,
            "randomness_strength": 0.0,
            "repetition_penalty_strength": 1.0,
            "recent_reply_penalty_strength": 1.0,
            "length_mode_weights": (0.25, 0.55, 0.2),
            "intonation_profile_strength": 0.0,
            "length_context_adaptation": 0.0,
            "markov_order": 3,
            "enable_backoff": True,
            "mention_cooldown_sec": 30,
        }
        values.update(overrides)
        return _fake_state(**values)

    def _services(self) -> tuple[AsyncMock, AsyncMock]:
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="сгенерированный ответ бота")
        return learning_service, generator

    async def _dispatch(self, msg: MagicMock, learning_service: AsyncMock,
                        generator: AsyncMock, state: MagicMock) -> None:
        from app.handlers.learning import on_text_message

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

    async def test_first_mention_replies_and_records_timestamp(self) -> None:
        learning_service, generator = self._services()
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()
        state.note_mention_reply.assert_called_once_with(
            msg.chat.id, msg.from_user.id, ANY
        )

    async def test_mention_within_cooldown_is_demoted_to_unprompted_path(self) -> None:
        import time as time_module

        learning_service, generator = self._services()
        state = self._state()
        msg = _fake_message(text="pepe расскажи ещё")
        state.last_mention_reply_ts = {
            (msg.chat.id, msg.from_user.id): time_module.monotonic()
        }

        await self._dispatch(msg, learning_service, generator, state)

        # Gated mention: no guaranteed reply (reply_probability=0 → silent),
        # but the message is still learned as usual.
        msg.reply.assert_not_awaited()
        state.note_mention_reply.assert_not_called()
        learning_service.record_message.assert_awaited_once()

    async def test_gate_disabled_keeps_guaranteed_mention_reply(self) -> None:
        import time as time_module

        learning_service, generator = self._services()
        state = self._state(mention_cooldown_sec=0)
        msg = _fake_message(text="pepe расскажи снова")
        state.last_mention_reply_ts = {
            (msg.chat.id, msg.from_user.id): time_module.monotonic()
        }

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()

    async def test_other_user_is_not_gated_by_someone_elses_mention(self) -> None:
        import time as time_module

        learning_service, generator = self._services()
        state = self._state()
        msg = _fake_message(text="pepe расскажи", user_id=2)
        state.last_mention_reply_ts = {
            (msg.chat.id, 1): time_module.monotonic()
        }

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()


class TestUserQuirks(unittest.IsolatedAsyncioTestCase):
    """L2: vocative quirks for regulars whose direct address got answered."""

    def setUp(self) -> None:
        mask_patcher = patch(
            "app.core.response_generator.mask_chat_id", return_value="chat"
        )
        mask_patcher.start()
        self.addCleanup(mask_patcher.stop)

    def _state(self, **overrides: object) -> MagicMock:
        values: dict[str, object] = {
            "normalize_lower": False,
            "learned_messages": {},
            "recent_short_replies": {},
            "min_tokens_for_model": 10,
            "min_cooldown_sec": 0,
            "last_reply_ts": {},
            "reply_probability": 0.0,
            "use_reply_context": False,
            "fuzzy_context_casefold": False,
            "reply_context_bias": 1.8,
            "reply_context_start_bias": 2.2,
            "context_start_affinity": 3.0,
            "max_reply_chars": 280,
            "max_reply_tokens": 45,
            "randomness_strength": 0.0,
            "repetition_penalty_strength": 1.0,
            "recent_reply_penalty_strength": 1.0,
            "length_mode_weights": (0.25, 0.55, 0.2),
            "intonation_profile_strength": 0.0,
            "length_context_adaptation": 0.0,
            "markov_order": 3,
            "enable_backoff": True,
            "recent_replies": {},
            # Chance 1.0 keeps the roll deterministic (random() < 1.0 always).
            "user_quirk_chance": 1.0,
            "user_quirk_min_interactions": 25,
            "user_quirk_name_share": 0.0,
        }
        values.update(overrides)
        return _fake_state(**values)

    def _services(
        self, *, interactions: int = 25
    ) -> tuple[AsyncMock, AsyncMock]:
        learning_service = AsyncMock()
        learning_service.get_token_volume = AsyncMock(return_value=100)
        learning_service.record_message = AsyncMock(return_value=102)
        learning_service.is_verbatim_copy = AsyncMock(return_value=False)
        learning_service.get_user_interaction_count = AsyncMock(
            return_value=interactions
        )
        generator = _traced_generator()
        generator.generate_text = AsyncMock(return_value="сгенерированный ответ бота")
        return learning_service, generator

    async def _dispatch(self, msg: MagicMock, learning_service: AsyncMock,
                        generator: AsyncMock, state: MagicMock) -> None:
        from app.handlers.learning import on_text_message

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                msg,
                learning_service,
                generator,
                state,
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
            )

    @staticmethod
    def _today_iso() -> str:
        from datetime import UTC, datetime

        # UTC, как в хендлере: локальная дата около полуночи дала бы флаки.
        return datetime.now(UTC).date().isoformat()

    async def test_regular_gets_vocative_as_separate_first_message(self) -> None:
        from app.presentation.fallback_phrases import USER_QUIRK_VOCATIVES

        learning_service, generator = self._services(interactions=25)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()
        self.assertIn(msg.reply.await_args.args[0], USER_QUIRK_VOCATIVES)
        msg.answer.assert_awaited_once_with("сгенерированный ответ бота")
        self.assertEqual(
            state.last_user_quirk_day,
            {(msg.chat.id, msg.from_user.id): self._today_iso()},
        )
        learning_service.record_user_interaction.assert_awaited_once_with(
            msg.chat.id, msg.from_user.id
        )

    async def test_below_threshold_sends_plain_reply(self) -> None:
        learning_service, generator = self._services(interactions=24)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once_with("сгенерированный ответ бота")
        msg.answer.assert_not_awaited()
        # Below threshold: no daily stamp, so a later regular can still quirk.
        self.assertEqual(state.last_user_quirk_day, {})

    async def test_same_user_same_day_is_suppressed(self) -> None:
        learning_service, generator = self._services(interactions=100)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")
        state.last_user_quirk_day = {
            (msg.chat.id, msg.from_user.id): self._today_iso()
        }

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once_with("сгенерированный ответ бота")
        msg.answer.assert_not_awaited()
        # The daily gate is checked before the chance roll and the DB read.
        learning_service.get_user_interaction_count.assert_not_awaited()

    async def test_unprompted_reply_is_never_quirked(self) -> None:
        learning_service, generator = self._services(interactions=100)
        state = self._state(reply_probability=1.0)
        msg = _fake_message(text="просто сообщение в чат без обращения")

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once_with("сгенерированный ответ бота")
        msg.answer.assert_not_awaited()
        learning_service.record_user_interaction.assert_not_awaited()
        learning_service.get_user_interaction_count.assert_not_awaited()

    async def test_not_enough_data_fallback_counts_but_never_quirks(self) -> None:
        learning_service, generator = self._services(interactions=100)
        learning_service.get_token_volume = AsyncMock(return_value=5)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        # The user did interact -> counter bumped; but a fallback answer never
        # carries a vocative.
        learning_service.record_user_interaction.assert_awaited_once_with(
            msg.chat.id, msg.from_user.id
        )
        learning_service.get_user_interaction_count.assert_not_awaited()
        msg.reply.assert_awaited_once()
        msg.answer.assert_not_awaited()
        self.assertEqual(state.last_user_quirk_day, {})

    async def test_generation_failed_fallback_counts_but_never_quirks(self) -> None:
        learning_service, generator = self._services(interactions=100)
        generator.generate_text = AsyncMock(return_value="")
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        learning_service.record_user_interaction.assert_awaited_once_with(
            msg.chat.id, msg.from_user.id
        )
        learning_service.get_user_interaction_count.assert_not_awaited()
        msg.reply.assert_awaited_once()
        msg.answer.assert_not_awaited()
        self.assertEqual(state.last_user_quirk_day, {})

    async def test_demoted_mention_neither_counts_nor_quirks(self) -> None:
        import time as time_module

        learning_service, generator = self._services(interactions=100)
        state = self._state(mention_cooldown_sec=30)
        msg = _fake_message(text="pepe расскажи ещё")
        state.last_mention_reply_ts = {
            (msg.chat.id, msg.from_user.id): time_module.monotonic()
        }

        await self._dispatch(msg, learning_service, generator, state)

        # Demoted to the unprompted path (reply_probability 0.0 -> silence):
        # regular status cannot be farmed faster than the mention cooldown.
        learning_service.record_user_interaction.assert_not_awaited()
        learning_service.get_user_interaction_count.assert_not_awaited()
        msg.reply.assert_not_awaited()

    async def test_quirked_reply_skips_rare_event_roll(self) -> None:
        from app.presentation.fallback_phrases import USER_QUIRK_VOCATIVES

        learning_service, generator = self._services(interactions=25)
        # false_start_chance 1.0 would otherwise always fire a filler.
        state = self._state(false_start_chance=1.0, rare_events_today={})
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        # One shape break per reply: vocative + reply, no filler, and the
        # rare-event daily budget is untouched.
        msg.reply.assert_awaited_once()
        self.assertIn(msg.reply.await_args.args[0], USER_QUIRK_VOCATIVES)
        msg.answer.assert_awaited_once_with("сгенерированный ответ бота")
        self.assertEqual(state.rare_events_today, {})

    async def test_name_share_uses_sanitized_first_name(self) -> None:
        learning_service, generator = self._services(interactions=25)
        state = self._state(user_quirk_name_share=1.0)
        msg = _fake_message(text="pepe расскажи что-нибудь")
        msg.from_user.first_name = "🔥Саня🔥"

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()
        vocative = msg.reply.await_args.args[0]
        self.assertTrue(
            vocative == "саня" or vocative.startswith("саня, "),
            vocative,
        )
        msg.answer.assert_awaited_once_with("сгенерированный ответ бота")

    async def test_name_share_unusable_name_falls_back_to_pool(self) -> None:
        from app.presentation.fallback_phrases import USER_QUIRK_VOCATIVES

        learning_service, generator = self._services(interactions=25)
        state = self._state(user_quirk_name_share=1.0)
        msg = _fake_message(text="pepe расскажи что-нибудь")
        msg.from_user.first_name = "🔥🔥🔥"

        await self._dispatch(msg, learning_service, generator, state)

        msg.reply.assert_awaited_once()
        self.assertIn(msg.reply.await_args.args[0], USER_QUIRK_VOCATIVES)

    async def test_zero_chance_disables_channel_entirely(self) -> None:
        learning_service, generator = self._services(interactions=100)
        state = self._state(user_quirk_chance=0.0)
        msg = _fake_message(text="pepe расскажи что-нибудь")

        await self._dispatch(msg, learning_service, generator, state)

        # Master gate: no counter writes, no reads at all.
        learning_service.record_user_interaction.assert_not_awaited()
        learning_service.get_user_interaction_count.assert_not_awaited()
        msg.reply.assert_awaited_once_with("сгенерированный ответ бота")

    async def test_bot_senders_never_feed_counters(self) -> None:
        # Anonymous admins post as GroupAnonymousBot (a bot account): the
        # handler's early is_bot return must keep them out of the counters.
        learning_service, generator = self._services(interactions=100)
        state = self._state()
        msg = _fake_message(text="pepe расскажи что-нибудь", is_bot=True)

        await self._dispatch(msg, learning_service, generator, state)

        learning_service.record_user_interaction.assert_not_awaited()
        msg.reply.assert_not_awaited()


class TestMaintenanceOwnerAlert(unittest.IsolatedAsyncioTestCase):
    """Сигнал владельцу о сбое обслуживания базы.

    Логи владельцу недоступны: без этого сигнала «счётчики перестали стареть»
    замечается через недели.
    """

    def _learning_service(self, alert: object | None) -> AsyncMock:
        service = AsyncMock()
        service.get_token_volume.return_value = 0
        service.run_due_maintenance.return_value = alert
        return service

    async def _run(
        self, service: AsyncMock, bot: object | None, settings: object | None
    ) -> None:
        from app.handlers.learning import on_text_message

        with patch("app.services.reply_pipeline.mask_chat_id", return_value="chat"):
            await on_text_message(
                _fake_message(text="сегодня хорошая погода в городе"),
                service,
                _traced_generator(),
                _fake_state(
                    normalize_lower=False,
                    learned_messages={},
                    min_tokens_for_model=1_000_000,
                ),
                "PepeEdtaBot",
                777,
                frozenset({"pepe", "пепе"}),
                _pivo_stub(),
                bot=bot,  # type: ignore[arg-type]
                settings=settings,  # type: ignore[arg-type]
            )

    async def test_alert_is_delivered_to_the_owner(self) -> None:
        from app.services.learning_service import MaintenanceAlert

        alert = MaintenanceAlert(
            recovered=False, failing_for_sec=7200.0, reason="database is locked"
        )
        bot = AsyncMock()

        await self._run(
            self._learning_service(alert), bot, SimpleNamespace(owner_id=555)
        )

        bot.send_message.assert_awaited_once()
        chat_id, text = bot.send_message.await_args.args
        self.assertEqual(chat_id, 555)
        self.assertIn("database is locked", text)
        self.assertIn("2 ч", text)

    async def test_recovery_is_delivered_too(self) -> None:
        from app.services.learning_service import MaintenanceAlert

        alert = MaintenanceAlert(recovered=True, failing_for_sec=3600.0, reason=None)
        bot = AsyncMock()

        await self._run(
            self._learning_service(alert), bot, SimpleNamespace(owner_id=555)
        )

        text = bot.send_message.await_args.args[1]
        self.assertIn("снова проходит", text)

    async def test_no_owner_means_no_message(self) -> None:
        from app.services.learning_service import MaintenanceAlert

        alert = MaintenanceAlert(recovered=False, failing_for_sec=7200.0, reason=None)
        bot = AsyncMock()

        await self._run(
            self._learning_service(alert), bot, SimpleNamespace(owner_id=None)
        )

        bot.send_message.assert_not_awaited()

    async def test_delivery_failure_does_not_break_the_message_path(self) -> None:
        from app.services.learning_service import MaintenanceAlert

        alert = MaintenanceAlert(recovered=False, failing_for_sec=7200.0, reason=None)
        bot = AsyncMock()
        # Самый вероятный случай — владелец не начинал диалог с ботом.
        bot.send_message.side_effect = RuntimeError("chat not found")
        service = self._learning_service(alert)

        with self.assertLogs("chat_markov", level="WARNING"):
            await self._run(service, bot, SimpleNamespace(owner_id=555))

        service.record_message.assert_awaited_once()

    async def test_quiet_maintenance_sends_nothing(self) -> None:
        bot = AsyncMock()

        await self._run(
            self._learning_service(None), bot, SimpleNamespace(owner_id=555)
        )

        bot.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Состав команд, доступных в личке (O13, spec private-chat-diagnostics)
# ---------------------------------------------------------------------------

class TestPrivateChatCommandSurface(unittest.TestCase):
    """Пиннит, какие команды отвечают в личном чате.

    Личка — диагностический канал владельца без доступа к логам: /ping и
    /help для всех, /db_snapshot — owner-only снимок БД; всё остальное живёт
    под GROUP_ONLY. Новая команда обязана попасть сюда явным решением, а не
    умолчанием регистрации.

    Проверено мутацией (2026-09-01): снятие GROUP_ONLY со /stats роняет тест
    со словом "stats" в сообщении; навешивание GROUP_ONLY на /ping — со словом
    "ping".
    """

    # Полный инвентарь: команда -> отвечает ли в личке.
    PINNED: dict[str, bool] = {
        "ping": True,
        "help": True,
        # owner-only и private-only: доступность в личке не значит «всем» —
        # состав фильтров закреплён отдельным тестом в TestAdminHandlers.
        "db_snapshot": True,
        "stats": False,
        "config": False,
        "set": False,
        "setprob": False,
        "quirk_stats": False,
        "clear": False,
        "pivo": False,
        "pivo_on": False,
        "pivo_off": False,
        "pivo_check": False,
        "pivo_privacy": False,
    }

    @staticmethod
    def _observed() -> dict[str, bool]:
        from aiogram.filters import Command

        from app.filters import GROUP_ONLY
        from app.handlers import admin, common, errors, learning, pivo

        observed: dict[str, bool] = {}
        routers = [common.router, admin.router, pivo.router,
                   learning.router, errors.router]
        for router in routers:
            for handler in router.message.handlers:
                commands: set[str] = set()
                group_only = False
                for filter_object in handler.filters or ():
                    if getattr(filter_object, "magic", None) is GROUP_ONLY:
                        group_only = True
                    if isinstance(filter_object.callback, Command):
                        commands.update(
                            str(c) for c in filter_object.callback.commands
                        )
                # Команда с несколькими регистрациями (например /set с формой
                # отказа) доступна в личке, если хотя бы одна регистрация без
                # GROUP_ONLY.
                for command in commands:
                    observed[command] = observed.get(command, False) or (
                        not group_only
                    )
        return observed

    def test_command_inventory_matches_pinned_private_surface(self) -> None:
        observed = self._observed()
        diverged = sorted(
            set(self.PINNED) ^ set(observed)
            | {c for c in set(self.PINNED) & set(observed)
               if self.PINNED[c] != observed[c]}
        )
        self.assertEqual(
            observed,
            self.PINNED,
            "Состав команд разошёлся с решением O13 "
            f"(разошлись: {', '.join(diverged)}). Личка — только /ping и "
            "/help; новая команда добавляется в PINNED явным решением.",
        )
