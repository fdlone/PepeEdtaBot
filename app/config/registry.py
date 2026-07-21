"""
Single source of truth for runtime-mutable configuration fields.

Each FieldSpec describes one configuration field that lives both in
``Settings`` (loaded from environment) and ``RuntimeState`` (mutable at
runtime via ``/set``). The same parser is reused by ``load_settings`` and
``apply_runtime_setting``, eliminating triple duplication of validation
logic across ``settings.py``, ``runtime_state.py`` and ``runtime_config.py``.

Each parser is paired with a human-readable ``hint`` describing its accepted
range (see ``Domain``), so ``/help`` / ``/set help`` can advertise the valid
values from the very same source of truth that validates them.

Fields that are NOT runtime-mutable (BOT_TOKEN, OWNER_ID, DB_PATH,
PIVO secrets, LOG_LEVEL) intentionally live only in ``settings.py``.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _num(value: float) -> str:
    """Render a numeric bound without a trailing ``.0`` for readable hints."""
    return format(value, "g")


@dataclass(frozen=True, slots=True)
class Domain:
    """A value parser paired with a human-readable description of the values
    it accepts.

    Keeping the range next to the parser means the ``/help`` and ``/set help``
    hints are generated from the exact source of truth that validates the
    value, so the advertised range and the enforced range cannot drift apart.
    """

    parse: Callable[[str], Any]
    hint: str

    def __call__(self, value: str) -> Any:
        return self.parse(value)


def _parse_bool(value: str) -> bool:
    val = value.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _bool() -> Domain:
    return Domain(_parse_bool, "true/false")


def _int_in_range(min_v: int, max_v: int) -> Domain:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed < min_v or parsed > max_v:
            raise ValueError(f"value must be in range [{min_v}..{max_v}]")
        return parsed
    return Domain(parse, f"{min_v}..{max_v}")


def _int_min(min_v: int) -> Domain:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed < min_v:
            raise ValueError(f"value must be >= {min_v}")
        return parsed
    return Domain(parse, f">= {min_v}")


def _int_in_set(allowed: set[int]) -> Domain:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed not in allowed:
            raise ValueError(f"value must be one of {sorted(allowed)}")
        return parsed
    return Domain(parse, "/".join(str(v) for v in sorted(allowed)))


def _float_in_range(min_v: float, max_v: float) -> Domain:
    def parse(value: str) -> float:
        parsed = float(value)
        # NaN must be rejected explicitly: every comparison against it is
        # False, so the range check below would wave it straight through.
        # A NaN knob does not fail loudly either — it poisons whatever it
        # touches (`min(1.0, nan)` returns 1.0, turning an invalid
        # probability into the maximum; a NaN score makes `max()` degenerate
        # to "first candidate wins"). `inf` is already caught by the range
        # check, but reject it here too so the guard reads as one rule.
        if not math.isfinite(parsed):
            raise ValueError("value must be a finite number")
        if parsed < min_v or parsed > max_v:
            raise ValueError(f"value must be in range [{min_v}..{max_v}]")
        return parsed
    return Domain(parse, f"{_num(min_v)}..{_num(max_v)}")


def _parse_length_mode_weights(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(
            "value must be three comma-separated weights: short,medium,long"
        )
    try:
        short_w, medium_w, long_w = (float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"invalid weight in {value!r}") from exc
    # Same reason as in `_float_in_range`, with a sharper edge: the
    # non-negative and positive-sum checks below are both False for NaN and
    # for +inf, so either value reaches `random.choices`, which then raises
    # "Total of weights must be finite" on *every* generation — the bot goes
    # mute chat-wide while /set reported success.
    if not all(math.isfinite(weight) for weight in (short_w, medium_w, long_w)):
        raise ValueError("weights must be finite numbers")
    if min(short_w, medium_w, long_w) < 0.0:
        raise ValueError("weights must be non-negative")
    if short_w + medium_w + long_w <= 0.0:
        raise ValueError("at least one weight must be positive")
    return short_w, medium_w, long_w


def _length_weights() -> Domain:
    return Domain(_parse_length_mode_weights, "short,medium,long")


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Metadata for a single runtime-mutable configuration field."""

    name: str
    env_var: str
    default: str
    parse: Domain


RUNTIME_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("reply_probability", "REPLY_PROBABILITY", "0.08",
              _float_in_range(0.0, 1.0)),
    FieldSpec("min_cooldown_sec", "MIN_COOLDOWN_SEC", "45", _int_min(0)),
    FieldSpec("min_tokens_for_model", "MIN_TOKENS_FOR_MODEL", "200", _int_min(0)),
    FieldSpec("max_reply_chars", "MAX_REPLY_CHARS", "280",
              _int_in_range(20, 4000)),
    FieldSpec("max_reply_tokens", "MAX_REPLY_TOKENS", "45",
              _int_in_range(1, 300)),
    FieldSpec("normalize_lower", "NORMALIZE_LOWER", "true", _bool()),
    FieldSpec(
        "auto_capitalize_replies",
        "AUTO_CAPITALIZE_REPLIES",
        "false",
        _bool(),
    ),
    FieldSpec("typing_min_ms", "TYPING_MIN_MS", "350", _int_min(0)),
    FieldSpec("typing_max_ms", "TYPING_MAX_MS", "1100", _int_min(0)),
    FieldSpec("typing_per_char_ms", "TYPING_PER_CHAR_MS", "12",
              _int_in_range(0, 200)),
    FieldSpec("randomness_strength", "RANDOMNESS_STRENGTH", "2.0",
              _float_in_range(0.0, 3.0)),
    # 0.7 (2026-07-09): three eval_prod sweeps found the temperature nearly
    # inert -- 1.3/0.7/0.3 land within 1pp of each other on every metric,
    # because SELECTION_SCORE_MARGIN culls the pool before the softmax has
    # anything to spread over. Set to the middle of the range that was tested;
    # tune the margin, not this, if the wrong candidate keeps winning.
    FieldSpec("candidate_selection_temperature", "CANDIDATE_SELECTION_TEMPERATURE",
              "0.7", _float_in_range(0.0, 3.0)),
    FieldSpec("reply_flavor_strength", "REPLY_FLAVOR_STRENGTH", "1.0",
              _float_in_range(0.0, 2.0)),
    # M3 emoji channel: chance to append a frequency-sampled emoji (from this
    # chat's own usage) to a reply. 0 disables. Suppressed after a "?" ending;
    # a heated mood scales it up.
    FieldSpec("emoji_append_chance", "EMOJI_APPEND_CHANCE", "0.15",
              _float_in_range(0.0, 1.0)),
    FieldSpec("repetition_penalty_strength", "REPETITION_PENALTY_STRENGTH", "1.0",
              _float_in_range(0.0, 3.0)),
    # 0.5 chosen by eval sweep (2026-07-02): in the case-preserved profile
    # strength 1.0 collapsed context_token_overlap 0.21->0.09; 0.5 keeps it at
    # 0.14 with the same distinct-1/2 gain and ~1% empty-result rate.
    FieldSpec("recent_reply_penalty_strength", "RECENT_REPLY_PENALTY_STRENGTH",
              "0.5", _float_in_range(0.0, 3.0)),
    # Score penalty for candidates that replay training messages verbatim:
    # strength × severity, where severity ramps linearly from 0 at a corpus
    # 4-gram share of VERBATIM_TOLERATED_SHARE (0.6, candidate_scorer.py) to 1
    # at share 1.0 — shares below the threshold are free (recombination is the
    # point of the generator). 0 disables (and skips the index read). Замена
    # бинарного гейта,
    # который цитаты с добавленной точкой раньше проходили насквозь.
    # 1.5 (2026-07-09): raising the context weight makes on-topic continuations
    # win, and continuing from an exact context state retraces the original
    # message -- so quoting rose with it. 1.5 pulls verbatim_run_ratio back to
    # the old 0.075 and near-verbatim replies to zero; past 1.5 the penalty
    # starts hitting the on-topic candidates themselves.
    FieldSpec("verbatim_penalty_strength", "VERBATIM_PENALTY_STRENGTH",
              "1.5", _float_in_range(0.0, 3.0)),
    FieldSpec("length_mode_weights", "LENGTH_MODE_WEIGHTS", "0.25,0.55,0.2",
              _length_weights()),
    # Length mirroring: tilt of the short/long weights above toward the length
    # of the message being answered (candidate_scorer.context_length_weights).
    # The weights alone are blind to the interlocutor, so a short answer to a
    # short question kept landing in a long mode and losing natural_length for
    # being right. 0 disables (weights stay as configured).
    # 1.0 (2026-07-14): 5-seed x 200-gen prod-copy sweep of 0/0.5/1/2/3 —
    # length_mirror_gap (long-in mean minus short-in mean) 2.0 -> 4.7 -> 5.3 ->
    # 5.9 -> 7.0, with context_anchored_win_rate, distinct_2 and the empty rate
    # flat throughout and verbatim_run_ratio drifting down (0.186 -> 0.175).
    # 1.0 takes most of the gain (replies to a short message 10.0 -> 8.6 tokens,
    # to a long one 12.0 -> 13.9) while the medium band stays put; past it the
    # ramp starts shortening mid-length replies too for little extra spread.
    # P4 intonation profile: how far the reply's length-mode weights and the
    # ending-flavor probabilities are blended toward the chat's own observed
    # habits (short/medium/long shares and final-punctuation shares over the
    # messages retention window). 0 disables entirely (no profile reads);
    # 1.0 replaces the configured distributions with the chat's. Applied
    # only once the chat has >=200 profiled messages.
    # Kept at 0 (2026-07-21, B3 deferred). Measured alone the knob looks free
    # (every metric flat, novelty +0.017), but verified *jointly* with the
    # B1/B2 package on the prod copy it is the one knob that costs anchoring:
    # with it ctx_anchored -0.017 and replies shorten 11.0 -> 10.1 tokens
    # (-9%), without it ctx_anchored is flat (-0.004) and length unchanged.
    # The shortening is the knob working as designed (blending toward the
    # chat's own 41%-short habits), so this is a taste trade, not a bug — it
    # just deserves its own observation window instead of riding along in a
    # metrics-driven tuning package. Enable as a separate, isolated change.
    FieldSpec("intonation_profile_strength", "INTONATION_PROFILE_STRENGTH",
              "0", _float_in_range(0.0, 1.0)),
    FieldSpec("length_context_adaptation", "LENGTH_CONTEXT_ADAPTATION", "1.0",
              _float_in_range(0.0, 3.0)),
    FieldSpec("markov_order", "MARKOV_ORDER", "3", _int_in_set({2, 3})),
    # Backoff bottoms out at order 2: the order-1 chain was removed entirely
    # (2026-07-12) after eval_prod showed order-1 walks are word salad — the
    # 2026-07-09 default already forbade them and nothing regressed.
    FieldSpec("enable_backoff", "ENABLE_BACKOFF", "true", _bool()),
    # M4 topic drift: probability per generation step (only after the reply has
    # ≥5 tokens — JUMP_MIN_GENERATED_TOKENS, markov.py — and only on the
    # order-3 walk) of jumping to a new learned sentence start, splicing a
    # connective ("..., кстати ...") so the shift reads as a deliberate aside.
    # 0 disables (the pre-M4 behaviour).
    # 0.12 (2026-07-06): tripled from 0.04 and the in-code minimum reply length
    # for a jump dropped 9 -> 5 tokens — деталь «агрессивного хаоса»; связки
    # («кстати», «короче») сохраняют читабельность сдвига темы.
    FieldSpec("markov_jump_probability", "MARKOV_JUMP_PROBABILITY", "0.12",
              _float_in_range(0.0, 1.0)),
    # Context+chaos: multiplier on markov_jump_probability applied only to
    # walks that started from a contextual anchor (start_source context /
    # hidden_context). The anchor voices the topic; without a jump the chain
    # then retraces what followed the matched window in the corpus — which is
    # why context anchoring and verbatim quoting rise and fall together. The
    # boost breaks exactly that retrace while global walks keep the base
    # probability. 1.0 keeps jumps uniform (pre-knob behaviour).
    # 2.0 (2026-07-21): the "context+chaos" package ran live via /set from
    # 2026-07-16 and its observation window (O1) closed clean, so the value is
    # baked in here — a restart used to silently drop back to 1.0. Sweep pass B
    # confirms the knob is monotonic (1.0 -> ctx_anchored -0.019, 4.0 -> +0.036);
    # 2.0 sits mid-range, 4.0 also pushes connective_reply_rate toward the tic.
    FieldSpec("context_jump_boost", "CONTEXT_JUMP_BOOST", "2.0",
              _float_in_range(1.0, 10.0)),
    # Context+chaos: corpus 4-gram share at/above which a *passing* candidate
    # is treated as a near-quote and gets the отсебятина extension (connective
    # + fresh global walk) — the same splice that full verbatim copies already
    # receive. Generalizes that channel from "exact copy of a training
    # message" to "quote with a word changed". The extension only replaces the
    # candidate when the combined text passes the same gates again. 0 disables
    # (legacy behaviour: only exact copies are extended).
    # 0.8 (2026-07-21): part of the same live "context+chaos" package, baked in
    # after O1 closed. Sweep pass B: the knob does NOT move context anchoring
    # (flat ~0.62 across 0/0.6/0.8 once extended winners are attributed to their
    # base — the earlier 0.6 "dip" was an eval artifact); it trades latency
    # (0 -> 181ms vs 230ms) for fewer bare near-quotes.
    FieldSpec("verbatim_extension_share", "VERBATIM_EXTENSION_SHARE", "0.8",
              _float_in_range(0.0, 1.0)),
    # Artificial branching valve: probability per generation step of taking
    # the transition from the wider order-2 pool even though the order-3 pool
    # has a continuation. ~98% of order-3 states on the prod corpus have
    # exactly one continuation (the walk replays the source message), while
    # order-2 states average 1.27 options (8.7 on frequent states) — this
    # knob buys per-step divergence at a bounded coherence cost (order 2 is
    # the established quality floor; order 1 was removed as word salad).
    # Diverts only when order-2 actually offers more options than order-3,
    # and only when backoff is enabled. 0 disables (pre-knob behaviour).
    # 0.75 (2026-07-21, B1): live at 0.5 via /set since 2026-07-16; raised after
    # O1 closed. Sweep pass B (10 seeds x 400) is the rare knob that moves three
    # metrics the right way at once vs 0.5: ctx_anchored +0.019, novelty +0.014,
    # and connective_reply_rate -0.026 (i.e. AWAY from the ~0.8 "connective tic"
    # threshold). Monotonic across 0/0.25/0.5/0.75 with no verbatim/length cost.
    FieldSpec("order_mix_probability", "ORDER_MIX_PROBABILITY", "0.75",
              _float_in_range(0.0, 1.0)),
    # Slot mutations ("отсебятина" channel): chance per accepted candidate to
    # also field a mutated copy — one content word swapped for a frequent chat
    # word agreeing morphologically (pymorphy3: POS + case/number/gender/
    # tense; ending+length prefilter). The copy competes in scoring against
    # its original, so the existing quality/verbatim signals arbitrate. New
    # word pairs are born exactly here — the walk itself cannot leave the
    # corpus (2026-07-15 chain-structure audit). 0 disables (no frequency
    # dictionary reads, no mutation rolls).
    # 0.15 (2026-07-21, B2): enabled after O1 closed. Sweep pass B (10 seeds x
    # 400): novelty +0.011 (significant) with anchoring/connective/verbatim flat
    # — the exact profile wanted from this channel. 0.3 buys more novelty
    # (+0.025) but dents distinct_1 and raises the morphological-garbage share
    # (manual review of ~150 winners put it at 8-12% already at 0.15, which is
    # the accepted ceiling); the residue is verb valency and broken idioms, not
    # morphology, so the pymorphy3 guard cannot catch it.
    FieldSpec("slot_mutation_probability", "SLOT_MUTATION_PROBABILITY", "0.15",
              _float_in_range(0.0, 1.0)),
    # L1 running jokes: chance to seed an *unprompted* reply from a currently
    # hot n-gram (a phrase the chat picked up in the last ~7 days). 0 disables
    # the whole channel (no recording, no reads) — same gate pattern as
    # emoji_append_chance. Mention replies are never seeded.
    # 0.25 (2026-07-12): at 0.05 the channel fired ~once a week in a live chat
    # (dialogue-simulation audit: 1-2 lookups per 900 messages); ~every 4th
    # unprompted reply now rolls for a seed, actual fires still gated by a hot
    # n-gram existing in the window.
    FieldSpec("hot_ngram_seed_chance", "HOT_NGRAM_SEED_CHANCE", "0.25",
              _float_in_range(0.0, 1.0)),
    # Minimum window occurrences before an n-gram can be considered hot.
    FieldSpec("hot_ngram_min_count", "HOT_NGRAM_MIN_COUNT", "3", _int_min(1)),
    # Hot = window count / all-time count >= this share; 0.5 means at least
    # half of all recorded occurrences happened inside the decay window.
    FieldSpec("hot_ngram_recency_share", "HOT_NGRAM_RECENCY_SHARE", "0.5",
              _float_in_range(0.0, 1.0)),
    # L3 rare events: chance that a generated reply becomes a "shape break" —
    # one-word verdict, ALL-CAPS, or a double message. Uniform among the three.
    # 0 disables the roll. Capped per chat per day by rare_event_daily_cap.
    # 0.03 (2026-07-12): 0.005 meant one event per ~200 replies (once in 1-2
    # weeks live); ~daily now, the daily cap still bounds the worst case.
    FieldSpec("rare_event_chance", "RARE_EVENT_CHANCE", "0.03",
              _float_in_range(0.0, 1.0)),
    # L3 false starts: chance to send a short filler, keep "typing", then the
    # real reply as a second message. 0 disables. Shares the daily cap.
    # 0.05 (2026-07-12): modest bump — false starts stay the most frequent
    # event but must not eat the shared daily budget of the other three kinds.
    FieldSpec("false_start_chance", "FALSE_START_CHANCE", "0.05",
              _float_in_range(0.0, 1.0)),
    # Combined per-chat daily budget for rare events + false starts.
    FieldSpec("rare_event_daily_cap", "RARE_EVENT_DAILY_CAP", "3", _int_min(0)),
    # L2 user quirks: chance to precede a generated answer to a "regular"'s
    # direct address with a short vocative message ("опять ты"). 0 disables
    # the whole channel — no interaction-counter writes, no reads (same gate
    # pattern as emoji_append_chance / hot_ngram_seed_chance). Additionally
    # capped in code at one quirk per (chat, user) per UTC day.
    FieldSpec("user_quirk_chance", "USER_QUIRK_CHANCE", "0.1",
              _float_in_range(0.0, 1.0)),
    # Answered-mention count (decayed over ~30 days, see
    # CHAT_USER_INTERACTION_DECAY_DAYS) at which a user counts as a regular.
    FieldSpec("user_quirk_min_interactions", "USER_QUIRK_MIN_INTERACTIONS",
              "25", _int_min(1)),
    # L2.1: share of fired quirks whose vocative addresses the regular by
    # first name (taken live from the incoming update, sanitized, never
    # stored or logged) instead of the anonymous pool phrase. The name is
    # bare half the time, fused with a pool phrase otherwise. 0 disables —
    # the pool-only legacy behaviour.
    FieldSpec("user_quirk_name_share", "USER_QUIRK_NAME_SHARE", "0",
              _float_in_range(0.0, 1.0)),
    FieldSpec("use_reply_context", "USE_REPLY_CONTEXT", "true", _bool()),
    FieldSpec(
        "fuzzy_context_casefold",
        "FUZZY_CONTEXT_CASEFOLD",
        "true",
        _bool(),
    ),
    FieldSpec("reply_context_max_tokens", "REPLY_CONTEXT_MAX_TOKENS", "12",
              _int_min(2)),
    FieldSpec("reply_context_bias", "REPLY_CONTEXT_BIAS", "1.8",
              _float_in_range(1.0, 4.0)),
    FieldSpec("reply_context_start_bias", "REPLY_CONTEXT_START_BIAS", "2.2",
              _float_in_range(1.0, 4.0)),
    # Context affinity of GLOBAL start sampling: each learned start's weight is
    # multiplied by affinity^(shared stems with the context's informative
    # tokens). Answers live in starts, not in continuations: for «кто гнойный
    # пидор?» the contextual anchor can only retrace what followed the question
    # in the corpus, while the wanted reply «слава гнойный пидор ...» is a
    # high-count learned START that plain sampling almost never draws among
    # ~3000. 1.0 disables.
    # 6.0 (2026-07-10): probe sweep on «кто гнойный пидор» — assertion-style
    # answers 1%/4%/8%/13% at affinity 1/3/6/10, with every 5-seed eval metric
    # flat up to 6.0; at 10.0 distinct_2 dips 0.904 -> 0.888 (the same starts
    # get re-picked). 6.0 is the knee: the gain is real, the cost is not yet.
    FieldSpec("context_start_affinity", "CONTEXT_START_AFFINITY", "6.0",
              _float_in_range(1.0, 10.0)),
    # Anchor segmentation («прыжок наоборот»): share of visible contextual
    # starts whose anchor is deferred — the walk begins from a GLOBAL start and
    # the anchor's emission tokens are spliced in later (connective + chain
    # continues from the anchor state), so the context surfaces mid- or
    # end-reply instead of always opening it. The splice position is uniform
    # over the token budget; if the global walk dead-ends first, the anchor is
    # spliced at the dead end. Shares the one-jump-per-reply budget with M4
    # drift: a pending anchor suppresses global jumps and its splice consumes
    # the jump slot. 0 disables (anchor always opens the reply, pre-knob
    # behaviour).
    # 0.25 (2026-07-16): prod-copy A/B, 5 seeds x 200 on the live
    # context+chaos package — anchored win rate 0.33 -> 0.58 and novelty
    # 0.60 -> 0.63 at half the cost: the scorer over-selects spliced
    # candidates, so 0.5 already pushes the connective share to 0.75 (the
    # #88 "connective tic" started reading at ~0.8) and 1.0 to 0.87.
    FieldSpec("context_anchor_splice_probability",
              "CONTEXT_ANCHOR_SPLICE_PROBABILITY", "0.25",
              _float_in_range(0.0, 1.0)),
    FieldSpec("reply_context_only_for_replies", "REPLY_CONTEXT_ONLY_FOR_REPLIES",
              "true", _bool()),
    FieldSpec("reply_context_include_current_message",
              "REPLY_CONTEXT_INCLUDE_CURRENT_MESSAGE", "true", _bool()),
    # S2: how many recently used /pivo template indices to remember per pool per
    # chat and exclude from the next pick. 0 disables anti-repeat.
    FieldSpec("pivo_recent_pool_window", "PIVO_RECENT_POOL_WINDOW", "5",
              _int_in_range(0, 50)),
    # S4: probability of swapping the /pivo closing line for a time-aware variant
    # (late-night / Friday / Monday) when one applies. 0 disables temporal flavor.
    FieldSpec("pivo_temporal_flavor_chance", "PIVO_TEMPORAL_FLAVOR_CHANCE", "0.5",
              _float_in_range(0.0, 1.0)),
    # M1 chat mood: a hidden per-chat state (sleepy/calm/lively/heated) that drifts
    # the bot's behaviour with conversation rhythm.
    FieldSpec("mood_enabled", "MOOD_ENABLED", "true", _bool()),
    # Scales every mood-driven knob deviation: 0 tracks mood but changes nothing,
    # 1 uses the built-in table, higher exaggerates.
    FieldSpec("mood_modulation_strength", "MOOD_MODULATION_STRENGTH", "1.0",
              _float_in_range(0.0, 3.0)),
    # EWMA smoothing for the mood signals (higher = reacts faster).
    FieldSpec("mood_ewma_alpha", "MOOD_EWMA_ALPHA", "0.3",
              _float_in_range(0.01, 1.0)),
    # Message-rate thresholds (msg/min) separating sleepy < calm < lively.
    FieldSpec("mood_lively_rate_per_min", "MOOD_LIVELY_RATE_PER_MIN", "12.0",
              _float_in_range(0.0, 600.0)),
    FieldSpec("mood_sleepy_rate_per_min", "MOOD_SLEEPY_RATE_PER_MIN", "2.0",
              _float_in_range(0.0, 600.0)),
    # Smoothed emphatic-intensity (!/?/caps) at/above which a chat reads as heated.
    FieldSpec("mood_heated_intensity", "MOOD_HEATED_INTENSITY", "0.4",
              _float_in_range(0.0, 1.0)),
    # Escalation chains: smoothed share of bot-addressed messages
    # (mention_ewma) at/above which the chat reads as heated — a series of
    # direct mentions "winds the bot up" (randomness/flavor up, punchier
    # length mix via the heated modifiers) even at modest pace and
    # punctuation. At alpha 0.3 three back-to-back mentions reach ~0.66, so
    # 0.6 ≈ "three in a row". 0 disables (mentions keep feeding only the M2
    # momentum, pre-knob behaviour).
    # 0.6 (2026-07-16): replay of 1000 real chat messages — fires on exactly
    # three consecutive mentions, defuses after one foreign message
    # (0.657*0.7=0.46), and alternating mentions plateau at ~0.59 so mixed
    # dialogue never fires; 0.4 fires on two (too eager), 0.8 needs five
    # (dead). Zero background mood flips at any threshold.
    FieldSpec("mood_mention_heated_share", "MOOD_MENTION_HEATED_SHARE", "0.6",
              _float_in_range(0.0, 1.0)),
    # Clamp for the instantaneous message rate so bursts cannot spike the EWMA.
    FieldSpec("mood_max_rate_per_min", "MOOD_MAX_RATE_PER_MIN", "120.0",
              _float_in_range(1.0, 6000.0)),
    # M2 reply-probability director: master toggle. When off, the flat
    # ``reply_probability`` (× mood multiplier) governs unprompted replies exactly
    # as before; when on, conversation momentum maps into the [min, max] band below.
    FieldSpec("reply_director_enabled", "REPLY_DIRECTOR_ENABLED", "true",
              _bool()),
    # Momentum band: a quiet chat sits near the min, a busy/addressed one climbs to
    # the max. The legacy flat ``reply_probability`` (0.08) is roughly the midpoint.
    FieldSpec("reply_probability_min", "REPLY_PROBABILITY_MIN", "0.02",
              _float_in_range(0.0, 1.0)),
    FieldSpec("reply_probability_max", "REPLY_PROBABILITY_MAX", "0.30",
              _float_in_range(0.0, 1.0)),
    # Burst rhythm: right after the bot replies its probability is boosted for
    # ``boost_sec`` (join the exchange), then suppressed for the following
    # ``suppress_sec`` (withdraw). ``min_cooldown_sec`` remains the hard floor.
    FieldSpec("reply_burst_boost_sec", "REPLY_BURST_BOOST_SEC", "180",
              _int_min(0)),
    FieldSpec("reply_burst_boost_mult", "REPLY_BURST_BOOST_MULT", "2.0",
              _float_in_range(0.0, 10.0)),
    FieldSpec("reply_burst_suppress_sec", "REPLY_BURST_SUPPRESS_SEC", "600",
              _int_min(0)),
    FieldSpec("reply_burst_suppress_mult", "REPLY_BURST_SUPPRESS_MULT", "0.5",
              _float_in_range(0.0, 10.0)),
    # Safety cap on unprompted replies per rolling hour per chat (mentions always
    # answered, never counted against the gate). 0 disables the cap.
    FieldSpec("reply_max_per_hour", "REPLY_MAX_PER_HOUR", "20",
              _int_in_range(0, 1000)),
    # Anti-flood gate for mention-triggered replies: a direct address bypasses
    # the chat cooldown and the hourly cap by design, so without this gate one
    # user can force a generation+reply per message. Per-user per-chat seconds
    # between mention-triggered replies; a gated mention falls back to the
    # unprompted-reply path. 0 disables the gate (legacy behaviour).
    FieldSpec("mention_cooldown_sec", "MENTION_COOLDOWN_SEC", "5",
              _int_in_range(0, 3600)),
)


_SPECS_BY_NAME: dict[str, FieldSpec] = {spec.name: spec for spec in RUNTIME_FIELDS}


def get_spec(name: str) -> FieldSpec | None:
    return _SPECS_BY_NAME.get(name)


def field_hint(name: str) -> str | None:
    """Human-readable accepted-range hint for ``name`` (e.g. ``"0..1"``),
    or ``None`` for an unknown field."""
    spec = _SPECS_BY_NAME.get(name)
    return spec.parse.hint if spec is not None else None


def runtime_field_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in RUNTIME_FIELDS)


def validate_cross_fields(obj: Any) -> None:
    """Cross-field validation shared by load_settings and /set.

    Raises ValueError if any invariant between fields is violated.
    """
    if obj.typing_min_ms > obj.typing_max_ms:
        raise ValueError("TYPING_MIN_MS must be <= TYPING_MAX_MS")
    if obj.mood_sleepy_rate_per_min >= obj.mood_lively_rate_per_min:
        raise ValueError(
            "MOOD_SLEEPY_RATE_PER_MIN must be lower than MOOD_LIVELY_RATE_PER_MIN"
        )
    if obj.reply_probability_min > obj.reply_probability_max:
        raise ValueError(
            "REPLY_PROBABILITY_MIN must be <= REPLY_PROBABILITY_MAX"
        )


def try_apply(state: Any, name: str, value: str) -> None:
    """Parse, validate and apply a runtime field on ``state``.

    Cross-field invariants are checked on a shallow copy first; the real
    state is mutated only if the resulting combination is valid.
    Raises ``KeyError`` for unknown names and ``ValueError`` for invalid
    values (either by spec.parse or by cross-field validation).
    """
    spec = _SPECS_BY_NAME.get(name)
    if spec is None:
        raise KeyError(name)
    parsed = spec.parse(value)
    candidate = copy.copy(state)
    setattr(candidate, name, parsed)
    validate_cross_fields(candidate)
    setattr(state, name, parsed)
