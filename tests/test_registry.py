"""Direct unit tests for the runtime config registry (audit T1).

The registry is the single source of truth for runtime-mutable fields: its
parsers, cross-field invariants and ``try_apply`` transactional semantics were
previously only exercised indirectly through settings/handlers. These tests
pin them down directly.
"""
from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from app.config import registry
from app.config.registry import (
    RUNTIME_FIELDS,
    FieldSpec,
    get_spec,
    runtime_field_names,
    try_apply,
    validate_cross_fields,
)


def _parses(spec: FieldSpec, value: str) -> bool:
    """True when ``spec`` accepts ``value`` (used to assert that it does not)."""
    try:
        spec.parse(value)
    except Exception:
        return False
    return True


def _make_state(**overrides: object) -> SimpleNamespace:
    """A state object carrying every field touched by validate_cross_fields."""
    base = {
        "typing_min_ms": 350,
        "typing_max_ms": 1100,
        "markov_order": 3,
        "reply_context_max_tokens": 12,
        "mood_sleepy_rate_per_min": 2.0,
        "mood_lively_rate_per_min": 12.0,
        "reply_probability_min": 0.02,
        "reply_probability_max": 0.30,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestParsers(unittest.TestCase):
    def test_parse_bool_truthy_and_falsy(self) -> None:
        for v in ("1", "true", "TRUE", "yes", "on"):
            self.assertTrue(registry._parse_bool(v))
        for v in ("0", "false", "No", "off"):
            self.assertFalse(registry._parse_bool(v))

    def test_parse_bool_rejects_garbage(self) -> None:
        with self.assertRaises(ValueError):
            registry._parse_bool("maybe")

    def test_int_in_range_bounds(self) -> None:
        parse = registry._int_in_range(1, 10)
        self.assertEqual(parse("1"), 1)
        self.assertEqual(parse("10"), 10)
        with self.assertRaises(ValueError):
            parse("0")
        with self.assertRaises(ValueError):
            parse("11")

    def test_int_min(self) -> None:
        parse = registry._int_min(2)
        self.assertEqual(parse("2"), 2)
        with self.assertRaises(ValueError):
            parse("1")

    def test_int_in_set(self) -> None:
        parse = registry._int_in_set({2, 3})
        self.assertEqual(parse("2"), 2)
        with self.assertRaises(ValueError):
            parse("4")

    def test_float_in_range(self) -> None:
        parse = registry._float_in_range(0.0, 1.0)
        self.assertEqual(parse("0.5"), 0.5)
        with self.assertRaises(ValueError):
            parse("1.5")

    def test_float_in_range_rejects_non_finite(self) -> None:
        # Every comparison against NaN is False, so a bare range check waves it
        # through. Downstream it does not fail loudly either: `min(1.0, nan)`
        # is 1.0, which silently promotes an invalid probability to the maximum.
        parse = registry._float_in_range(0.0, 1.0)
        for value in ("nan", "NaN", "inf", "-inf", "Infinity"):
            with self.assertRaises(ValueError):
                parse(value)

    def test_parse_length_mode_weights_valid(self) -> None:
        self.assertEqual(
            registry._parse_length_mode_weights("0.25, 0.55, 0.2"),
            (0.25, 0.55, 0.2),
        )
        self.assertEqual(
            registry._parse_length_mode_weights("1,0,0"), (1.0, 0.0, 0.0)
        )

    def test_parse_length_mode_weights_rejects_garbage(self) -> None:
        for value in ("0.5,0.5", "1,2,3,4", "a,b,c", "-1,1,1", "0,0,0"):
            with self.assertRaises(ValueError):
                registry._parse_length_mode_weights(value)

    def test_parse_length_mode_weights_rejects_non_finite(self) -> None:
        # Both guards below ("non-negative", "sum positive") are False for NaN
        # and for +inf, so either one reaches random.choices, which raises
        # "Total of weights must be finite" on every single generation.
        for value in ("nan,1,1", "1,nan,1", "1,1,nan", "inf,1,1", "1,inf,1"):
            with self.assertRaises(ValueError):
                registry._parse_length_mode_weights(value)


class TestNumericKnobsAreBoundedBothWays(unittest.TestCase):
    """Ceilings for the knobs that used to be declared with a floor only."""

    # (field, ceiling) — the value one past the ceiling must be rejected.
    BOUNDED = (
        ("typing_max_ms", 15000),
        ("min_cooldown_sec", 3600),
        ("min_tokens_for_model", 1000000),
        ("reply_burst_boost_sec", 3600),
        ("reply_burst_suppress_sec", 3600),
    )

    def test_value_at_ceiling_is_accepted(self) -> None:
        for name, ceiling in self.BOUNDED:
            with self.subTest(name=name):
                spec = get_spec(name)
                assert spec is not None
                self.assertEqual(spec.parse(str(ceiling)), ceiling)

    def test_value_above_ceiling_is_rejected(self) -> None:
        for name, ceiling in self.BOUNDED:
            with self.subTest(name=name):
                spec = get_spec(name)
                assert spec is not None
                self.assertFalse(_parses(spec, str(ceiling + 1)))

    def test_rejection_names_the_allowed_range(self) -> None:
        # The refusal has to tell the operator what is allowed — that text is
        # what /set echoes back.
        for name, _ in self.BOUNDED:
            with self.subTest(name=name):
                spec = get_spec(name)
                assert spec is not None
                self.assertIn("..", spec.parse.hint)

    # Knobs that still accept an arbitrarily large value. This revision was
    # deliberately scoped to the five the 2026-07-21 review named (see
    # openspec/changes/add-knob-upper-bounds), so the rest are recorded here
    # rather than fixed silently — and a *new* knob without a ceiling still
    # fails the guard below. Tracked in docs/OPEN.md.
    KNOWN_UNBOUNDED = {
        # Bounded transitively: validate_cross_fields requires
        # typing_min_ms <= typing_max_ms, and that one is capped at 15000.
        "typing_min_ms",
        "hot_ngram_min_count",
        "rare_event_daily_cap",
        "user_quirk_min_interactions",
        "reply_context_max_tokens",
    }
    # Ceilings legitimately above the probe below.
    KNOWN_HIGH = {"min_tokens_for_model", "max_reply_chars"}

    def test_no_numeric_knob_is_left_without_a_ceiling(self) -> None:
        probe = "100000000"
        unbounded = {
            spec.name
            for spec in RUNTIME_FIELDS
            if spec.name not in self.KNOWN_HIGH and _parses(spec, probe)
        }
        self.assertEqual(unbounded - self.KNOWN_UNBOUNDED, set())

    def test_known_unbounded_list_does_not_go_stale(self) -> None:
        # If one of these gets a ceiling later, this test fails and the entry
        # must be removed — the list must never outlive the gap it records.
        probe = "100000000"
        still_unbounded = {
            name for name in self.KNOWN_UNBOUNDED if _parses(get_spec(name), probe)  # type: ignore[arg-type]
        }
        self.assertEqual(still_unbounded, self.KNOWN_UNBOUNDED)

    def test_typing_pause_ceiling_keeps_configured_max_above_hard_cap(self) -> None:
        # The deliberate rule "a configured max outranks TYPING_HARD_CAP_MS"
        # must stay reachable through a legal config.
        from app.handlers._helpers import TYPING_HARD_CAP_MS, compute_typing_delay_ms

        spec = get_spec("typing_max_ms")
        assert spec is not None
        configured = spec.parse("15000")
        self.assertGreater(configured, TYPING_HARD_CAP_MS)
        delay = compute_typing_delay_ms(0, configured, configured, 0, rng=random.Random(1))
        self.assertEqual(delay, configured)


class TestNoKnobAcceptsNonFinite(unittest.TestCase):
    """Registry-wide net: no numeric field may accept a non-finite value.

    Pins the whole surface rather than the two parsers, so a future field that
    brings its own parser cannot quietly reopen the hole.
    """

    def test_no_spec_accepts_non_finite_scalar(self) -> None:
        accepted = [
            (spec.name, value)
            for spec in RUNTIME_FIELDS
            for value in ("nan", "NaN", "inf", "-inf", "Infinity")
            if _parses(spec, value)
        ]
        self.assertEqual(accepted, [])

    def test_no_spec_accepts_non_finite_triplet(self) -> None:
        accepted = [
            (spec.name, value)
            for spec in RUNTIME_FIELDS
            for value in ("nan,1,1", "inf,1,1", "1,1,nan")
            if _parses(spec, value)
        ]
        self.assertEqual(accepted, [])


class TestSpecLookup(unittest.TestCase):
    def test_get_spec_known_and_unknown(self) -> None:
        self.assertIsNotNone(get_spec("markov_order"))
        self.assertIsNone(get_spec("does_not_exist"))

    def test_runtime_field_names_matches_specs(self) -> None:
        self.assertEqual(
            runtime_field_names(), tuple(s.name for s in RUNTIME_FIELDS)
        )

    def test_field_names_are_unique(self) -> None:
        names = [s.name for s in RUNTIME_FIELDS]
        self.assertEqual(len(names), len(set(names)))


class TestValidateCrossFields(unittest.TestCase):
    def test_valid_state_passes(self) -> None:
        validate_cross_fields(_make_state())  # no raise

    def test_typing_min_must_not_exceed_max(self) -> None:
        with self.assertRaises(ValueError):
            validate_cross_fields(_make_state(typing_min_ms=2000))


class TestTryApply(unittest.TestCase):
    def test_unknown_key_raises_keyerror(self) -> None:
        with self.assertRaises(KeyError):
            try_apply(_make_state(), "nope", "1")

    def test_invalid_value_raises_valueerror_and_leaves_state(self) -> None:
        state = _make_state(markov_order=3)
        with self.assertRaises(ValueError):
            try_apply(state, "markov_order", "5")  # not in {2, 3}
        self.assertEqual(state.markov_order, 3)

    def test_cross_field_violation_does_not_mutate_state(self) -> None:
        # typing_min_ms=500 with max=1500 is valid, but lowering the max below
        # the min must be rejected without touching the state.
        state = _make_state(typing_min_ms=500, typing_max_ms=1500)
        with self.assertRaises(ValueError):
            try_apply(state, "typing_max_ms", "300")
        self.assertEqual(state.typing_max_ms, 1500)

    def test_valid_apply_mutates_state(self) -> None:
        state = _make_state(markov_order=3)
        try_apply(state, "markov_order", "2")
        self.assertEqual(state.markov_order, 2)

    def test_pivo_mention_by_id_rollback_applies_to_next_call(self) -> None:
        """O2: аварийный откат должен работать через /set, без рестарта."""
        from app.domain.pivo import (
            MENTION_BY_ID,
            MENTION_BY_USERNAME,
            PivoMember,
            PivoSecurity,
            build_pivo_mention,
        )

        security = PivoSecurity("hmac-secret-value", "encryption-secret-value")
        member = PivoMember(
            encrypted_user_id=security.encrypt_value("100"),
            encrypted_username=security.encrypt_value("PepeUser"),
            encrypted_display_name=security.encrypt_value("Pepe"),
        )
        # Дефолт реестра — включено; ручка отката должна уметь выключить.
        self.assertEqual(get_spec("pivo_mention_by_id").default, "true")
        state = _make_state(pivo_mention_by_id=True)
        before = build_pivo_mention(
            member, security, mention_by_id=state.pivo_mention_by_id
        )
        self.assertEqual(before.path, MENTION_BY_ID)

        try_apply(state, "pivo_mention_by_id", "false")

        after = build_pivo_mention(
            member, security, mention_by_id=state.pivo_mention_by_id
        )
        self.assertEqual(after.path, MENTION_BY_USERNAME)
        self.assertEqual(after.text, "@PepeUser")


if __name__ == "__main__":
    unittest.main()
