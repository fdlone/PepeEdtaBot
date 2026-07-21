"""Direct unit tests for the runtime config registry (audit T1).

The registry is the single source of truth for runtime-mutable fields: its
parsers, cross-field invariants and ``try_apply`` transactional semantics were
previously only exercised indirectly through settings/handlers. These tests
pin them down directly.
"""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
