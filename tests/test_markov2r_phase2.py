"""Markov 2.0R Phase 2: entropy-aware sampling and the branching-aware target.

The load-bearing test is neutrality: at gain 0 the mapping must return the
input power through an early return, because the phase's whole contract is
that the shipped default is byte-identical to Markov 1.x. Everything else here
checks that a non-zero gain moves the temperature in the direction the design
claims, and that no legal knob combination can produce a degenerate weight.
"""
from __future__ import annotations

import math
import random
import unittest

from hypothesis import given, settings
from hypothesis import strategies as st

from app.core.generation_telemetry import GenerationTelemetry
from app.core.markov import (
    _MIN_SAMPLING_TEMPERATURE,
    EntropySampling,
    _DiagnosticsAccumulator,
    _step_power,
    pool_diagnostics,
    weighted_next_choice,
)
from app.core.response_generator import CANDIDATE_TARGET, branching_aware_target
from app.presentation.bot_messages import format_stats_message

# Registry bounds (app/config/registry.py) — "any legal knob values" below
# means these, not arbitrary floats.
LEGAL_GAIN = st.floats(min_value=-2.0, max_value=2.0)
LEGAL_PIVOT = st.floats(min_value=0.0, max_value=1.0)
LEGAL_TEMP = st.floats(min_value=0.05, max_value=50.0)
# next_power spans [0.24, 0.72] for randomness_strength in [0, 3]; the
# exploration floor can push it to 0.02.
LEGAL_POWER = st.floats(min_value=0.02, max_value=1.0)


class TestTemperatureMapping(unittest.TestCase):
    def test_zero_gain_returns_the_input_unchanged(self) -> None:
        neutral = EntropySampling(gain=0.0, temp_min=5.0, temp_max=6.0)
        # Deliberately with clamps that would move any temperature: the early
        # return must fire before the clamp, not after it.
        for power in (0.02, 0.15, 0.4, 0.72):
            for entropy in (0.0, 0.5, 1.0):
                self.assertEqual(neutral.power_for(power, entropy), power)

    def test_pivot_is_the_fixed_point(self) -> None:
        sampling = EntropySampling(gain=0.8, pivot=0.37)
        power = 0.5
        self.assertAlmostEqual(sampling.power_for(power, 0.37), power, places=12)

    def test_positive_gain_flattens_high_entropy_pools(self) -> None:
        sampling = EntropySampling(gain=0.6, pivot=0.5)
        base = 0.5
        # Higher temperature = lower power = flatter weights.
        self.assertLess(sampling.power_for(base, 0.9), base)
        self.assertGreater(sampling.power_for(base, 0.1), base)

    def test_negative_gain_reverses_the_direction(self) -> None:
        sampling = EntropySampling(gain=-0.6, pivot=0.5)
        base = 0.5
        self.assertGreater(sampling.power_for(base, 0.9), base)
        self.assertLess(sampling.power_for(base, 0.1), base)

    def test_clamp_binds_at_both_ends(self) -> None:
        low = EntropySampling(gain=2.0, pivot=0.0, temp_min=0.5, temp_max=2.5)
        # T_base = 1/0.25 = 4.0 is already above temp_max, so the clamp pins it.
        self.assertAlmostEqual(low.power_for(0.25, 1.0), 1.0 / 2.5, places=12)
        high = EntropySampling(gain=-2.0, pivot=0.0, temp_min=3.0, temp_max=9.0)
        self.assertAlmostEqual(high.power_for(0.25, 1.0), 1.0 / 3.0, places=12)

    def test_inverted_clamp_bounds_are_ordered_not_obeyed(self) -> None:
        """Two individually valid /set calls can leave min above max."""
        inverted = EntropySampling(gain=0.5, pivot=0.5, temp_min=9.0, temp_max=2.0)
        ordered = EntropySampling(gain=0.5, pivot=0.5, temp_min=2.0, temp_max=9.0)
        self.assertEqual(
            inverted.power_for(0.4, 0.8), ordered.power_for(0.4, 0.8)
        )

    def test_degenerate_pool_entropy_is_handled(self) -> None:
        _, normalized, branching, _ = pool_diagnostics([7])
        self.assertEqual((normalized, branching), (0.0, 1))
        sampling = EntropySampling(gain=0.6, pivot=0.5)
        self.assertGreater(sampling.power_for(0.5, normalized), 0.0)

    def test_large_negative_gain_cannot_divide_by_zero(self) -> None:
        """gain * (H - pivot) <= -1 drives T to zero before the floor."""
        sampling = EntropySampling(gain=-2.0, pivot=0.0, temp_min=0.05, temp_max=50.0)
        power = sampling.power_for(0.5, 1.0)
        self.assertTrue(math.isfinite(power))
        self.assertGreater(power, 0.0)
        self.assertLessEqual(power, 1.0 / _MIN_SAMPLING_TEMPERATURE)


class TestTemperatureProperties(unittest.TestCase):
    @settings(max_examples=200, deadline=None)
    @given(
        power=LEGAL_POWER,
        entropy=st.floats(min_value=0.0, max_value=1.0),
        gain=LEGAL_GAIN,
        pivot=LEGAL_PIVOT,
        temp_a=LEGAL_TEMP,
        temp_b=LEGAL_TEMP,
    )
    def test_result_is_always_a_usable_power(
        self,
        power: float,
        entropy: float,
        gain: float,
        pivot: float,
        temp_a: float,
        temp_b: float,
    ) -> None:
        result = EntropySampling(
            gain=gain, pivot=pivot, temp_min=temp_a, temp_max=temp_b
        ).power_for(power, entropy)
        self.assertTrue(math.isfinite(result))
        self.assertGreater(result, 0.0)

    @settings(max_examples=200, deadline=None)
    @given(
        counts=st.lists(st.integers(min_value=1, max_value=5000), min_size=1, max_size=40),
        power=LEGAL_POWER,
        entropy=st.floats(min_value=0.0, max_value=1.0),
        gain=LEGAL_GAIN,
        pivot=LEGAL_PIVOT,
    )
    def test_weights_stay_finite_and_someone_keeps_mass(
        self,
        counts: list[int],
        power: float,
        entropy: float,
        gain: float,
        pivot: float,
    ) -> None:
        """TZ §19 invariant: no NaN/inf weights, and the pool never goes dead."""
        adjusted = EntropySampling(gain=gain, pivot=pivot).power_for(power, entropy)
        weights = [max(count, 1) ** adjusted for count in counts]
        for weight in weights:
            self.assertTrue(math.isfinite(weight))
            self.assertGreaterEqual(weight, 0.0)
        self.assertGreater(max(weights), 0.0)

    @settings(max_examples=100, deadline=None)
    @given(
        power=LEGAL_POWER,
        gain=st.floats(min_value=0.05, max_value=2.0),
        pivot=LEGAL_PIVOT,
        low=st.floats(min_value=0.0, max_value=1.0),
        high=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_monotonic_in_entropy_for_a_fixed_sign(
        self, power: float, gain: float, pivot: float, low: float, high: float
    ) -> None:
        """More entropy never yields a sharper distribution at positive gain.

        The direction the design claims must be the direction the code
        produces; the clamp may flatten the curve but must not invert it.
        """
        if low > high:
            low, high = high, low
        sampling = EntropySampling(gain=gain, pivot=pivot)
        self.assertLessEqual(
            sampling.power_for(power, high),
            sampling.power_for(power, low) + 1e-12,
        )


class TestSamplingStaysInsideThePool(unittest.TestCase):
    POOL = [("а", 100), ("б", 10), ("в", 1)]

    def test_choice_is_always_a_pool_member(self) -> None:
        """Entropy reweights an already-legal pool; it never admits a token."""
        tokens = {token for token, _ in self.POOL}
        for gain in (-2.0, -0.6, 0.0, 0.6, 2.0):
            adjusted = EntropySampling(gain=gain, pivot=0.5).power_for(0.5, 0.8)
            for seed in range(40):
                choice = weighted_next_choice(
                    self.POOL, 0.0, adjusted, random.Random(seed)
                )
                self.assertIn(choice, tokens)

    def test_flatter_power_spreads_the_winners(self) -> None:
        """The mapping's effect is visible in outcomes, not only in the number."""
        sharp = self._winner_spread(power=2.0)
        flat = self._winner_spread(power=0.05)
        self.assertLess(sharp, flat)

    def _winner_spread(self, *, power: float) -> int:
        winners = {
            weighted_next_choice(self.POOL, 0.0, power, random.Random(seed))
            for seed in range(200)
        }
        return len(winners)


class TestAppliedTemperatureIsObservable(unittest.TestCase):
    """Spec: the temperature actually applied has to be readable, or the knob
    is invisible in the live chat."""

    def test_accumulator_records_the_temperature_it_sampled_at(self) -> None:
        diagnostics = _DiagnosticsAccumulator()
        pool = [("а", 9), ("б", 1)]
        neutral = _step_power(diagnostics, pool, 0.4, EntropySampling())
        self.assertEqual(neutral, 0.4)
        self.assertAlmostEqual(diagnostics.applied_temperature_sum, 2.5, places=12)
        self.assertEqual(diagnostics.steps, 1)

    def test_non_zero_gain_shows_up_as_a_different_temperature(self) -> None:
        neutral_diag = _DiagnosticsAccumulator()
        tuned_diag = _DiagnosticsAccumulator()
        pool = [("а", 5), ("б", 5), ("в", 5), ("г", 5)]  # maximal entropy
        _step_power(neutral_diag, pool, 0.4, EntropySampling())
        _step_power(tuned_diag, pool, 0.4, EntropySampling(gain=0.6, pivot=0.5))
        self.assertGreater(
            tuned_diag.applied_temperature_sum, neutral_diag.applied_temperature_sum
        )

    def test_telemetry_publishes_the_mean(self) -> None:
        telemetry = GenerationTelemetry()
        telemetry.note_generation(
            entropy_bits_sum=2.0,
            normalized_entropy_sum=1.0,
            branching_sum=6.0,
            applied_temperature_sum=5.0,
            steps=2,
        )
        self.assertAlmostEqual(
            telemetry.snapshot()["mean_applied_temperature"], 2.5, places=12
        )

    def test_stats_shows_the_temperature(self) -> None:
        telemetry = GenerationTelemetry()
        telemetry.note_generation(
            entropy_bits_sum=2.0,
            normalized_entropy_sum=1.0,
            branching_sum=6.0,
            applied_temperature_sum=5.0,
            steps=2,
        )
        message = format_stats_message(
            {"volume": "1 000 слов"}, telemetry=telemetry.snapshot()
        )
        self.assertIn("температура шага: 2.50", message)

    def test_nothing_measured_publishes_nothing(self) -> None:
        self.assertIsNone(
            GenerationTelemetry().snapshot()["mean_applied_temperature"]
        )


class TestBranchingAwareTarget(unittest.TestCase):
    def test_disabled_keeps_the_previous_constant(self) -> None:
        for samples in ([], [1.0], [1.0, 1.0, 1.0], [12.0]):
            self.assertEqual(
                branching_aware_target(
                    CANDIDATE_TARGET, samples, degenerate_max=0.0, floor=2
                ),
                CANDIDATE_TARGET,
            )

    def test_no_samples_yet_keeps_the_full_target(self) -> None:
        self.assertEqual(
            branching_aware_target(CANDIDATE_TARGET, [], degenerate_max=1.5, floor=2),
            CANDIDATE_TARGET,
        )

    def test_degenerate_chain_drops_to_the_floor(self) -> None:
        self.assertEqual(
            branching_aware_target(
                CANDIDATE_TARGET, [1.0, 1.2], degenerate_max=1.5, floor=2
            ),
            2,
        )

    def test_wide_chain_keeps_the_full_target(self) -> None:
        self.assertEqual(
            branching_aware_target(
                CANDIDATE_TARGET, [1.0, 8.0], degenerate_max=1.5, floor=2
            ),
            CANDIDATE_TARGET,
        )

    def test_floor_never_exceeds_the_base_target(self) -> None:
        self.assertEqual(
            branching_aware_target(1, [1.0], degenerate_max=2.0, floor=5), 1
        )

    def test_result_is_always_at_least_one(self) -> None:
        """An early stop must never ask for zero candidates — that would mean
        no reply at all, which is not what "fewer candidates" may cost."""
        for floor in range(1, 6):
            for base in range(1, CANDIDATE_TARGET + 1):
                self.assertGreaterEqual(
                    branching_aware_target(
                        base, [1.0], degenerate_max=3.0, floor=floor
                    ),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
