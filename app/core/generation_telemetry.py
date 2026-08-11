"""Process-lifetime generation telemetry (Markov 2.0R Phase 1, M2R-010/020/030).

Counters only — numbers and labels, no text, no chat identifiers (the privacy
contract of the generation-telemetry spec). Owned by ``MarkovGenerator``,
surfaced through ``/stats``; reset on restart by construction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GenerationTelemetry:
    generations: int = 0
    entropy_bits_sum: float = 0.0
    normalized_entropy_sum: float = 0.0
    branching_sum: float = 0.0
    applied_temperature_sum: float = 0.0
    diagnostic_steps: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    shadow_order4_eligible: int = 0
    shadow_order4_selected: int = 0

    def note_cache(self, *, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def note_generation(
        self,
        *,
        entropy_bits_sum: float,
        normalized_entropy_sum: float,
        branching_sum: float,
        steps: int,
        applied_temperature_sum: float = 0.0,
    ) -> None:
        self.generations += 1
        self.entropy_bits_sum += entropy_bits_sum
        self.normalized_entropy_sum += normalized_entropy_sum
        self.branching_sum += branching_sum
        self.applied_temperature_sum += applied_temperature_sum
        self.diagnostic_steps += steps

    def note_shadow(self, *, eligible: int, selected: int) -> None:
        self.shadow_order4_eligible += eligible
        self.shadow_order4_selected += selected

    def snapshot(self) -> dict[str, float | int | None]:
        """Aggregates for ``/stats``; ``None`` where no data exists yet."""
        steps = self.diagnostic_steps
        lookups = self.cache_hits + self.cache_misses
        eligible = self.shadow_order4_eligible
        return {
            "generations": self.generations,
            "mean_entropy_bits": self.entropy_bits_sum / steps if steps else None,
            "mean_normalized_entropy": (
                self.normalized_entropy_sum / steps if steps else None
            ),
            "mean_branching": self.branching_sum / steps if steps else None,
            "mean_applied_temperature": (
                self.applied_temperature_sum / steps if steps else None
            ),
            "cache_hit_rate": self.cache_hits / lookups if lookups else None,
            "shadow_order4_eligible": eligible,
            "shadow_order4_selected_share": (
                self.shadow_order4_selected / eligible if eligible else None
            ),
        }
