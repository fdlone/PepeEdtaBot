# Proposal: markov2r-phase2-entropy-sampling

## Why

The chain already measures its own uncertainty and then ignores it. Phase 1
(M2R-010) computes entropy, branching factor and confidence for **every** pool
the walk samples from (`markov.py:1757`, `1786`) and spends them only on
telemetry. Sampling itself still runs at one global temperature derived from
`randomness_strength`: the same `weight = cnt ** frequency_power` is applied to
a pool with a single continuation and to a pool with forty. A flat temperature
is precisely what makes a confident (near-degenerate) state replay its source
message verbatim while a genuinely open state never uses the room it has.
ADR-003 accepted uncertainty as the generation signal; TZ §6 fixes the mapping.

This is also the first 2.0R phase that changes behavior people can hear in the
chat, so it ships with the discipline the protocol demands: an ablation
configuration, thresholds pre-registered before the numbers are looked at, and
a `GAIN = 0` identity contract that keeps the door open.

## What Changes

- **M2R-100 — Entropy → temperature (TZ §6).** Per walk step, the pool's
  normalized entropy modulates the sampling temperature:
  `T = T_base · (1 + GAIN · (H_norm − H_pivot))`, clamped to `[T_min, T_max]`.
  The mapping lands on the mechanism that already exists — the frequency power
  (`weight = max(cnt,1) ** power`) *is* an inverse temperature, so `power = 1/T`
  and `randomness_strength` keeps its role as the scale of `T_base` (audit
  §"Состояние, порядки, вероятности" confirms the correspondence). Entropy
  never overrides the hard gates; it only reweights an already-legal pool.
- **M2R-110 — Branching-aware candidate count.** The best-of-N target
  (`CANDIDATE_TARGET = 5` inside a budget of 10) stops being a constant and
  becomes a function of the branching the generator actually observed. On a
  degenerate chain, extra attempts produce near-duplicates and buy nothing but
  latency; on a wide-branching chain, more candidates give the scorer something
  to choose between. Expected direction of the latency effect is **downward**,
  and that is part of the acceptance check, not an assumption.
- **Knobs (TZ §18 names), all runtime-settable via `/set`:**
  `markov_entropy_enabled`, `markov_entropy_temp_gain`, plus the pivot and
  clamp bounds; `markov_branching_candidates_enabled`. Every knob off ⇒ the
  generator is byte-identical to the frozen 1.x baseline.
- **Identity contract.** `GAIN = 0` (or the knob off) SHALL reproduce 1.x
  output bit-for-bit, proven with `tools/generation_hash.py`, not argued. The
  entropy path consumes no random draws, so the RNG stream is untouched.
- **Eval (doc 05 §2).** `C1` becomes available in `tools/eval/matrix.yaml`,
  split into two arms so the two knobs attribute separately (temperature only;
  temperature + branching). Phase 2 gate thresholds are added to
  `eval_thresholds.yaml` **before** the run, per the pre-registration rule.
  Report goes to `docs/eval_reports/`.
- **Default decision inside this change.** Merge with `GAIN = 0`, run the
  offline grid on the prod copy, then: gate passes ⇒ the calibrated `GAIN`
  becomes the registry default in this same change, with the numbers cited from
  the report; gate fails ⇒ the default stays 0, and the phase closes with a
  documented negative result instead of a dead knob.
- **Riding along (no functional content):** archive the completed
  `report-pivo-mentions-to-owner` change (all 16 tasks done) so it does not need
  a PR of its own. Explicitly out of scope of everything above.

### Non-goals

- The direction and strength of `GAIN` are **not** decided here. ADR open
  question #2 ("which entropy → temperature mapping works better") is answered
  by the eval grid, and both signs are on the grid: amplifying (confident states
  sharpen further) and damping (confident states get loosened to break the
  replay) are opposite bets, and the data picks.
- Start-pool selection stays untouched. Phase 1 instruments walk steps only;
  extending entropy to start selection without diagnostics there would be an
  unmeasured change.
- No schema change, no new dependency, no temporal layer (that is Phase 3).

## Capabilities

### New Capabilities

- `generation-entropy-sampling`: how the distribution's own uncertainty is
  allowed to modulate sampling — the temperature mapping and its clamps, the
  branching-aware candidate target, the neutrality contract at `GAIN = 0`, the
  boundary against the hard gates, and the requirement that the phase's default
  is decided by a pre-registered ablation gate rather than by taste.

### Modified Capabilities

- `generation-eval`: the pre-registered-thresholds requirement currently
  enumerates the Phase 5/6/7 gates; the Phase 2 gate joins them (copy must not
  rise, distinct-2/3 must rise, p95 within budget), together with the rule that
  a phase whose gate fails ships its feature disabled. The matrix requirement
  gains C1 availability and the demand that two knobs landing in one phase are
  attributed by separate arms rather than as one lump.
- `generation-telemetry`: the diagnostics requirement states that computing
  diagnostics never changes generation behavior — true, and it stays true, but
  once a consumer exists the boundary must be stated rather than implied.
  Telemetry additionally has to report the temperature actually applied, or the
  knob becomes unobservable in the live chat.

## Impact

- `app/core/markov.py` — the per-step power derivation and its call sites in the
  walk; entropy is already in hand at both of them, so no new per-step math.
- `app/core/response_generator.py` — candidate target derivation (M2R-110),
  currently the module constant `CANDIDATE_TARGET`.
- `app/config/registry.py`, `settings.py`, `runtime_state.py`, `.env.example` —
  new knobs through the established 5-step pattern (drift tests cover it).
- `app/core/generation_telemetry.py`, `app/core/gen_trace_log.py`,
  `app/presentation/bot_messages.py` — effective temperature in trace/`/stats`.
- `tools/eval/matrix.yaml`, `tools/eval/eval_thresholds.yaml`,
  `docs/eval_reports/`, `docs/v2/00_STATUS.md`.
- Risk to watch: this is a live-behavior change on a real chat, and the eval
  corpus cannot see mood modifiers or the flavor layer (both silenced in C0 by
  the established convention). The knob is runtime-revertible without a restart,
  which is the mitigation.
