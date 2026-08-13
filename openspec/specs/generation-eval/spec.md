# generation-eval Specification

## Purpose
Offline evaluation harness for the Markov generation pipeline: ablation configuration matrix, normative metric definitions, seeded reproducibility, pre-registered gate thresholds, and report format. It is the acceptance gate for every Markov 2.0R phase (normative source: `docs/v2/05_MARKOV_2_0R_EVAL_PROTOCOL.md`).
## Requirements
### Requirement: Ablation configuration matrix

The eval runner SHALL evaluate a configuration matrix that includes, at minimum, C0 (frozen Markov 1.x baseline) and CF (all currently enabled 2.0R features), and SHALL support the per-feature configurations C1–C5 defined in doc 05 §2 as those features come into existence. For each configuration the report SHALL show every metric's value and its delta against C0. A feature's contribution SHALL be reportable both in isolation (its C-config vs C0) and incrementally (CF with the feature vs CF without it).

When a single phase lands more than one independently switchable knob, the matrix SHALL carry one arm per knob in addition to the phase's combined C-configuration, so that each knob's contribution is attributable on its own. A phase SHALL NOT report the joint effect of several knobs as the contribution of one.

#### Scenario: Full matrix run

- **WHEN** the runner is invoked with the full matrix on a snapshot
- **THEN** the report contains one column per configuration, each metric row shows value and delta vs C0 with confidence intervals, and no configuration is silently skipped

#### Scenario: Before any V2 feature exists

- **WHEN** the runner is invoked during Phase 0, when only Markov 1.x exists
- **THEN** the matrix degenerates to C0 (and CF ≡ C0), the run succeeds, and the report states that no V2 configurations are available yet

#### Scenario: Two knobs in one phase

- **WHEN** a phase ships two independently switchable knobs and its configuration is evaluated
- **THEN** the report shows a separate arm for each knob alone plus the combined arm, and each knob's delta against C0 is printed separately

### Requirement: Bit-for-bit reproducibility

Given a fixed snapshot, a fixed prompt-set version, and fixed seeds, two runs of the same configuration matrix SHALL produce byte-identical metric results. The `--seed` parameter SHALL control both prompt selection and the generation RNG. The default protocol run SHALL aggregate the three seeds 42, 1337, 2026.

#### Scenario: Repeated run

- **WHEN** the runner is executed twice with identical snapshot, prompts, and seeds
- **THEN** all metric values in the two reports are identical

#### Scenario: Seed changes output

- **WHEN** the runner is executed with a different seed
- **THEN** prompt selection and generation sampling differ, demonstrating the seed is actually threaded through

### Requirement: Normative metric definitions

Metrics SHALL be implemented per the definitions in doc 05 §3, and the implementation SHALL reference the section numbers it implements. Seeded metrics SHALL always be published as the pair `seeded_present_rate` and `seeded_win_rate_given_present` with absolute counts; a seeded win rate without its denominator SHALL NOT appear in any report. `context_affinity` SHALL always be published together with `context_affinity_without_copy`, the latter computed only over answers not flagged by `exact_context_copy_rate`.

#### Scenario: Seeded metrics denominators

- **WHEN** a configuration with seeded generation is evaluated
- **THEN** the report shows present-rate and win-rate-given-present with absolute counts, never a bare win rate

#### Scenario: Copy-robust affinity

- **WHEN** answers contain verbatim copies of the prompt
- **THEN** those answers are excluded from both numerator and denominator of `context_affinity_without_copy` while still counting toward `exact_context_copy_rate`

### Requirement: Fixed prompt set

The prompt set SHALL live in a versioned `eval_prompts.yaml` with four categories — generic, topical, meme-bait, short/degenerate — of at least 30 prompts each. Topical and meme-bait prompts SHALL be derived from the temporal snapshot by a deterministic seeded script, not written by hand. The report SHALL record the prompt-set version it used.

#### Scenario: Prompt provenance

- **WHEN** the prompt set is regenerated with the same snapshot and seed
- **THEN** the resulting `eval_prompts.yaml` is identical

### Requirement: Statistical treatment

Every proportion metric SHALL carry a 95% bootstrap confidence interval (≥1000 resamples, pooled over the three protocol seeds). A delta between configurations SHALL be reported as significant only when the delta's interval excludes zero; the report SHALL always print the interval, not only the point estimate. Deltas below the protocol's resolution (~3 p.p. at n=500) SHALL be treated as "no effect".

#### Scenario: Insignificant delta

- **WHEN** a configuration's metric delta interval covers zero
- **THEN** the report marks the delta as not significant regardless of the point estimate's sign

### Requirement: Pre-registered gate thresholds

Gate thresholds (Phase 2 entropy sampling, Phase 5 promotion, Phase 6 rate×harm, Phase 7 shadow) SHALL live in a versioned `eval_thresholds.yaml`, fixed before the corresponding gated data is examined. The runner SHALL evaluate gates strictly against the thresholds file and report each active gate as pass / fail / insufficient data. Threshold edits after fixation SHALL happen only via a dedicated commit with written justification.

The Phase 2 gate SHALL be two-sided and SHALL require all of: no significant increase in the exact-copy metric, a significant increase in lexical diversity (distinct-2 and distinct-3), no significant loss of topical affinity measured without copies, and generation latency within the performance budget. A phase whose gate is not met SHALL ship its feature disabled by default rather than adjusting the thresholds.

#### Scenario: Gate evaluation

- **WHEN** a run includes data relevant to a registered gate
- **THEN** the report's Gates section shows the gate verdict computed from `eval_thresholds.yaml`, with the numbers that produced it

#### Scenario: Diversity gained at the cost of copying

- **WHEN** a Phase 2 arm raises distinct-2/3 significantly but also raises the exact-copy metric significantly
- **THEN** the gate verdict is fail, and the arm is not eligible to become the default

#### Scenario: Effect below protocol resolution

- **WHEN** a Phase 2 arm moves every gated metric by less than the protocol's resolution, so no delta interval excludes zero
- **THEN** the gate reports fail rather than pass, because the required increase in diversity was not demonstrated

### Requirement: Report format and storage

The runner SHALL emit a markdown report with the sections defined in doc 05 §6 (header with snapshot/prompts/seeds, config matrix, metric table with CIs and deltas, per-category breakdown, gates, manual-eval summary when available, per-phase verdict). Reports SHALL be stored under `docs/eval_reports/` and referenced from the phase's change; phase decisions (including "closed without implementation") SHALL cite a concrete report.

#### Scenario: Report written

- **WHEN** a protocol run completes
- **THEN** a dated markdown report exists under `docs/eval_reports/` containing all mandatory sections

### Requirement: CI smoke

CI SHALL run a smoke evaluation — C0 and CF on `snapshot_synthetic`, 40 generations, one seed — for pull requests touching the generation core, and SHALL fail the job on runner errors or invariant violations (invalid distributions, empty-output collapse). The full matrix SHALL NOT run in CI; it is executed manually at phase completion.

#### Scenario: Smoke on generation PR

- **WHEN** a PR modifies generation-core code
- **THEN** the smoke job runs the reduced protocol on the synthetic snapshot and fails on error or invariant violation

### Requirement: Snapshot and report privacy

Snapshots (full and temporal) SHALL NOT leave the developer's machine and SHALL NOT be committed to the repository. Committed eval artifacts — reports, prompt sets, thresholds, fixtures — SHALL NOT contain real chat identifiers or other identifiers covered by the project's log-privacy rules; where an identifier is needed in a committed artifact, a synthetic one SHALL be used.

#### Scenario: Committed artifacts scrubbed

- **WHEN** an eval report or prompt set is committed
- **THEN** it contains no real chat identifiers (guarded by the existing no-real-chat-ids test convention)

### Requirement: Temporal snapshot is a buildable artifact

The eval harness SHALL be able to build a temporal snapshot — a chain whose
transitions carry real, differing observation times — from the retained messages
of an existing snapshot, using their stored timestamps and a deterministic
replay. The builder SHALL write to a separate evaluation database and SHALL NOT
modify the source snapshot or any live chain.

The report SHALL state, for every run on such a snapshot, that the temporal
distribution was reconstructed from a retention window rather than accumulated
live, and SHALL print the resulting time span and the size of the fresh slice.
Freshness metrics computed on a reconstructed snapshot SHALL NOT be presented as
equivalent to metrics from live accumulation.

#### Scenario: Building the temporal snapshot twice

- **WHEN** the builder runs twice over the same source snapshot with the same seed
- **THEN** the two resulting databases produce byte-identical metric results

#### Scenario: Source snapshot untouched

- **WHEN** the builder runs
- **THEN** the source snapshot's tables are unchanged and the reconstructed chain lives in a separate file

#### Scenario: Provenance stated in the report

- **WHEN** a run uses a reconstructed temporal snapshot
- **THEN** the report names the snapshot as reconstructed, prints its time span and fresh-slice size, and freshness metrics carry that qualification

### Requirement: Freshness is measured against a fresh slice

On a snapshot carrying a temporal distribution, the runner SHALL report the
share of generated tokens drawn from the fresh slice of the corpus, and SHALL
report it for each configuration with a confidence interval and a delta against
C0, exactly as every other protocol metric. Where the snapshot has no temporal
distribution, the metric SHALL report insufficient data rather than a number.

#### Scenario: Freshness on a temporal snapshot

- **WHEN** a configuration is evaluated on a snapshot with a fresh slice
- **THEN** the report shows its fresh-token share with a confidence interval and a delta against C0

#### Scenario: Freshness without a temporal snapshot

- **WHEN** the same configuration is evaluated on a snapshot without observation times
- **THEN** the metric reports insufficient data and no number is printed

### Requirement: Memes from the historical slice must survive the blend

The runner SHALL evaluate whether n-grams belonging to the historical slice
remain reproducible when the blend is active. A configuration SHALL fail this
check when the historical n-gram list is reproduced significantly less often
than under C0 — the blend is required to add fresh language, not to overwrite
the chat's long memory.

#### Scenario: Blend erases long memory

- **WHEN** a configuration reproduces significantly fewer historical n-grams than C0
- **THEN** the check fails for that configuration and the report names the shortfall with its interval

### Requirement: Phase 3 gate is pre-registered and two-sided

The `phase3_temporal` gate SHALL be registered in the thresholds file before any
Phase 3 grid is run, and SHALL be two-sided: a configuration passes only if the
fresh-token share rises significantly AND the historical-meme check holds AND
copying does not rise significantly AND topicality measured without copies does
not drop significantly AND latency stays within budget. The runner SHALL print
the numbers that produced each verdict and SHALL report insufficient data when
an arm is missing.

#### Scenario: Arm improves freshness at the cost of long memory

- **WHEN** an arm raises the fresh-token share significantly but fails the historical-meme check
- **THEN** the gate verdict for that arm is fail, and the report shows both numbers

#### Scenario: Missing arm

- **WHEN** the grid lacks an arm the gate refers to
- **THEN** the verdict is insufficient data, never a pass

### Requirement: The seeded-generation arm and its promotion gate are computed

The eval matrix SHALL carry the C4 arm (lexical anchoring: seeded candidates in
the pool) as an available configuration once seeded generation exists, and the
report SHALL compute the `phase5_promotion` gate from the run rather than
stating that seeded generation does not exist.

The gate SHALL be computed strictly from the pre-registered thresholds: seeded
present rate, seeded win rate given present, the `context_affinity_without_copy`
delta against the no-seeded arm, latency p95, and storage growth. A promotion
verdict SHALL require all of them; failing any one SHALL NOT read as a pass.

#### Scenario: Seeded arm in a protocol run

- **WHEN** the runner evaluates the matrix with seeded generation available
- **THEN** the C4 arm is run and the report shows its seeded present rate, seeded win rate given present, affinity delta vs the no-seeded arm, p95 and storage against the pre-registered thresholds

### Requirement: The promotion gate withholds a pass until df is accumulated

Document frequency is accumulated on live prod after the reverse-index
migration and cannot be reconstructed from the retention window. Until a
protocol run exists over prod-accumulated df, the `phase5_promotion` gate SHALL
report `insufficient data`, never `pass`: an eval whose df is approximated from
the retained message window measures the machinery, not the phase's real
effect, and a green verdict bought that way would be a pass over the easy half.

#### Scenario: Eval over window-approximated df

- **WHEN** the seeded arm runs with df populated from the retained message window rather than prod-accumulated df
- **THEN** the report computes and prints the automatic gate conditions but the `phase5_promotion` verdict is `insufficient data`, with the reason stated

### Requirement: A two-dimensional gate closes on a decisively-failing arm

When a phase gate requires two conditions to both hold (rate AND harm, ADR-015),
and one condition is measured *significantly* below its threshold — the
measured interval lies wholly below the bar — the gate SHALL resolve to a
close/fail verdict without waiting on the other condition. A condition whose
manual or slower component is still missing SHALL NOT keep the gate at
`insufficient data` when the other condition already makes the required
conjunction impossible.

This is the "a demonstrated miss outranks a missing part" rule the other gates
already apply, extended to a conjunction: if either arm cannot be met, the
conjunction cannot be met, and the phase closes without implementation
(roadmap: "непревышение любого порога ⇒ фаза закрывается без реализации").

The report SHALL state which arm failed and by how much, so the closed verdict
is auditable from its numbers. If neither arm is decisively below its threshold
and a required arm is unmeasured, the gate SHALL remain `insufficient data`.

#### Scenario: Detection arm decisively below threshold

- **WHEN** the anti-cycle gate's detection rate is measured with its whole confidence interval below the detection threshold, and the harm arm's manual component has not been collected
- **THEN** the gate resolves to close/fail — the phase is closed without implementation — and the report shows the detection rate, its interval, and the threshold it missed

#### Scenario: Neither arm decisive, one unmeasured

- **WHEN** the detection rate is near or above its threshold but the harm arm is still unmeasured
- **THEN** the gate remains `insufficient data`, because the conjunction is not yet decidable

### Requirement: The shadow order-4 gate renders a verdict once its sample suffices

The Phase 7 shadow gate SHALL report `insufficient data` only while the
shadow-eligible step count is below its minimum; once the eligible count clears
that minimum, the gate SHALL resolve to pass or fail from the measured order-4
selected share against its threshold. A selected share below the threshold at a
sufficient sample SHALL render **fail**, which closes the phase without building
the order-4 index (ADR-002).

The report SHALL show the eligible step count, the selected share, and the
threshold, so the verdict is auditable from its numbers. The shadow selector is
measurement-only: its verdict SHALL NOT depend on, and SHALL NOT change,
generated output.

#### Scenario: Sufficient sample, order-4 never selected

- **WHEN** the shadow-eligible step count is at or above its minimum and the order-4 selected share is below the threshold
- **THEN** the gate renders fail, the phase is closed without implementation, and the report shows the eligible count, the selected share, and the threshold

#### Scenario: Sample still below the minimum

- **WHEN** the shadow-eligible step count is below its minimum
- **THEN** the gate reports `insufficient data`, stating how many eligible steps were observed and how many are required


### Requirement: Phase 4 gate rests on human judgement, and says so

The `phase4_memes` gate SHALL be registered in the thresholds file before any
Phase 4 ranking is examined, and SHALL require that at least a configured share
of the top-ranked memes are judged genuine by human raters (doc 05 §5), together
with the automatic condition that the meme-aware configuration does not worsen
copying or copy-free topicality significantly.

The runner SHALL report `insufficient data` — never `pass` — while the manual
rating is missing. An automatic-only verdict SHALL NOT be presented as a gate
result, because the thing this phase claims to improve is exactly the thing no
metric here can measure.

#### Scenario: Ranking produced, rating not yet done

- **WHEN** the analyzer has produced a ranking but no manual rating exists
- **THEN** the gate reports insufficient data and names the missing rating

#### Scenario: Rating below the bar

- **WHEN** fewer than the required share of the top memes are rated genuine
- **THEN** the gate fails, and the report carries the counts that produced the verdict

#### Scenario: Automatic conditions worsen

- **WHEN** the meme-aware configuration significantly raises copying or lowers copy-free topicality
- **THEN** the gate fails regardless of how the manual rating came out

### Requirement: The rating round carries a blind control and validity decoys

The rated list SHALL mix the association-ranked memes with an equally sized
selection from the frequency-based mechanism they are meant to replace, plus a
small number of decoy pairs drawn from neither, all shuffled and presented
without their source. Raters SHALL NOT be told which item came from where.

The association half SHALL be required to score no lower than the frequency
half. The requirement is "no worse" rather than "measurably better" on purpose:
one binary judgement over twenty items cannot resolve a difference of a few
positions, and a gate no achievable evidence can pass is a veto rather than a
gate. The observed difference SHALL be reported as a number regardless.

An absolute share alone SHALL NOT decide this gate, because a lenient rater
raises every share and a strict one lowers every share — the absolute number
measures the rater as much as the ranking, while the difference between halves
does not.

#### Scenario: Association ranking ties the frequency control

- **WHEN** both halves are rated genuine at the same share and the absolute bar is met
- **THEN** the gate passes on this condition, and the report shows the difference as approximately zero

#### Scenario: Association ranking scores below the control

- **WHEN** the association half is rated genuine less often than the frequency half
- **THEN** the gate fails, because the mechanism being replaced performed better

#### Scenario: Raters see no sources

- **WHEN** the rating list is produced
- **THEN** items from all three sources are shuffled together and carry no marking, and the mapping back to sources is kept separate from the list that is sent out

### Requirement: A noisy rating round is undecided, not failed

When the decoy items are rated genuine more often than a configured bar, the
round SHALL be reported as `insufficient data` rather than `fail`. A rating
round whose own controls failed says nothing about the phase, and recording it
as a failure would later be read as evidence the approach does not work.

#### Scenario: Decoys rated as genuine memes

- **WHEN** the decoy false-positive share exceeds the bar
- **THEN** the verdict is insufficient data, the decoy share is named as the reason, and neither half's share is treated as meaningful

### Requirement: Manual ratings are versioned without leaking chat content

The result of a manual rating round SHALL be recorded alongside the run it
belongs to, in a form that survives the session: which ranking version was
rated, how many raters took part, the per-category counts, and the
inter-rater agreement when there was more than one rater.

The rated items themselves are verbatim chat phrases and SHALL NOT be committed.
A committed report SHALL therefore carry the aggregate outcome and the pointer
to the local rating sheet, never the phrases.

#### Scenario: Rating round completed

- **WHEN** raters finish a round
- **THEN** the committed report shows counts, shares and agreement, and contains no rated phrase

#### Scenario: Single rater

- **WHEN** only one person rated
- **THEN** the report states that agreement is unavailable rather than implying consensus
