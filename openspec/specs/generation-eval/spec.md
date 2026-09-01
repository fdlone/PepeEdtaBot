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

Матрица конфигураций — **парный дизайн**: все армы проходят один и тот же
набор промптов под одними и теми же сидами, различаясь ровно одной ручкой.
Интервал дельты между армами SHALL учитывать эту парность: ресэмплироваться
SHALL наблюдения как пары, а не два арма независимо.

Независимый ресэмплинг двух армов SHALL считаться несоответствием, а не
допустимой реализацией: он оценивает `Var_A + Var_B` вместо дисперсии парной
разности и потому расширяет интервал тем сильнее, чем выше корреляция армов.
Ошибка направленная — она порождает ложные вердикты «эффекта нет», а не
ложные «эффект есть».

Наблюдение, у которого один из армов не дал ответа, SHALL исключаться из пары
целиком: полупара не является наблюдением разности.

Отчёт SHALL указывать, что интервал дельты парный, — иначе читатель сравнит
его с интервалами прежних отчётов, снятыми другой статистикой.

#### Scenario: Insignificant delta

- **WHEN** a configuration's metric delta interval covers zero
- **THEN** the report marks the delta as not significant regardless of the point estimate's sign

#### Scenario: Коррелированные армы

- **WHEN** два арма дают положительно коррелированные наблюдения на одних и
  тех же промптах
- **THEN** интервал парной дельты уже, чем интервал, полученный независимым
  ресэмплингом тех же данных
- **AND** точечная оценка дельты у обоих способов совпадает

#### Scenario: Арм не дал ответа на промпт

- **WHEN** на каком-то промпте один из армов не дал ответа
- **THEN** это наблюдение не участвует в оценке дельты ни одной своей половиной

#### Scenario: Разрыв сопоставимости объявлен

- **WHEN** отчёт напечатан после перехода на парную оценку
- **THEN** из него видно, что интервал дельты парный
- **AND** прошлые отчёты не пересчитываются и не сравниваются с новыми по
  ширине интервала

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

Every section SHALL describe the run that produced it. A section SHALL NOT state that an input was absent when that input was supplied: where a section has a "not available" wording, that wording SHALL be reachable only when the corresponding input is genuinely missing.

#### Scenario: Report written

- **WHEN** a protocol run completes
- **THEN** a dated markdown report exists under `docs/eval_reports/` containing all mandatory sections

#### Scenario: Manual rating supplied

- **WHEN** a protocol run is given a manual rating aggregate
- **THEN** the manual-eval section reports that aggregate — the rated and genuine counts of both halves, the decoy counts, the number of raters and the agreement — and does not state that no round was conducted

#### Scenario: Manual rating absent

- **WHEN** a protocol run is given no manual rating aggregate
- **THEN** the manual-eval section says the round was not conducted in this run

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

Prod-accumulated df SHALL NOT by itself satisfy this precondition. The gate
SHALL additionally require a pre-registered floor on corpus **volume** and a
pre-registered ceiling on the **share of tokens observed in exactly one
document**, both fixed in the thresholds file before the gated run. Where either
is not met, the verdict SHALL be `insufficient data`, never `pass` or `fail`.

Both parts are required because each is blind to the other failure. Volume alone
repeats a bar the project has already retired: the "≥1000 phrases" floor was
dropped once it was shown to measure supply while selection is what decides.
A corpus of any size in which nearly every token appears in one document gives
`log(n_docs / (1 + df))` the same value for nearly every token, so seed ranking
has nothing to rank. Conversely a favourable singleton share over a handful of
documents is noise, not evidence.

The report SHALL print both measured quantities next to the verdict, so that
"not yet" is distinguishable from "measured and did not hold" without reading
the database.

The precondition SHALL NOT be expressed as elapsed time since deployment.
Accumulation is bounded by chat activity, which is not constant and is not under
the project's control.

#### Scenario: Eval over window-approximated df

- **WHEN** the seeded arm runs with df populated from the retained message window rather than prod-accumulated df
- **THEN** the report computes and prints the automatic gate conditions but the `phase5_promotion` verdict is `insufficient data`, with the reason stated

#### Scenario: Prod-accumulated df below the volume floor

- **WHEN** the seeded arm runs over prod-accumulated df whose document count is below the pre-registered floor
- **THEN** the verdict is `insufficient data`, the report names the volume floor and the measured document count, and no `pass` or `fail` is issued

#### Scenario: Enough documents, but df cannot discriminate

- **WHEN** the document count meets the floor while the share of tokens observed in exactly one document exceeds the pre-registered ceiling
- **THEN** the verdict is `insufficient data`, the report names the ceiling and the measured share, and no `pass` or `fail` is issued

#### Scenario: Both parts of the precondition met

- **WHEN** the document count meets the floor and the singleton share is within the ceiling
- **THEN** the precondition is satisfied, both measured quantities are printed, and the gate resolves to `pass` or `fail` strictly from the remaining pre-registered conditions

#### Scenario: Thresholds are read, not inferred

- **WHEN** the run needs the volume floor or the singleton ceiling
- **THEN** the values come from the thresholds file, and a run whose thresholds file lacks them reports `insufficient data` rather than substituting a default

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

### Requirement: The frozen-baseline generation hash is recorded and checked

The generation hash that defines the frozen baseline SHALL be recorded in a single machine-readable location, and the hash guard SHALL compare its computed value against that record and report a mismatch as a failure. Prose in documents and change tasks MAY cite the value, but SHALL NOT be the place it is defined.

The recorded value SHALL be changed only by a deliberate edit that states which change moved it and why the move is legitimate. A run that produces a different hash SHALL therefore surface as a failing check at the moment it happens, rather than as a discrepancy noticed later against a number quoted in a document.

#### Scenario: Hash matches the record

- **WHEN** the hash guard runs on the frozen snapshot at neutral settings and computes the recorded value
- **THEN** the guard reports a match and exits successfully

#### Scenario: Hash differs from the record

- **WHEN** the hash guard runs and computes a value other than the recorded one
- **THEN** the guard reports the mismatch with both values and exits with a failure status, so the drift is attributed to the change that caused it

#### Scenario: Re-anchoring the baseline

- **WHEN** a change legitimately alters generation output at neutral settings and the recorded value is updated
- **THEN** the record carries the reason and the originating change, so a later reader can tell a deliberate re-anchor from an unnoticed regression

### Requirement: Phase 9 gate is pre-registered and two-sided on quality

Гейт `phase9_interp` SHALL быть записан в файл порогов с датой регистрации
**раньше** даты прогона грида и после прогона SHALL NOT изменяться. Рука
проходит, только если выполнены **все** условия (CI 95%, значимость — интервал
исключает 0):

1. `distinct-2` **и** `distinct-3` выросли значимо против C0 — целевая метрика
   фазы: впервые в проекте у неё есть механизм расти;
2. `exact_context_copy_rate` не вырос значимо (снижение ожидается, но не
   требуется);
3. `context_affinity_without_copy`: нижняя граница CI дельты **больше −0.02** —
   пол, тот же класс защиты, что отклонил M2R-110;
4. `repetition_rate` не вырос значимо и `cycle_detection_rate` < 0.05 —
   страховка: смешивание меняет топологию блужданий, а вердикт Phase 6 снимался
   на другой топологии;
5. латентность `p95` ≤ 150 мс;
6. раунд связности по соло-протоколу: доля связных ответов руки не ниже C0
   более чем на 10 процентных пунктов.

Двусторонность по качеству обязательна: у фазы есть механизм поднять
разнообразие ценой связности, и односторонний гейт по `distinct` пропустил бы
именно этот размен.

Если ни одна рука не прошла, вес SHALL остаться нулевым, а результат SHALL быть
оформлен вердикт-документом — «закрыто с цифрами» здесь полноценный исход, а не
неудача.

#### Scenario: Grid run against a pre-registered gate

- **WHEN** грид прогнан и отчёт строится
- **THEN** каждая рука получает вердикт по всем шести условиям, а не по одному целевому

#### Scenario: Diversity grows at the cost of topicality

- **WHEN** рука значимо поднимает `distinct-2`/`distinct-3`, но нижняя граница CI дельты `context_affinity_without_copy` не превышает −0.02
- **THEN** рука не проходит гейт

#### Scenario: No arm passes

- **WHEN** ни одна рука не выполнила все условия
- **THEN** вес остаётся нулевым, пороги не трогаются, результат фиксируется вердикт-документом

### Requirement: The connectedness round has a validity floor of its own

Раунд связности SHALL проводиться по бинарной рубрике «связно / салат» с
письменным определением и калибровочными примерами, размеченными **до**
основного раунда; подача SHALL быть слепой (рука не видна), состав —
перемешанные C0, руки и декои; объём — не менее 30 ответов на руку.

Раунд SHALL иметь порог валидности, и порог SHALL быть свойством того
инструмента, которым раунд реально проводится. Панельная согласованность
(Fleiss' kappa) для этого не годится: панель из ≥3 незаинтересованных оценщиков
проекту недоступна, и попытка её собрать уже измерена — согласованность 0.17 в
Phase 4 сделала вердикт резиновым. Валидность SHALL проверяться
само-консистентностью соло-протокола: не менее 20% выборки предъявляются
повторно, согласие оценщика с собой не ниже 0.8. Ниже порога раунд **не
засчитывается** — рубрика дорабатывается и раунд повторяется; усреднение
несогласованных оценок SHALL NOT считаться результатом.

Содержательный порог гейта при этом SHALL оставаться неизменным: меняется
инструмент измерения связности, а не требование к ней.

#### Scenario: Rater disagrees with themselves

- **WHEN** согласие оценщика с собой на повторно предъявленной части ниже порога
- **THEN** раунд не засчитывается, и гейт остаётся нерешённым — не пройденным и не провалённым

#### Scenario: Valid round

- **WHEN** само-консистентность не ниже порога и объём набран
- **THEN** доля связных ответов каждой руки сравнивается с C0 по условию гейта

### Requirement: Every configuration is evaluable in both context modes

The eval runner SHALL support two generation modes and SHALL be able to run any
configuration of the matrix in either:

- `ctx` — the prompt is supplied to the generator as context (the historical
  behavior of every run to date);
- `noctx` — the generator receives no context tokens; the prompt still selects
  the generation and seeds the RNG, so the two modes remain paired per prompt.

The mode SHALL be selectable per run and SHALL default to `ctx`, so an
invocation written before this requirement keeps producing the same numbers.
Within a run the mode SHALL be fixed: a single report SHALL NOT mix modes into
one metric value.

#### Scenario: Explicit noctx run

- **WHEN** the runner is invoked in `noctx` mode
- **THEN** every generation is requested without context tokens, and the run completes with the same metric set as a `ctx` run

#### Scenario: Default mode is unchanged

- **WHEN** the runner is invoked without naming a mode
- **THEN** it runs in `ctx` mode and reproduces the numbers of the pre-existing invocation bit-for-bit under the same seeds and snapshot

#### Scenario: Modes are paired per prompt

- **WHEN** the same configuration, snapshot, and seeds are run in both modes
- **THEN** both runs draw the same prompts in the same order, so metric deltas between modes are attributable to the context alone

### Requirement: A report names its mode, and a one-mode verdict is not a passed gate

Every eval report SHALL state the mode it was produced in, in a place a reader
cannot miss, and SHALL carry the mode in its machine-readable output so a
verdict note cannot cite the wrong one.

A gate whose definition requires both modes SHALL NOT be reported as passed on
the strength of one: with only one mode measured, the gate result SHALL be
`insufficient data`, exactly as a gate lacking any other pre-registered input.
Where a verdict weighs the two modes together, the weight SHALL come from the
measured production share of each mode, and the report SHALL name the share it
used and where it came from.

#### Scenario: Mode is visible in the report

- **WHEN** a report is produced in either mode
- **THEN** the mode is stated in the report header and present in the machine-readable output

#### Scenario: Gate needing both modes, only one measured

- **WHEN** a gate requiring both modes is evaluated from a single-mode run
- **THEN** the gate reports `insufficient data`, never `pass`

#### Scenario: Weighted verdict states its weight

- **WHEN** a verdict combines the two modes into one number
- **THEN** the report states the production share used as the weight and its source

### Requirement: Structural diversity is measured per input, as a pair

For every generation the harness SHALL record how many **substantially
different trajectories** the machinery produced, at two scopes:

- over the whole candidate pool (the pool-level count, the safety invariant);
- over the **selection window** — the candidates within the selection score
  margin of the best one, i.e. those the final draw could actually have
  returned.

Two trajectories are substantially different when their **edge overlap** — the
share of adjacent token pairs they have in common — is below a threshold
registered in the thresholds file before any run. Overlap SHALL be normalized
by the smaller of the two edge sets, so that a candidate which is a sub-path of
another counts as the same trajectory rather than as a new one. The count is
the number of groups the window's candidates fall into once every
above-threshold pair is treated as the same trajectory.

Both numbers SHALL be reported **together, always**. A report SHALL NOT publish
the pool count without the window count or the reverse: the gap between them is
itself the finding ("there is diversity, none of it survives into the window"),
and either number alone reads as its opposite.

The quality criterion SHALL be window-relative, not an absolute score bar: the
window is defined by distance from the best candidate of that same generation,
because scores are not calibrated absolutely and their scale differs between
context modes.

Computing the metric SHALL NOT change generation output and SHALL NOT consume
random draws.

#### Scenario: A pool of near-identical candidates

- **WHEN** every candidate in a generation's pool shares most of its adjacent token pairs with the others
- **THEN** both counts are 1 — one trajectory with variants, not several trajectories

#### Scenario: Diversity that does not survive scoring

- **WHEN** the pool holds several substantially different trajectories but only one of them lies within the selection margin
- **THEN** the pool count is greater than one and the window count is 1, and both appear in the report

#### Scenario: A sub-path is not a new trajectory

- **WHEN** one candidate's adjacent token pairs are a subset of another's
- **THEN** the two are counted as the same trajectory

#### Scenario: The pair is never split

- **WHEN** a report publishes the structural metric
- **THEN** it contains the pool count and the window count for the same runs

#### Scenario: No candidates

- **WHEN** a generation collects no candidates at all
- **THEN** both counts are 0 for that generation and it does not silently drop out of the denominator

### Requirement: The structural escape gate is pre-registered

The thresholds file SHALL carry a `structural_escape` block registered before
the first run that reports the metric, holding at minimum: the edge-overlap
threshold that defines "substantially different", the minimum window count a
configuration must reach, and the pool-level safety floor that any change to
the pool composition must preserve.

A configuration SHALL pass the gate only when the mean window count meets its
minimum **and** the mean pool count stays at or above the safety floor. Until a
configuration's numbers exist the gate SHALL report insufficient data rather
than a pass, and it SHALL be subject to the same two-mode rule as every other
gate that declares it.

Thresholds SHALL NOT be changed after a run has been seen; a claim that a route
improved quality SHALL NOT be accepted as a structural result without this
metric alongside it.

#### Scenario: Gate registered before data

- **WHEN** the metric is reported for the first time
- **THEN** its thresholds already exist in the thresholds file and were not derived from the numbers being reported

#### Scenario: Window count too low

- **WHEN** a configuration's mean window count is below the registered minimum
- **THEN** the gate fails for that configuration, whatever its quality metrics show

#### Scenario: Pool safety floor breached

- **WHEN** a change to the pool composition drops the mean pool count below the registered floor
- **THEN** the gate fails even if the window count rose

### Requirement: The meme-regression set carries a support floor

N-граммы попадают в набор мем-регресса только при поддержке не ниже
пре-регистрированного порога. Порог SHALL храниться в файле порогов вместе с
обоснованием, а не быть константой кода.

Основание измеримое: окно горячих n-грамм затухает, поэтому подавляющее
большинство его строк имеет поддержку в одно сообщение (замер 2026-09-01:
1079 строк из 1097). Требование «воспроизведи n-грамму, встреченную однажды»
проверяет совпадение, а не память модели.

Набор SHALL нести свою версию, и её смена при перегенерации SHALL читаться как
разрыв сопоставимости с прошлыми отчётами, а не как обновление того же набора.

#### Scenario: Малоподдержанная n-грамма в набор не попадает

- **WHEN** набор строится по снапшоту, где часть n-грамм встречалась однажды
- **THEN** такие n-граммы в набор не входят, а вошедшие имеют поддержку не ниже порога

#### Scenario: Перегенерация объявлена

- **WHEN** набор перегенерирован с другим порогом или на другом снапшоте
- **THEN** версия набора меняется, и отчёт показывает, что числа несопоставимы с прежними

### Requirement: Meme regression is a share, measured per configuration

Гейт SHALL считать **долю воспроизведённых** мемов набора и SHALL публиковать
её вместе с абсолютными числами (сколько из скольких). Доля SHALL считаться
для **каждой** конфигурации прогона, а не только для базлайна: гейт существует,
чтобы поймать конфигурацию, стирающую память чата, и по одному базлайну этого
не видно.

Вердикт SHALL быть **относительным к базлайну**: конфигурация проваливает
проверку, когда её доля ниже доли базлайна больше чем на пре-регистрированный
допуск. Абсолютная планка SHALL NOT использоваться: она мерила бы толщину
корпуса, а не поведение конфигурации.

Если после порога поддержки в наборе меньше пре-регистрированного минимума
мемов, вердикт SHALL быть `insufficient data` с указанием размера набора —
доля по нескольким мемам не является свидетельством.

#### Scenario: Конфигурация стирает мемы

- **WHEN** доля воспроизведённых мемов у конфигурации ниже базлайнной больше чем на допуск
- **THEN** гейт отдаёт `fail`, и отчёт называет обе доли, абсолютные числа и допуск

#### Scenario: Конфигурация держит память

- **WHEN** доля конфигурации не ниже базлайнной за вычетом допуска
- **THEN** гейт отдаёт `pass`, и обе доли напечатаны

#### Scenario: Набор слишком мал

- **WHEN** после порога поддержки в наборе меньше минимального числа мемов
- **THEN** вердикт `insufficient data`, и отчёт называет размер набора и минимум

#### Scenario: Пустой набор

- **WHEN** набор пуст
- **THEN** вердикт `insufficient data`, а не `pass` по отсутствию нарушений

### Requirement: The connectedness condition is computed, not permanently absent

Условие связности гейта фазы 9 SHALL вычисляться из агрегата соло-раунда,
когда агрегат передан прогону, и SHALL давать `insufficient data` с названной
причиной, когда агрегата нет или раунд невалиден. Оно SHALL NOT оставаться
безусловно отсутствующим: «инструмента нет» и «инструмент есть, раунд не
проведён» — разные состояния, и отчёт обязан их различать.

Агрегат SHALL сверяться с тем же пре-регистрированным порогом отставания от
базлайна, который записан в файле порогов; порог SHALL NOT дублироваться в
коде.

#### Scenario: Раунд проведён и валиден

- **WHEN** прогону передан агрегат валидного раунда, где доля связных ответов руки не ниже базлайнной за вычетом порога
- **THEN** условие связности выполнено, и отчёт называет обе доли

#### Scenario: Рука теряет связность

- **WHEN** доля связных ответов руки ниже базлайнной больше чем на порог
- **THEN** условие провалено, и это провал гейта, а не недостаток данных

#### Scenario: Раунда нет

- **WHEN** агрегат не передан
- **THEN** условие отсутствует с причиной «раунд не проведён», и гейт даёт `insufficient data`
