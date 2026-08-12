## MODIFIED Requirements

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
