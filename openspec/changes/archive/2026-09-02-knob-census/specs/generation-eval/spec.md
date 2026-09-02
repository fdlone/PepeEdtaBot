## ADDED Requirements

### Requirement: The knob census is a reproducible run with a pre-registered rule

The eval tooling SHALL provide a knob census that builds its arms from the
runtime registry (the extremes of each measurable knob's domain, a flipped
value for booleans, and the same extremes with the parent knob enabled for
knobs gated by another knob), runs them on a snapshot in both context modes
against one baseline, and classifies every knob by a rule registered in
`eval_thresholds.yaml` before the run: dead (not read on a live path), gated,
inert, weak, strong. Knobs the harness cannot exercise SHALL be listed as
outside the offline measurement, never as inert. The census report SHALL
carry numbers only — no reply text and no n-grams — and SHALL NOT change any
default: deletions and merges are a separate change.

#### Scenario: Inert knob

- **WHEN** both extremes of a knob leave every classification metric's delta
  interval inside the tolerance band in both modes
- **THEN** the knob is classified inert and the report proposes removing it or
  reducing it to a constant

#### Scenario: Gated knob

- **WHEN** a knob's extremes move nothing while its parent is at the default
  but move a metric significantly with the parent enabled
- **THEN** the knob is classified gated, not inert

#### Scenario: Knob outside the harness

- **WHEN** a knob is read only by the reply pipeline or handlers
- **THEN** the report lists it as outside the offline measurement with the
  place it is read
