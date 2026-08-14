## ADDED Requirements

### Requirement: The context mode of every reply is counted

The system SHALL count, per process, how many replies were generated **with**
context tokens and how many **without**, and SHALL publish the share in the same
place the other generation aggregates surface (`/stats`). The mode is decided by
the request the pipeline built, not by whether the reply happened to use the
context: a request carrying no context tokens is `noctx` even if the generator
found an anchor by other means.

The counters exist so that a gate verdict can be weighted by the measured
production share of each mode (`docs/PRE_ROADMAP.md` §4). A verdict SHALL NOT
claim a production-weighted result while this share is unmeasured.

Counting SHALL NOT consume random draws and SHALL NOT change generation output.

#### Scenario: Reply with context

- **WHEN** a reply is generated from a request whose context token list is non-empty
- **THEN** the `ctx` counter increments and the `noctx` counter does not

#### Scenario: Reply without context

- **WHEN** a reply is generated from a request whose context token list is empty
- **THEN** the `noctx` counter increments and the `ctx` counter does not

#### Scenario: Share is reported

- **WHEN** `/stats` is called after replies in both modes
- **THEN** the reply includes the share of generations built with context, derived from the process counters

#### Scenario: Output unchanged

- **WHEN** the same generation runs with the mode counters present and absent, under the same seed
- **THEN** the generated text is byte-identical

### Requirement: A channel emptied by data is distinguishable from one turned off by a knob

For every data-fed generation channel, the system SHALL count the case where the
channel **ran and returned nothing because its data source was empty**,
separately from the case where its knob is neutral and the channel never ran. At
minimum this SHALL cover:

- the seeded channel skipped because the chat's document count is zero, counted
  only when the seeded ratio knob is non-zero (that is: the channel was asked
  for and could not answer);
- the hot-n-gram seeding channel whose selection query returned an empty list
  when a seed was actually drawn for;
- the context of a generation dropped by exhausting the with-context attempt
  budget, counted per generation.

Each counter SHALL be published with the denominator that makes it readable —
the number of generations in which the channel was asked for — so that a zero
rate means "asked and always answered" rather than "never asked". A channel with
its knob at the neutral value SHALL report zero in both numerator and
denominator rather than being silently absent.

Counting SHALL NOT consume random draws and SHALL NOT change generation output.

#### Scenario: Seeded channel starved of document frequency

- **WHEN** the seeded ratio knob is non-zero and the chat's document count is zero
- **THEN** the "seeded skipped, no corpus" counter increments and its denominator (generations where seeding was asked for) increments too

#### Scenario: Seeded channel not asked for

- **WHEN** the seeded ratio knob is at its neutral value
- **THEN** neither the skip counter nor its denominator increments, and the reported rate is undefined rather than zero

#### Scenario: Hot n-grams selection returns nothing

- **WHEN** a seed draw queries the hot n-grams and the query returns an empty list
- **THEN** the "hot n-grams empty" counter increments against the number of draws that queried it

#### Scenario: Context dropped by attempt budget

- **WHEN** a generation consumes more attempts than the with-context budget allows, so later attempts run without the context
- **THEN** the generation is counted as one whose context was dropped, and the generation trace records that it happened

#### Scenario: Trace records the drop

- **WHEN** the optional per-generation trace is enabled and a generation drops its context by attempt budget
- **THEN** the trace line for that generation states the drop, alongside the existing context fields

#### Scenario: Output unchanged

- **WHEN** the same generation runs with the empty-channel counters present and absent, under the same seed
- **THEN** the generated text is byte-identical
