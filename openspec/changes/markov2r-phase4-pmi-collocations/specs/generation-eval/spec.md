## ADDED Requirements

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
