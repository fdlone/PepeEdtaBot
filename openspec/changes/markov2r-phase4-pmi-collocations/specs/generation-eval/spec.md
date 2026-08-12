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
