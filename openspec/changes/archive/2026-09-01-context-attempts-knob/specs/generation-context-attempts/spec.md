## ADDED Requirements

### Requirement: The number of context-bearing attempts is a runtime knob

The number of pool-building attempts that receive the reply context SHALL be
a runtime knob (`generation_attempts_with_context`, env
`GENERATION_ATTEMPTS_WITH_CONTEXT`) with the default equal to the previous
module constant (5) and a range from 0 to the attempt budget. Attempts past
that number SHALL run without context, exactly as before, and the existing
`CONTEXT DROPPED` trace event and `context_dropped` counter SHALL keep
reporting the switch. At the default the generation SHALL be byte-identical
to the pre-knob behaviour.

#### Scenario: Default keeps behaviour

- **WHEN** the knob is at its default
- **THEN** the sixth attempt of a context generation runs without context
- **AND** `generation_hash` does not move

#### Scenario: Knob lowers the drop point

- **WHEN** the knob is set to 2 via `/set` or an eval override
- **THEN** the third attempt already runs without context
- **AND** the drop is reported once per generation, as before

#### Scenario: Zero disables context for the pool

- **WHEN** the knob is 0 and the request carries context
- **THEN** no attempt receives context tokens
- **AND** the drop is reported on the first attempt
