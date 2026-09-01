# generation-telemetry — delta (extension-seam-accounting)

## ADDED Requirements

### Requirement: The winner's route is reported per reply

The generation result SHALL carry the winning candidate's route (the same
closed route enumeration the trace and aggregate telemetry already use), so
that a consumer can tell per reply whether the winner was produced by the
main walk, a verbatim extension, a seeded assembly, or a mutation — without
scraping logs. When no reply is produced, the route SHALL be absent.
Reporting the route SHALL NOT change generation behavior.

#### Scenario: Extended winner is visible per reply

- **WHEN** the selected winner is a verbatim-extended candidate
- **THEN** the generation result names the extension route, even when the spliced connective is the silent one that text scanning cannot see

#### Scenario: No reply, no route

- **WHEN** generation produces no reply
- **THEN** the result carries no winner route

### Requirement: Seam share is measured honestly in the sweep harness

The sweep harness's connective-reply metric SHALL count a reply as carrying a
seam when a wordy connective is found in the text **or** the winner carried a
seam by construction (extension route, or a walk with at least one jump). The
share of replies won by an extended candidate SHALL also be published as its
own number, and the harness output SHALL state that the amended numerator
breaks comparability with pre-amendment sweeps.

#### Scenario: Silent extension seam is counted

- **WHEN** a sweep reply's winner is an extension whose connective is the silent one
- **THEN** the connective-reply metric counts that reply, and the extension share reflects it

#### Scenario: The comparability break is visible

- **WHEN** the harness reports the amended metric
- **THEN** the output names the break with pre-amendment sweeps rather than presenting the numbers as directly comparable
