## MODIFIED Requirements

### Requirement: Changing the half-life resets the short layer

The short counter is only meaningful against the half-life it accumulated under.
When the half-life changes, the system SHALL reset the short layer to empty and
SHALL report the reset explicitly to whoever made the change. The reset SHALL
NOT happen silently, and the long layer SHALL NOT be touched by it.

The reset SHALL cover exactly the chats whose **effective** half-life changed.
A global change SHALL NOT reset the short layer of a chat that overrides the
half-life for itself: that chat's counter still accumulates under its own,
unchanged half-life, so wiping it would discard data without changing any
scale. The acknowledgement of a global change SHALL state how many chats kept
their layer for that reason, as a count, without naming the chats.

#### Scenario: Half-life changed at runtime

- **WHEN** the half-life setting is changed
- **THEN** the short layer is emptied, the change is acknowledged with an explicit warning that fresh-language memory was discarded, and long counts are unchanged

#### Scenario: Half-life set to its current value

- **WHEN** the half-life is set to the value it already has
- **THEN** the short layer is left intact

#### Scenario: Global change spares chats with their own half-life

- **WHEN** the global half-life is changed while some chats override the half-life for themselves
- **THEN** the short layer is emptied only in the chats living on the global value, the overriding chats keep their short layer intact, and the acknowledgement states how many chats were spared

#### Scenario: Per-chat change touches only the chat of the call

- **WHEN** the half-life is changed or its override is cleared for one chat
- **THEN** only that chat's short layer is emptied, regardless of what other chats override
