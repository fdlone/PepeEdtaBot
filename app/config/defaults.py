MESSAGES_RETENTION_PER_CHAT = 1000
SQLITE_BUSY_TIMEOUT_MS = 5000
SQLITE_WAL_AUTOCHECKPOINT_PAGES = 1000
# M2R-210 (TZ §7.2). The short layer accumulates against this half-life, so the
# writer and every reader must agree on it; the runtime knob of the same name
# overrides it per chat and resets the layer when it changes.
SHORT_HALF_LIFE_DAYS = 3.0
