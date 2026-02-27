# Architecture Notes

## Overview
- `main.py`: bot runtime, handlers, trigger policy, runtime config commands.
- `db.py`: SQLite schema, migrations, atomic writes, stats/queries.
- `markov.py`: tokenization + variable-order generation (`3 -> 2 -> 1`).
- `text_utils.py`: text sanitation pipeline.
- `settings.py`: `.env` loading and validation.

## Data Model
- `messages`: raw incoming messages for audit/debug.
- `starts3` / `transitions3`: primary trigram model.
- `starts` / `transitions`: bigram fallback model.
- `transitions1`: unigram fallback model.

## Runtime Controls
All runtime overrides are process-local and reset on restart:
- `/set reply_probability ...`
- `/set min_cooldown_sec ...`
- `/set min_tokens_for_model ...`
- `/set max_reply_chars ...`
- `/set normalize_lower ...`
- `/set typing_min_ms ...`
- `/set typing_max_ms ...`
- `/set randomness_strength ...`
- `/set markov_order ...`
- `/set enable_backoff ...`
- `/set backoff_min_order ...`

## Safety
- SQL queries are parameterized.
- `.env` is ignored by git.
- message deduplication guard prevents exact copy replies.
