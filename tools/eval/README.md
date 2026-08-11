# tools/eval — Markov 2.0R evaluation protocol runner

Normative source: `docs/v2/05_MARKOV_2_0R_EVAL_PROTOCOL.md`. This package is the
protocol implementation; the knob-sweep harness `tools/eval_prod.py` and the
byte-identity guard `tools/generation_hash.py` stay independent.

## Usage

```bash
# Full protocol run (500 generations x seeds 42,1337,2026 per configuration):
python -m tools.eval --db db_prod_copy/markov.db --label baseline-C0-2026-07-13

# CI smoke (doc 05 §7): C0+CF, synthetic snapshot, 40 generations, 1 seed:
python -m tools.eval --smoke

# Regenerate the prompt set (a deliberate act — produces a new version):
python -m tools.eval --generate-prompts --db <snapshot> --label <snapshot-id>
```

Reports land in `docs/eval_reports/`; `--json-out` additionally writes the
deterministic metric summary used by reproducibility checks.

## Snapshots (doc 05 §1.1)

- **snapshot_full** — an anonymized copy of the production DB. Preparation:
  copy the DB file together with its `-wal`/`-shm` sidecars (the runner does
  this itself via `copy_database` and applies pending migrations to the copy,
  never to the source). The snapshot id is `<date of the dump>`; record it in
  `--label`. Snapshots never leave the developer's machine and are never
  committed (`db_prod_copy/` is gitignored). The frozen Phase 0 baseline:
  `db_prod_copy/markov.db`, dated **2026-07-13** (owner decision 2026-08-11:
  a fresher dump is not available — audit §10.2).
- **snapshot_temporal** — does not exist until Phase 3 adds timestamps and a
  real accumulation window (audit §10.1). Temporal metrics report
  `insufficient data` until then.
- **snapshot_synthetic** — built on demand, deterministically, by
  `tools/eval/synthetic.py` (seeded wrapper around `tools/seed_diverse`);
  never committed.

## Files

- `matrix.yaml` — ablation matrix C0–CF (doc 05 §2). C0 is frozen; phases add
  their configurations by editing this file.
- `eval_prompts.yaml` — versioned prompt set (doc 05 §1.2), generated from the
  snapshot, four categories ≥30 prompts each, plus the meme list for the
  §3.5 gate. The `version` field is a content hash; `load_prompts` rejects
  hand-edited files.
- `eval_thresholds.yaml` — pre-registered gate thresholds (doc 05 §4). Editing
  after fixation requires a dedicated commit with written justification.
