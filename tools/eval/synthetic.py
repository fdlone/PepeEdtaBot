"""Deterministic ``snapshot_synthetic`` builder (doc 05 §1.1).

Wraps ``tools/seed_diverse`` into a temp DB with the global RNG seeded, so CI
smoke runs and tests get the same database every time without committing a
binary snapshot to the repository.
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import seed_diverse  # noqa: E402

SYNTHETIC_CHAT_ID = 1
SYNTHETIC_SEED = 20260811
SYNTHETIC_MESSAGES = 400


async def build_synthetic_snapshot(
    target_dir: Path | None = None,
    *,
    messages: int = SYNTHETIC_MESSAGES,
    seed: int = SYNTHETIC_SEED,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Create the synthetic snapshot; returns (db_path, tempdir-or-None).

    ``seed_diverse.run`` draws from the module-level ``random`` — seeding it
    here pins the corpus. The caller owns the returned TemporaryDirectory
    (``None`` when ``target_dir`` was supplied).
    """
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if target_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="pepe_eval_synth_")
        target_dir = Path(temp_dir.name)
    db_path = target_dir / "synthetic.db"
    random.seed(seed)
    await seed_diverse.run(str(db_path), SYNTHETIC_CHAT_ID, messages)
    return db_path, temp_dir
