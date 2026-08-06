"""Guard: no real chat identifier lives in the tracked files of the project.

The project already refuses to write a raw ``chat_id`` into a log line — that
is what ``app/log_masking.py`` exists for. The same identifier was nonetheless
committed into tests, fixtures, documentation and stored sweep results, i.e.
into exactly the artifacts that leave the machine together with the repository.

Recognition is by shape and reuses the sanitizer's own pattern, so a synthetic
identifier has to be declared here explicitly. That is the point: adding one
is a deliberate act, while pasting a real one from a live session is not.
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from app.log_masking import _CHAT_ID_IN_TEXT

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Synthetic identifiers used across tests and docs. Deliberately made of
# obvious digit runs so nobody mistakes one for a real chat.
ALLOWED_SYNTHETIC_CHAT_IDS = frozenset(
    {
        "-1001234567890",
        "-1009876543210",
        "-1002233445566",
        "-123456789",
    }
)

# Binary and generated content: matching bytes there say nothing about
# whether an identifier was published as text.
_SKIPPED_SUFFIXES = frozenset({".db", ".sqlite", ".png", ".jpg", ".jpeg", ".ico"})


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [_REPO_ROOT / line for line in result.stdout.splitlines() if line]


class TestNoRealChatIds(unittest.TestCase):
    def test_tracked_files_carry_only_synthetic_chat_ids(self) -> None:
        try:
            tracked = _tracked_files()
        except (OSError, subprocess.CalledProcessError) as exc:
            self.skipTest(f"git is unavailable: {exc}")

        offenders: dict[str, set[str]] = {}
        for path in tracked:
            if path.suffix.lower() in _SKIPPED_SUFFIXES or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            found = {
                match.group(0)
                for match in _CHAT_ID_IN_TEXT.finditer(text)
                if match.group(0) not in ALLOWED_SYNTHETIC_CHAT_IDS
            }
            if found:
                offenders[str(path.relative_to(_REPO_ROOT))] = found

        self.assertEqual(
            offenders,
            {},
            "Chat identifiers of a real shape must not be committed. Use one of "
            f"{sorted(ALLOWED_SYNTHETIC_CHAT_IDS)} or declare a new synthetic id here.",
        )

    def test_guard_recognizes_a_foreign_identifier(self) -> None:
        """The guard must actually catch something — a pattern that matches
        nothing would pass silently forever."""
        matches = [
            match.group(0)
            for match in _CHAT_ID_IN_TEXT.finditer("flood in chat -1007654321098")
        ]

        self.assertEqual(matches, ["-1007654321098"])
        self.assertNotIn(matches[0], ALLOWED_SYNTHETIC_CHAT_IDS)


if __name__ == "__main__":
    unittest.main()
