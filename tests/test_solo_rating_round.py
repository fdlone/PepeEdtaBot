"""Соло-раунд оценки связности (M3R-020): подготовка, подсчёт, гейт фазы 9."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from tools.eval.report import _connectedness_part
from tools.solo_rating_round import (
    build_round,
    main,
    parse_answers,
    score,
    score_round,
)

THRESHOLDS: dict[str, Any] = {
    "phase9_interp": {
        "manual_rated_min": 4,
        "manual_repeat_share_min": 0.20,
        "manual_self_consistency_min": 0.80,
        "manual_decoy_detected_min": 0.80,
    }
}


def _replies(arm: str, count: int) -> list[str]:
    return [f"{arm} ответ номер {index} про пиво" for index in range(count)]


class BuildRoundTest(unittest.TestCase):
    def _round(self, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        replies = {"C0": _replies("C0", 8), "C4": _replies("C4", 8)}
        params: dict[str, Any] = {
            "rated_min": 4,
            "repeat_share": 0.25,
            "seed": 7,
        }
        params.update(kwargs)
        return build_round(replies, **params)

    def test_list_is_blind_and_key_carries_the_correspondence(self) -> None:
        listing, key = self._round()

        # Ни один арм не назван в раздаваемом списке.
        body = listing.split("\n\n", 1)[1]
        self.assertNotIn("арм", body.lower())
        for position, meta in key["positions"].items():
            self.assertIn(meta["arm"], {"C0", "C4"})
            self.assertIn("decoy", meta)
            self.assertIn(f"\n{position}. ", "\n" + body)

    def test_repeats_do_not_sit_next_to_each_other(self) -> None:
        """Иначе согласие меряет память, а не устойчивость суждения (D3)."""
        _listing, key = self._round()
        order = [
            key["positions"][str(position)]["item"]
            for position in range(1, key["counts"]["positions"] + 1)
        ]
        self.assertGreater(key["counts"]["repeat_pairs"], 0)
        for first, second in zip(order, order[1:]):
            self.assertNotEqual(first, second)

    def test_decoys_keep_the_words_and_lose_the_order(self) -> None:
        """Декой — тот же ответ с переставленными токенами (D2)."""
        listing, key = self._round()
        texts = {
            position.split(". ", 1)[0]: position.split(". ", 1)[1]
            for position in listing.splitlines()
            if ". " in position and position.split(". ", 1)[0].isdigit()
        }
        decoys = [
            texts[position]
            for position, meta in key["positions"].items()
            if meta["decoy"]
        ]
        self.assertTrue(decoys)
        sources = {
            tuple(sorted(reply.split()))
            for reply in _replies("C0", 8) + _replies("C4", 8)
        }
        for decoy in decoys:
            # Тот же набор слов, что у какого-то настоящего ответа...
            self.assertIn(tuple(sorted(decoy.split())), sources)
            # ...но не он сам.
            self.assertNotIn(decoy, _replies("C0", 8) + _replies("C4", 8))

    def test_short_sample_is_named_in_the_key(self) -> None:
        _listing, key = build_round(
            {"C0": _replies("C0", 8), "C4": _replies("C4", 2)},
            rated_min=4,
            repeat_share=0.25,
            seed=7,
        )

        self.assertEqual(key["below_minimum"], ["C4"])

    def test_round_is_reproducible_for_a_seed(self) -> None:
        self.assertEqual(self._round()[0], self._round()[0])


class ParseAnswersTest(unittest.TestCase):
    def test_scores_and_classes(self) -> None:
        self.assertEqual(
            parse_answers("1:3 2:1/F3_generic 3:2"),
            {1: (3, None), 2: (1, "F3_generic"), 3: (2, None)},
        )

    def test_unknown_class_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_answers("1:1/F42_нечто")
        self.assertIn("unknown failure class", str(caught.exception))


def _key(positions: dict[int, tuple[str, bool, str]]) -> dict[str, Any]:
    return {
        "seed": 7,
        "context_mode": "ctx",
        "positions": {
            str(position): {"arm": arm, "decoy": decoy, "item": item}
            for position, (arm, decoy, item) in positions.items()
        },
    }


class ScoreRoundTest(unittest.TestCase):
    """Раунд из 4 позиций на арм, двух декоев и двух повторов."""

    def _valid_round(self) -> tuple[dict[str, Any], str]:
        positions = {}
        answers = []
        position = 1
        for arm, scores in (("C0", [3, 3, 3, 1]), ("C4", [3, 3, 2, 1])):
            for index, value in enumerate(scores):
                positions[position] = (arm, False, f"{arm}-{index}")
                answers.append(f"{position}:{value}")
                position += 1
        for arm in ("C0", "C4"):
            positions[position] = (arm, True, f"{arm}-decoy-0")
            answers.append(f"{position}:1")
            position += 1
        # Повторы (позиции 11-13): те же элементы, та же оценка.
        for item in ("C0-0", "C4-0", "C0-1"):
            positions[position] = (item.split("-")[0], False, item)
            answers.append(f"{position}:3")
            position += 1
        return _key(positions), " ".join(answers)

    def test_valid_round_reports_shares_per_arm(self) -> None:
        key, answers = self._valid_round()

        aggregate = score_round(key, parse_answers(answers), THRESHOLDS)

        self.assertTrue(aggregate["valid"], aggregate["invalid_reasons"])
        # По РАЗЛИЧНЫМ ответам: повтор не даёт ответу двойной голос.
        self.assertEqual(aggregate["arms"]["C0"]["rated"], 4)
        self.assertEqual(aggregate["arms"]["C0"]["connected_share"], 0.75)
        self.assertEqual(aggregate["arms"]["C4"]["connected_share"], 0.75)
        self.assertEqual(aggregate["self_agreement"], 1.0)
        self.assertEqual(aggregate["decoy_detected_share"], 1.0)

    def test_unstable_rater_invalidates_the_round(self) -> None:
        key, answers = self._valid_round()
        # Все повторы получают другую оценку, чем первое предъявление.
        answers = answers.replace("11:3 12:3 13:3", "11:1 12:1 13:1")

        aggregate = score_round(key, parse_answers(answers), THRESHOLDS)

        self.assertFalse(aggregate["valid"])
        self.assertTrue(
            any("self-agreement" in reason for reason in aggregate["invalid_reasons"])
        )

    def test_missed_decoys_invalidate_the_round_with_their_own_reason(self) -> None:
        key, answers = self._valid_round()
        answers = answers.replace("9:1 10:1", "9:3 10:3")

        aggregate = score_round(key, parse_answers(answers), THRESHOLDS)

        self.assertFalse(aggregate["valid"])
        reasons = " ".join(aggregate["invalid_reasons"])
        self.assertIn("decoys caught", reasons)
        self.assertNotIn("self-agreement", reasons)

    def test_aggregate_holds_numbers_only(self) -> None:
        key, answers = self._valid_round()

        aggregate = score_round(key, parse_answers(answers), THRESHOLDS)

        # Ни одного текста ответа: агрегат — единственный файл раунда, которому
        # разрешено попасть в репозиторий.
        dumped = json.dumps(aggregate, ensure_ascii=False)
        self.assertNotIn("ответ номер", dumped)
        self.assertNotIn("пиво", dumped)

    def test_failure_classes_are_counted(self) -> None:
        key, answers = self._valid_round()
        answers = answers.replace("4:1", "4:1/F3_generic")

        aggregate = score_round(key, parse_answers(answers), THRESHOLDS)

        self.assertEqual(aggregate["failure_classes"], {"F3_generic": 1})


class ConnectednessConditionTest(unittest.TestCase):
    """Условие 6 гейта фазы 9 считается, а не отсутствует безусловно."""

    def _part(self, solo: dict[str, Any] | None) -> tuple[list[str], list[str], list[str]]:
        parts: list[str] = []
        missing: list[str] = []
        failures: list[str] = []
        _connectedness_part(
            "C0", "C4", solo, -0.10, parts=parts, missing=missing, failures=failures
        )
        return parts, missing, failures

    @staticmethod
    def _aggregate(c0: float, c4: float, valid: bool = True) -> dict[str, Any]:
        return {
            "valid": valid,
            "invalid_reasons": [] if valid else ["self-agreement 0.50 below 0.80"],
            "arms": {
                "C0": {"rated": 30, "connected": 0, "connected_share": c0},
                "C4": {"rated": 30, "connected": 0, "connected_share": c4},
            },
        }

    def test_arm_within_the_floor_passes(self) -> None:
        parts, missing, failures = self._part(self._aggregate(0.80, 0.75))

        self.assertEqual(missing, [])
        self.assertEqual(failures, [])
        self.assertIn("connected 75% vs 80%", parts[0])

    def test_arm_losing_connectedness_fails(self) -> None:
        _parts, missing, failures = self._part(self._aggregate(0.80, 0.60))

        self.assertEqual(missing, [])
        self.assertTrue(failures)

    def test_no_round_is_missing_data_not_a_failure(self) -> None:
        _parts, missing, failures = self._part(None)

        self.assertEqual(failures, [])
        self.assertIn("not conducted", missing[0])

    def test_invalid_round_is_missing_data_and_names_the_reason(self) -> None:
        _parts, missing, failures = self._part(self._aggregate(0.80, 0.60, valid=False))

        self.assertEqual(failures, [])
        self.assertIn("self-agreement", missing[0])


class SyntheticSnapshotTest(unittest.IsolatedAsyncioTestCase):
    """Шов «прогон → раунд» на синтетическом снапшоте.

    Чистые тесты выше проверяют устройство раунда, но не то, что подготовка
    берёт из прогона именно текст ответа и именно идентификатор арма. Ошибка
    ровно здесь ничем другим не ловится.
    """

    async def test_round_builds_from_a_real_run(self) -> None:
        from tools.eval.config import load_matrix
        from tools.eval.prompts import generate_prompts
        from tools.eval.run import run_matrix
        from tools.eval.synthetic import SYNTHETIC_CHAT_ID, build_synthetic_snapshot

        db_path, temp_dir = await build_synthetic_snapshot(messages=80)
        try:
            runs, _skipped = await run_matrix(
                db_source=db_path,
                chat_id=SYNTHETIC_CHAT_ID,
                configs={"C0": load_matrix()["C0"]},
                prompt_set=generate_prompts(
                    db_path, chat_id=SYNTHETIC_CHAT_ID, seed=42, snapshot_label="t"
                ),
                seeds=(42,),
                generations=16,
            )
            replies_by_arm = {
                run.config_id: [r.reply_text for r in run.records if r.success]
                for run in runs.values()
                if run.shared_with is None
            }
            self.assertTrue(replies_by_arm["C0"])

            listing, key = build_round(
                replies_by_arm, rated_min=4, repeat_share=0.25, seed=42
            )

            self.assertEqual({"C0"}, {m["arm"] for m in key["positions"].values()})
            self.assertGreater(key["counts"]["decoys"], 0)
            texts = {
                line.split(". ", 1)[0]: line.split(". ", 1)[1]
                for line in listing.splitlines()
                if ". " in line and line.split(". ", 1)[0].isdigit()
            }
            produced = set(replies_by_arm["C0"])
            for position, meta in key["positions"].items():
                if not meta["decoy"]:
                    self.assertIn(texts[position], produced)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


class CommandLineTest(unittest.TestCase):
    def test_score_prints_counts_only_and_writes_the_aggregate(self) -> None:
        import tempfile

        key_positions = {
            1: ("C0", False, "C0-0"),
            2: ("C0", False, "C0-0"),
            3: ("C0", True, "C0-decoy-0"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "rating_key.json").write_text(
                json.dumps(_key(key_positions)), encoding="utf-8"
            )
            answers_path = out_dir / "answers.txt"
            answers_path.write_text("1:3 2:3 3:1", encoding="utf-8")

            class _Args:
                thresholds = str(
                    Path(__file__).resolve().parents[1]
                    / "tools"
                    / "eval"
                    / "eval_thresholds.yaml"
                )
                answers = str(answers_path)

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                score(_Args(), out_dir)  # type: ignore[arg-type]

            printed = buffer.getvalue()
            self.assertNotIn("пиво", printed)
            self.assertIn("C0: rated 1", printed)
            aggregate = json.loads(
                (out_dir / "solo_rating.json").read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["decoys"], 1)

    def test_score_without_answers_is_an_error(self) -> None:
        import sys
        from unittest import mock

        with mock.patch.object(
            sys, "argv", ["solo_rating_round", "score", "--label", "x"]
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
