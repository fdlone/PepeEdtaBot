from __future__ import annotations

import unittest
from types import SimpleNamespace

from aiogram.types import MessageEntity, User

from app.services.pivo_parser import MAX_TARGET_CHARS, parse_pivo_command


def _message(
    text: str,
    *,
    entities: list[MessageEntity] | None = None,
) -> object:
    return SimpleNamespace(text=text, entities=entities)


class TestPivoParser(unittest.TestCase):
    def test_no_arguments(self) -> None:
        args = parse_pivo_command(_message("/pivo"))  # type: ignore[arg-type]

        self.assertIsNone(args.planned_time)
        self.assertIsNone(args.target)
        self.assertEqual(args.explicit_mentions, ())

    def test_only_time(self) -> None:
        args = parse_pivo_command(_message("/pivo 20:00"))  # type: ignore[arg-type]

        self.assertEqual(args.planned_time, "20:00")
        self.assertIsNone(args.target)
        self.assertEqual(args.explicit_mentions, ())

    def test_only_target(self) -> None:
        args = parse_pivo_command(_message("/pivo watch movie"))  # type: ignore[arg-type]

        self.assertIsNone(args.planned_time)
        self.assertEqual(args.target, "watch movie")
        self.assertEqual(args.explicit_mentions, ())

    def test_only_plain_mention(self) -> None:
        args = parse_pivo_command(_message("/pivo @friend_user"))  # type: ignore[arg-type]

        self.assertIsNone(args.planned_time)
        self.assertIsNone(args.target)
        self.assertEqual(args.explicit_mentions, ("@friend_user",))

    def test_short_at_word_is_target_not_mention(self) -> None:
        args = parse_pivo_command(_message("/pivo @all"))  # type: ignore[arg-type]

        self.assertIsNone(args.planned_time)
        self.assertEqual(args.target, "@all")
        self.assertEqual(args.explicit_mentions, ())

    def test_plain_words_are_target_not_mentions(self) -> None:
        args = parse_pivo_command(_message("/pivo all contact mail"))  # type: ignore[arg-type]

        self.assertIsNone(args.planned_time)
        self.assertEqual(args.target, "all contact mail")
        self.assertEqual(args.explicit_mentions, ())

    def test_at_word_starting_with_digit_is_target_not_mention(self) -> None:
        args = parse_pivo_command(_message("/pivo @1friend"))  # type: ignore[arg-type]

        self.assertIsNone(args.planned_time)
        self.assertEqual(args.target, "@1friend")
        self.assertEqual(args.explicit_mentions, ())

    def test_time_and_target(self) -> None:
        args = parse_pivo_command(_message("/pivo today 21:00 watch movie"))  # type: ignore[arg-type]

        self.assertEqual(args.planned_time, "today 21:00")
        self.assertEqual(args.target, "watch movie")
        self.assertEqual(args.explicit_mentions, ())

    def test_time_with_preposition_and_target(self) -> None:
        args = parse_pivo_command(  # type: ignore[arg-type]
            _message("/pivo сегодня в 21:30 watch movie")
        )

        self.assertEqual(args.planned_time, "сегодня в 21:30")
        self.assertEqual(args.target, "watch movie")
        self.assertEqual(args.explicit_mentions, ())

    def test_english_time_with_preposition_and_target(self) -> None:
        args = parse_pivo_command(  # type: ignore[arg-type]
            _message("/pivo tomorrow at 21:30 watch movie")
        )

        self.assertEqual(args.planned_time, "tomorrow at 21:30")
        self.assertEqual(args.target, "watch movie")
        self.assertEqual(args.explicit_mentions, ())

    def test_time_and_mention(self) -> None:
        args = parse_pivo_command(_message("/pivo завтра вечером @friend_user"))  # type: ignore[arg-type]

        self.assertEqual(args.planned_time, "завтра вечером")
        self.assertIsNone(args.target)
        self.assertEqual(args.explicit_mentions, ("@friend_user",))

    def test_target_and_mention(self) -> None:
        args = parse_pivo_command(_message("/pivo board games @friend_user"))  # type: ignore[arg-type]

        self.assertIsNone(args.planned_time)
        self.assertEqual(args.target, "board games")
        self.assertEqual(args.explicit_mentions, ("@friend_user",))

    def test_time_target_and_mention(self) -> None:
        args = parse_pivo_command(  # type: ignore[arg-type]
            _message("/pivo tomorrow evening board games @friend_user @second_user")
        )

        self.assertEqual(args.planned_time, "tomorrow evening")
        self.assertEqual(args.target, "board games")
        self.assertEqual(args.explicit_mentions, ("@friend_user", "@second_user"))

    def test_time_in_middle_is_target(self) -> None:
        args = parse_pivo_command(_message("/pivo movie at 20:00"))  # type: ignore[arg-type]

        self.assertIsNone(args.planned_time)
        self.assertEqual(args.target, "movie at 20:00")

    def test_leading_date_word_is_time(self) -> None:
        args = parse_pivo_command(_message("/pivo today movie"))  # type: ignore[arg-type]

        self.assertEqual(args.planned_time, "today")
        self.assertEqual(args.target, "movie")

    def test_russian_leading_date_word_is_time(self) -> None:
        args = parse_pivo_command(_message("/pivo завтра сосать бибу"))  # type: ignore[arg-type]

        self.assertEqual(args.planned_time, "завтра")
        self.assertEqual(args.target, "сосать бибу")

    def test_long_target_is_truncated(self) -> None:
        """Повод режется по длине: без потолка отправка падала бы всегда.

        Сообщение Telegram — до 4096 символов, шаблон /pivo добавляет к поводу
        ещё несколько сотен. Ввод в 4000 символов делал вызов заведомо
        недоставляемым, а квота при этом возвращалась — приём был повторяем.
        """
        long_target = "го в дотку " * 400

        args = parse_pivo_command(_message(f"/pivo {long_target}"))  # type: ignore[arg-type]

        self.assertIsNotNone(args.target)
        assert args.target is not None
        self.assertLessEqual(len(args.target), MAX_TARGET_CHARS)
        # Режем по границе слова — обрубок посреди слова читается как поломка.
        self.assertFalse(args.target.endswith("го в дот"))

    def test_short_target_is_untouched(self) -> None:
        args = parse_pivo_command(_message("/pivo смотреть фильм"))  # type: ignore[arg-type]

        self.assertEqual(args.target, "смотреть фильм")

    def test_target_without_spaces_is_truncated_by_length(self) -> None:
        args = parse_pivo_command(_message("/pivo " + "я" * 500))  # type: ignore[arg-type]

        assert args.target is not None
        self.assertEqual(len(args.target), MAX_TARGET_CHARS)

    def test_telegram_text_mention_without_username(self) -> None:
        text = "/pivo 20:00 Friend"
        user = User(id=42, is_bot=False, first_name="No", last_name="Username")
        entity = MessageEntity(type="text_mention", offset=12, length=6, user=user)

        args = parse_pivo_command(_message(text, entities=[entity]))  # type: ignore[arg-type]

        self.assertEqual(args.planned_time, "20:00")
        self.assertIsNone(args.target)
        self.assertEqual(
            args.explicit_mentions,
            ('<a href="tg://user?id=42">No Username</a>',),
        )


if __name__ == "__main__":
    unittest.main()
