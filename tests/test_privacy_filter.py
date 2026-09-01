from __future__ import annotations

import unittest

from app.core.privacy_filter import (
    redact_emails,
    redact_phones,
    redact_secrets,
    redact_sensitive_data,
)


class TestEmailRedaction(unittest.TestCase):
    def test_redacts_email_in_russian_context(self) -> None:
        result = redact_emails("Напиши мне на User.Name+chat@example.co.uk завтра")

        self.assertEqual(result.split(), ["Напиши", "мне", "на", "завтра"])

    def test_redacts_email_in_english_context(self) -> None:
        result = redact_emails("Contact support_team@example.com for help")

        self.assertEqual(result.split(), ["Contact", "for", "help"])

    def test_redacts_tagged_and_dotted_local_parts(self) -> None:
        self.assertEqual(redact_emails("first.last+tag@sub.example.org").strip(), "")

    def test_preserves_host_without_dot(self) -> None:
        text = "user@localhost"

        self.assertEqual(redact_emails(text), text)


class TestPhoneRedaction(unittest.TestCase):
    def test_redacts_supported_phone_formats(self) -> None:
        phones = (
            "+380 44 123 45 67",
            "+380441234567",
            "044 123 45 67",
            "0441234567",
            "+7 999 123-45-67",
            "8 (999) 123-45-67",
            "+44 20 7946 0958",
        )

        for phone in phones:
            with self.subTest(phone=phone):
                self.assertEqual(redact_phones(phone).strip(), "")

    def test_preserves_non_phone_numbers(self) -> None:
        values = (
            "2026",
            "1500",
            "21.06.2026",
            "192.168.0.1",
            "1234567890",
            "оценки 5 4 5 4 5 4 5 4 5 4 отлично",
            "12 34 56 78 90",
            "числа 1 2 3 4 5 6 7 8 9 10",
        )

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(redact_phones(value), value)

    def test_redacts_space_separated_ukrainian_national_number(self) -> None:
        self.assertEqual(redact_phones("044 123 45 67").strip(), "")

    def test_keeps_surrounding_words(self) -> None:
        result = redact_phones("Звони +380 44 123 45 67 вечером")

        self.assertEqual(result.split(), ["Звони", "вечером"])


class TestSecretRedaction(unittest.TestCase):
    def test_redacts_known_secret_forms(self) -> None:
        secrets = (
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcd_12345",
            "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd",
        )

        for secret in secrets:
            with self.subTest(secret=secret):
                self.assertEqual(redact_secrets(secret).strip(), "")

    def test_redacts_generic_hex_secret(self) -> None:
        secret = "0123456789abcdef0123456789ABCDEF"

        self.assertEqual(redact_secrets(secret).strip(), "")

    def test_redacts_high_entropy_mixed_class_secret(self) -> None:
        secret = "aB3_dE5-fG7_hJ9-kL2_mN4-pQ6"

        self.assertEqual(redact_secrets(secret).strip(), "")

    def test_redacts_generic_secret_before_terminal_punctuation(self) -> None:
        secret = "aB3_dE5-fG7_hJ9-kL2_mN4-pQ6"
        result = redact_secrets(f"вот токен: {secret}.")

        self.assertNotIn(secret, result)
        self.assertIn("вот токен:", result)

    def test_redacts_generic_secret_adjacent_to_punctuation(self) -> None:
        secret = "aB3_dE5-fG7_hJ9-kL2_mN4-pQ6"
        for wrapped in (
            f"({secret})",
            f'"{secret}"',
            f"«{secret}»",
            f"{secret},",
        ):
            with self.subTest(wrapped=wrapped):
                self.assertNotIn(secret, redact_secrets(wrapped))

    def test_redacts_hex_secret_adjacent_to_punctuation(self) -> None:
        secret = "0123456789abcdef0123456789ABCDEF"
        for wrapped in (f"{secret}.", f"({secret})", f'"{secret}"'):
            with self.subTest(wrapped=wrapped):
                self.assertNotIn(secret, redact_secrets(wrapped))

    def test_preserves_uuid(self) -> None:
        value = "123e4567-e89b-12d3-a456-426614174000"

        self.assertEqual(redact_secrets(value), value)

    def test_preserves_long_non_ascii_word(self) -> None:
        value = "сверхдлинноерусскоесловобезсекретов"

        self.assertEqual(redact_secrets(value), value)

    def test_preserves_readable_english_slug(self) -> None:
        value = "super-long-readable-phrase"

        self.assertEqual(redact_secrets(value), value)

    def test_preserves_low_entropy_repeated_string(self) -> None:
        value = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"

        self.assertEqual(redact_secrets(value), value)


class TestSensitiveDataRedaction(unittest.TestCase):
    def test_composes_detectors_and_keeps_surrounding_words(self) -> None:
        text = (
            "Email user@example.com phone +380 44 123 45 67 "
            "key 0123456789abcdef0123456789abcdef done"
        )

        self.assertEqual(
            redact_sensitive_data(text).split(),
            ["Email", "phone", "key", "done"],
        )

    def test_pii_only_message_has_no_non_whitespace_content(self) -> None:
        text = "user@example.com +380441234567"

        self.assertEqual(redact_sensitive_data(text).split(), [])

    def test_redaction_is_deterministic(self) -> None:
        text = (
            "user@example.com +7 999 123-45-67 "
            "aB3_dE5-fG7_hJ9-kL2_mN4-pQ6"
        )

        self.assertEqual(redact_sensitive_data(text), redact_sensitive_data(text))


class TestLongSecretsHaveNoUpperBound(unittest.TestCase):
    """У редакции секретов нет верхней границы длины.

    Пока в общем детекторе стояло `{24,128}`, отсечка означала не «частичное
    совпадение», а полную слепоту: для токена длиннее 128 символов движок не
    мог закрыть lookahead ни при какой длине из диапазона, а lookbehind не
    давал сдвинуться внутрь токена. Совпадений не было вообще, и секрет уходил
    в `messages.normalized_text` целиком — вопреки контракту `sanitize_text`
    («корпус, контекст ответа и проверки дословных копий не видят
    чувствительных данных»). Найдено ревью 2026-08-26; в тестах до него не
    было ни одного кейса длиннее ~40 символов.
    """

    @staticmethod
    def _token(length: int) -> str:
        # Высокая энтропия и три класса символов: детектор решает по ним, а не
        # по длине. Детерминированно, чтобы падение было воспроизводимым.
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(alphabet[(index * 37 + 11) % len(alphabet)] for index in range(length))

    def test_secret_longer_than_the_old_ceiling_is_redacted(self) -> None:
        for length in (128, 129, 240, 1000):
            with self.subTest(length=length):
                token = self._token(length)
                redacted = redact_sensitive_data(f"ключ {token} конец")
                self.assertNotIn(token, redacted, f"секрет длиной {length} прошёл насквозь")

    def test_ordinary_words_survive(self) -> None:
        text = "привет как дела сегодня вечером"
        self.assertEqual(redact_sensitive_data(text), text)
