from __future__ import annotations

import random
from collections.abc import Iterable
from datetime import datetime

# Fallback replies the bot sends when it cannot generate from the model.
# Pools are intentionally in the same ironic register as the pivo templates so
# the "service" voice does not break character. Selection avoids the phrases
# used most recently in the chat (see pick_fallback_phrase).

NOT_ENOUGH_DATA_PHRASES: tuple[str, ...] = (
    "Пока мало материала, поболтайте ещё 🙂",
    "Мне нужно больше ваших сообщений. Да, это официальный повод пофлудить.",
    "Словарный запас ещё не набрался. Продолжайте в том же духе.",
    "Я пока только слушаю. Говорите, говорите — всё записывается.",
    "Рано. Наговорите мне хотя бы на пару предложений.",
    "База знаний размером с чек из магазина. Пополните её.",
    "Учусь на ваших сообщениях, а их подозрительно мало.",
    "Пока нечем ответить — вы слишком мало наболтали.",
    "Дайте мне ещё материала, и я начну отвечать как положено.",
    "Молчу не из вредности: корпус пустой. Исправляйте.",
    "Мне буквально не из чего собрать ответ. Флудите смелее.",
    "Ещё немного ваших гениальных мыслей — и я заговорю.",
)

GENERATION_FAILED_PHRASES: tuple[str, ...] = (
    "Собираю мысли... Напишите ещё пару сообщений 🙂",
    "Мысль была, но убежала. Напишите что-нибудь ещё.",
    "Слова кончились на самом интересном месте.",
    "Хотел ответить умно, не получилось. Попробуйте ещё раз.",
    "Генератор гениальных ответов временно перегрелся.",
    "Ответ не собрался. Считайте, что я многозначительно промолчал.",
    "Тут даже мне нечего сказать, а это показатель.",
    "Сформулировать не вышло. Сделаю вид, что так и задумано.",
    "Мысли есть, слов нет. Бывает и у людей.",
    "Ответ застрял где-то между токенами. Спросите ещё раз.",
    "Пока думал, забыл, что хотел сказать.",
    "Красивого ответа не вышло, а некрасивый я вам не покажу.",
)


# S4 temporal modifier: extra fallbacks that fit the small-hours mood. They are
# merged into the base pool only late at night (see late_night_pool) so the
# neutral pools stay the default and no bucket is ever empty.
LATE_NIGHT_FALLBACK_PHRASES: tuple[str, ...] = (
    "Ночь на дворе, а вы всё пишете. Я, конечно, тоже не сплю.",
    "В такое время связных мыслей нет ни у меня, ни у вас.",
    "Поздно уже. Давайте я сделаю вид, что глубокомысленно молчу.",
    "Ночью даже мой генератор работает в режиме энергосбережения.",
)


# M1 mood modifier: punchier fallbacks for a heated chat. Merged into the base
# pool only when the mood is ``heated`` (see mood_fallback_pool); every other
# mood uses the neutral pools, so no pool is ever left empty.
HEATED_FALLBACK_PHRASES: tuple[str, ...] = (
    "Так, стоп, я не успеваю за этим цирком.",
    "Слишком жарко тут у вас, дайте отдышаться.",
    "Вы там разошлись, а я ещё думаю.",
    "Не гоните, мне бы мысль поймать в этом бардаке.",
)


def is_late_night(now: datetime) -> bool:
    """True for the small hours (00:00–05:59) where late-night flavor applies."""
    return 0 <= now.hour < 6


def mood_fallback_pool(base: tuple[str, ...], mood: str | None) -> tuple[str, ...]:
    """Return ``base`` extended with heated phrases when the chat mood is heated."""
    if mood == "heated":
        return base + HEATED_FALLBACK_PHRASES
    return base


def late_night_pool(base: tuple[str, ...], now: datetime | None) -> tuple[str, ...]:
    """Return ``base`` extended with late-night phrases when ``now`` is late night."""
    if now is not None and is_late_night(now):
        return base + LATE_NIGHT_FALLBACK_PHRASES
    return base


def pick_fallback_phrase(
    pool: tuple[str, ...],
    recent: Iterable[str] = (),
    rng: random.Random | None = None,
) -> str:
    """Pick a phrase from ``pool``, avoiding recently used ones when possible.

    Falling back to ``random.choice`` (not a fresh ``Random``) keeps the
    function patchable in tests, matching the pivo builder convention.
    """
    chooser = random.choice if rng is None else rng.choice
    recent_set = set(recent)
    fresh = [phrase for phrase in pool if phrase not in recent_set]
    return chooser(fresh if fresh else list(pool))
