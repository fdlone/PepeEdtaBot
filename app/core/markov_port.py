"""Порт чтения модели: что генерации нужно от хранилища, и ничего сверх того.

Ядро генерации не должно знать, откуда берутся цепи. До появления этого порта
``app/core/markov.py`` и ``app/core/context_state_matcher.py`` импортировали
конкретный класс ``Database`` из инфраструктуры — направление зависимости,
обратное описанному в ``docs/ARCHITECTURE.md``.

Цена той зависимости была не абстрактной. Из-за возникающего цикла импортов
инфраструктура держала собственные копии набора знаков пунктуации, размера
контентной n-граммы и функции построения окон — с комментарием «importing it
would cycle». Любое изменение правил токенизации требовалось повторить в
нескольких местах, а рассинхронизация тихо развела бы индекс цитат с тем, что
видит генератор.

Порт описан по фактическому использованию: только те семь методов, которые
ядро действительно вызывает. Реализует его ``MarkovRepo`` — тот объект, который
и владеет соответствующим SQL; фасад ``Database`` отдаёт его через свойство
``markov``. Соответствие сигнатур проверяет ``mypy`` в строгом режиме.
"""
from __future__ import annotations

from typing import Protocol

from app.core.temporal import TransitionRow


class MarkovReadPort(Protocol):
    """Чтение цепи Маркова по чату."""

    async def get_starts(self, chat_id: int) -> list[tuple[str, str, int]]:
        """Стартовые пары ``(w1, w2, cnt)``, упорядоченные по ``(w1, w2)``.

        Упорядоченность — часть контракта: функции взвешенного выбора на неё
        опираются и сами не сортируют (см. ``weighted_start2_choice``).
        """
        ...

    async def get_starts3(self, chat_id: int) -> list[tuple[str, str, str, int]]:
        """Стартовые тройки ``(w1, w2, w3, cnt)``, упорядоченные по ключу."""
        ...

    async def get_start_if_exists(
        self, chat_id: int, w1: str, w2: str
    ) -> tuple[str, str, int] | None:
        """Стартовая пара, если такая есть в модели чата."""
        ...

    async def get_start3_if_exists(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> tuple[str, str, str, int] | None:
        """Стартовая тройка, если такая есть в модели чата."""
        ...

    async def get_transitions(
        self, chat_id: int, w1: str, w2: str
    ) -> list[TransitionRow]:
        """Продолжения состояния порядка 2, упорядоченные по токену.

        Строка несёт оба слоя (M2R-200): долгий счётчик и пару короткого слоя.
        Пул читается один раз на шаг, поэтому оба слоя приходят вместе — иначе
        включённый blend стоил бы второго запроса на каждый шаг прогулки.
        """
        ...

    async def get_transitions3(
        self, chat_id: int, w1: str, w2: str, w3: str
    ) -> list[TransitionRow]:
        """Продолжения состояния порядка 3, упорядоченные по токену."""
        ...

    async def get_reverse_transitions(
        self, chat_id: int, w2: str, w3: str
    ) -> list[TransitionRow]:
        """Токены-предшественники состояния ``(w2, w3)`` (M2R-400/410).

        Обратный переход — это та же строка прямой таблицы, прочитанная по
        последним двум колонкам; несёт оба слоя, как и прямой пул.
        """
        ...

    async def get_seed_forward(
        self, chat_id: int, token: str
    ) -> list[tuple[str, int]]:
        """Продолжения ``token`` как первого токена: ``(w2, суммарный count)``
        для бутстрапа хвоста seeded-кандидата (M2R-410)."""
        ...

    async def get_reverse_branch(self, chat_id: int, token: str) -> int:
        """Обратное ветвление сида: сколько разных токенов ему предшествуют."""
        ...

    async def get_states(
        self, chat_id: int, order: int
    ) -> list[tuple[tuple[str, ...], int]]:
        """Состояния цепи заданного порядка с суммарным числом переходов."""
        ...
