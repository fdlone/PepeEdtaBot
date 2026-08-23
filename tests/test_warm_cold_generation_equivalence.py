"""Тёплый инкрементальный кэш обязан генерировать как холодное чтение (O5).

Контракт §2 CLAUDE.md требует, чтобы `generation_hash` тёплого пути совпадал с
холодным. Проверял его до сих пор один гард — и по построению не мог:
`tools/generation_hash.py` строит генератор на **холодном** старте, то есть
сравнивает холодное с холодным. Прод же работает на тёплом кэше, который
пополняется по мере обучения.

Что это стоило: дефект `get_word_frequencies`/`pick_replacement` (карта §3.7а)
прожил до внешней разведки **при зелёном гарде**. Тёплый кэш дописывал
выученные слова в конец словаря, холодное чтение отдавало их отсортированными,
и слот-мутации в проде разыгрывались из иначе упорядоченного списка, чем под
хешем. Починка хеш не сдвинула — на холодном пути её эффект тождественен.

Соседние тесты (`test_incremental_*_match_full_rebuild` в
`test_learning_service.py`) писались ровно против этого класса и всё равно его
пропустили: они сверяют **состав** (`set`, `dict`), а в розыгрыш попадает ещё и
**порядок**.

Отсюда форма этого теста. Он не перечисляет тёплые структуры — перечисление и
подвело. Он сравнивает то единственное, что в итоге важно: **текст ответа** на
прогретом кэше после дообучения против текста на том же корпусе холодным
чтением.

**Что он ловит — проверено мутацией, а не заявлено.** Если в
``_fold_transition`` (`app/core/markov.py`) заменить вставку по порядку на
дописку в конец, тест падает и печатает оба разошедшихся ответа. До него кэши
распределений генератора — ``_cache3``, ``_cache2``, ``_cache_starts*`` и
индекс состояний — не были покрыты ничем: гард сравнивал холодное с холодным.

**Чего он не ловит на этом корпусе — тоже проверено, и это граница, а не
недосмотр.** Снятие сортировки в ``pick_replacement``
(`app/core/slot_mutation.py`) — тот самый исторический дефект O5 — тест не
роняет: словарь корпуса мал, морфологический фильтр оставляет замене нечего
выбирать, и порядок пула ни на что не влияет. Расширение словаря втрое
положения не изменило, зато стоило пяти секунд, поэтому корпус оставлен
компактным. Этот путь закрыт прицельно: ту же мутацию ловит
``test_warm_frequency_cache_draws_like_a_cold_read`` в
``test_learning_service.py``.

То есть тёплый контракт держат два теста с разной зоной ответственности, и
границу между ними стоит держать в уме, добавляя третий: сюда попадает то, что
доходит до розыгрыша **через прогулку по цепи**, туда — то, что разыгрывается
из словарей `LearningService`.

Две вещи, без которых тест был бы зелёным впустую, и обе проверяются явно:

1. **Прогрев.** `apply_learning_deltas` трогает только уже закэшированные
   ключи, так что без прогрева тёплого пути не существует вовсе и сверка
   выродилась бы в холодное против холодного — то самое, чем болен гард.
2. **Свёртка действительно произошла.** Прогрев кэширует лишь те состояния,
   через которые прошла прогулка. Если дообучение не задело ни одного из них,
   сравнивать нечего — и тест это утверждает, а не надеется.

Корпус собирается здесь, а не берётся из `tools/eval/synthetic.py`: тесту нужна
не похожесть на прод, а **ветвление** — состояния с несколькими
продолжениями, где порядок вообще может разойтись. Комбинаторный корпус даёт
его в четыре строки и на порядок дешевле: `IsolatedAsyncioTestCase` включает
asyncio-debug принудительно, а он умножает цену каждого `await` в `aiosqlite`
примерно на пятнадцать, так что 400-сообщенческий снапшот стоил бы сорока
секунд на одну сборку.
"""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import log_masking  # noqa: E402
from app.config.registry import RUNTIME_FIELDS  # noqa: E402
from app.core.markov import MarkovGenerator, content_tokens, tokenize  # noqa: E402
from app.core.response_generator import (  # noqa: E402
    CANDIDATE_TARGET,
    GenerationRequest,
    ResponseGenerator,
)
from app.core.text import sanitize_text  # noqa: E402
from app.infrastructure.database import Database  # noqa: E402
from app.services.learning_service import LearningService  # noqa: E402

CHAT_ID = -1001234567890  # синтетический, по инварианту tests/test_no_real_chat_ids.py

# Комбинаторный корпус: каждая тройка «когда + какая + что» получает четыре
# разных продолжения, то есть у состояния order-3 есть из чего выбирать. Без
# ветвления цепь детерминирована, и разошедшийся порядок нечем обнаружить.
_WHEN = ("сегодня", "вчера", "иногда")
_ADJ = ("хорошая", "странная")
_NOUN = ("погода", "история", "музыка")
_TAIL = ("радует всех", "бывает редко", "случается снова", "удивляет меня")

CORPUS = tuple(
    f"{when} {adj} {noun} {tail}"
    for when, adj, noun, tail in product(_WHEN, _ADJ, _NOUN, _TAIL)
)

# Дообучаемое между прогревом и сверкой — и одновременно источник контекстов.
# Это не экономия, а чувствительность: расходятся представления там, где дельта
# ложится в уже закэшированное распределение, поэтому и прогревать, и сверять
# надо на состояниях, которых дообучение касается.
#
# Первые два повторяют существующие продолжения (путь инкремента), два
# последних вводят новые токены, которые по алфавиту попадают **в середину**
# уже отсортированного списка (`кончается` между `бывает` и `радует`) — это
# путь вставки бисекцией, где неверный порядок и проявляется.
LEARNED_MESSAGES = (
    "сегодня хорошая погода радует всех",
    "вчера странная музыка бывает редко",
    "сегодня хорошая погода кончается быстро",
    "вчера странная музыка кончается тихо",
)

ROUNDS = 8


def _runtime_state() -> SimpleNamespace:
    """Дефолты реестра — та конфигурация, которую прод и везёт."""
    state = SimpleNamespace(
        **{spec.name: spec.parse(spec.default) for spec in RUNTIME_FIELDS}
    )
    state.recent_short_replies = {}
    state.recent_replies = {}
    return state


class TestWarmCacheGeneratesLikeAColdRead(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        log_masking.init_masking("secret-for-warm-cold-equivalence")
        temp_dir = tempfile.TemporaryDirectory(prefix="pepe_warm_cold_")
        self.addCleanup(temp_dir.cleanup)
        self.db = Database(str(Path(temp_dir.name) / "markov.db"))
        await self.db.init()
        self.addAsyncCleanup(self.db.close)

        seeding_generator = MarkovGenerator(self.db.markov)
        seeding = LearningService(self.db, seeding_generator)
        for text in CORPUS:
            await seeding.record_message(CHAT_ID, text, tokenize(text))

        self.contexts = [
            content_tokens(tokenize(text))[:8] for text in LEARNED_MESSAGES
        ]

    def _pipeline(self) -> tuple[ResponseGenerator, MarkovGenerator, LearningService]:
        """Свежая пара «генератор + сервис» на той же базе: кэши пустые."""
        generator = MarkovGenerator(self.db.markov)
        service = LearningService(self.db, generator)
        return (
            ResponseGenerator(
                generator=generator,
                learning_service=service,
                runtime_state=_runtime_state(),
            ),
            generator,
            service,
        )

    async def _generate(self, pipeline: ResponseGenerator) -> list[str]:
        """Партия генераций с фиксированными сидами — попарно сопоставимая.

        Свежий RNG на каждую генерацию: так сверяется порядок, в котором
        конвейер потребляет случайность, а не только итоговый текст.
        """
        produced: list[str] = []
        for index in range(ROUNDS):
            source = LEARNED_MESSAGES[index % len(LEARNED_MESSAGES)]
            result = await pipeline.generate_with_result(
                GenerationRequest(
                    chat_id=CHAT_ID,
                    context_tokens=self.contexts[index % len(self.contexts)],
                    seed=None,
                    current_message_normalized=sanitize_text(source).lower(),
                ),
                rng=random.Random(900_000 + index),
                candidate_target=CANDIDATE_TARGET,
            )
            produced.append(result.text or "")
        return produced

    @staticmethod
    def _snapshot(generator: MarkovGenerator) -> dict[object, tuple[object, ...]]:
        """Слепок кэшированных распределений — вместе с порядком строк."""
        return {
            key: tuple(rows)
            for cache in (
                generator._cache3,
                generator._cache2,
                generator._cache_starts3,
                generator._cache_starts2,
            )
            for key, rows in cache.items()
        }

    async def test_warm_incremental_cache_generates_like_a_cold_read(self) -> None:
        warm_pipeline, warm_generator, warm_service = self._pipeline()

        # 1. Прогрев: без него apply_learning_deltas не к чему применяться.
        await self._generate(warm_pipeline)
        before = self._snapshot(warm_generator)
        self.assertTrue(before, "прогрев не закэшировал ни одного распределения")

        # 2. Дообучение прод-путём: запись в SQL и свёртка кэша одним вызовом,
        #    с одним и тем же моментом наблюдения (M2R-210).
        for text in LEARNED_MESSAGES:
            await warm_service.record_message(
                CHAT_ID, text, tokenize(text), incremental_cache=True
            )

        # 3. Свёртка обязана была что-то тронуть, иначе сверять нечего.
        after = self._snapshot(warm_generator)
        folded = [key for key, rows in after.items() if before.get(key) != rows]
        self.assertTrue(
            folded,
            "дообучение не изменило ни одного кэшированного распределения — "
            "тёплого пути в этом прогоне не было, тест выродился",
        )

        # 4. Эталон: пустые кэши на той же базе, то есть полное чтение из SQL
        #    уже после записей выше.
        cold_pipeline, _cold_generator, _cold_service = self._pipeline()

        warm = await self._generate(warm_pipeline)
        cold = await self._generate(cold_pipeline)

        self.assertTrue(
            [text for text in warm if text.strip()],
            "все ответы пусты — сверять нечего",
        )
        first = next((i for i, (a, b) in enumerate(zip(warm, cold)) if a != b), None)
        self.assertIsNone(
            first,
            "тёплый путь разошёлся с холодным"
            + (
                f" на генерации {first}:\n  тёплый:   {warm[first]!r}\n"
                f"  холодный: {cold[first]!r}"
                if first is not None
                else ""
            ),
        )


if __name__ == "__main__":
    unittest.main()
