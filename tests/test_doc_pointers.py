"""Указатели `файл.py:строка` в документации обязаны попадать в код.

Навигация этого проекта построена на них: CLAUDE.md грузится в каждую сессию,
`GENERATION_MAP.md` — реестр дефектов с адресами. Указатель, съехавший на
соседнюю функцию, хуже отсутствующего: он молча приводит читателя не туда, а
выглядит точным. Ревью 2026-08-26 нашло `learning_service.py:504` (заявлен
`forget_chat`, фактически обрывок чужой сигнатуры) и
`generation_telemetry.py:201` (заявлен `note_seed_ranking`, фактически тело
`note_routes`) — оба разъехались правками, которые до документа не дошли.

Тест намеренно слабый: он не проверяет, что по адресу лежит *обещанная*
сущность, — только что там вообще есть код, а не пустая строка, закрывающая
скобка или граница докстринга. Этого хватает, чтобы сдвиг на десятки строк
падал в CI; смысловую сверку по-прежнему делает человек.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# `app/core/markov.py:743` и `markov.py:2031` — обе формы встречаются в доках.
_POINTER = re.compile(r"`?((?:[\w/]+/)?[\w_]+\.py):(\d+)")

# Строки, на которые указывать бессмысленно: по ним нельзя понять, о чём речь.
_EMPTY_TARGETS = {"", ")", "):", '"""', "'''", "(", "]", "}"}

# Храповик, как `fail_under` у покрытия: список только сокращается.
# Это указатели `GENERATION_MAP.md`, разъехавшиеся до появления гарда
# (найдены ревью 2026-08-26). Чинить их вслепую нельзя — карта — реестр
# дефектов, и указатель, переставленный по догадке, хуже явно протухшего:
# каждый требует сверки с тем, что описывает соседняя ячейка таблицы.
# Список — потолок, а не точное множество, и вычёркивается вручную при проходе
# по карте. Автоматической проверки «этот указатель починен» тут быть не может:
# правка соседнего кода сдвигает строки, и протухший адрес начинает попадать на
# какую-то строку сам собой. Что он попадает на *обещанную* сущность, тест не
# знает — он проверяет только, что там не пустота. Первая редакция гарда такую
# проверку содержала и сразу же дала ложную тревогу на правке O8.
_KNOWN_STALE = {
    "GENERATION_MAP.md: reply_pipeline.py:353 — пустая строка или скобка",
    "GENERATION_MAP.md: reply_pipeline.py:371 — пустая строка или скобка",
    "GENERATION_MAP.md: reply_pipeline.py:392 — пустая строка или скобка",
    "GENERATION_MAP.md: reply_pipeline.py:411 — пустая строка или скобка",
    "GENERATION_MAP.md: reply_pipeline.py:423 — пустая строка или скобка",
    "GENERATION_MAP.md: reply_pipeline.py:465 — пустая строка или скобка",
    "GENERATION_MAP.md: reply_pipeline.py:174 — пустая строка или скобка",
    "GENERATION_MAP.md: reply_pipeline.py:413 — пустая строка или скобка",
    "GENERATION_MAP.md: reply_pipeline.py:514 — пустая строка или скобка",
    "GENERATION_MAP.md: response_generator.py:61 — пустая строка или скобка",
    "GENERATION_MAP.md: response_generator.py:694 — пустая строка или скобка",
    "GENERATION_MAP.md: markov.py:2349 — пустая строка или скобка",
    "GENERATION_MAP.md: markov.py:631 — пустая строка или скобка",
}

_DOCS = [
    _ROOT / "CLAUDE.md",
    _ROOT / "docs" / "GENERATION_MAP.md",
    _ROOT / "docs" / "PRE_ROADMAP.md",
]


def _resolve(rel: str) -> Path | None:
    """Путь из указателя → файл в репозитории, если он однозначен."""
    direct = _ROOT / rel
    if direct.is_file():
        return direct
    matches = [p for p in _ROOT.glob(f"app/**/{Path(rel).name}") if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    return None


class TestDocPointers(unittest.TestCase):
    def test_pointers_land_on_code(self) -> None:
        broken: list[str] = []
        checked = 0
        for doc in _DOCS:
            if not doc.is_file():
                continue
            for raw in _POINTER.finditer(doc.read_text(encoding="utf-8")):
                rel, lineno = raw.group(1), int(raw.group(2))
                target = _resolve(rel)
                if target is None:
                    continue  # файл переименован или указатель на чужой проект
                lines = target.read_text(encoding="utf-8").splitlines()
                if not 1 <= lineno <= len(lines):
                    broken.append(f"{doc.name}: {rel}:{lineno} — за концом файла")
                    continue
                checked += 1
                if lines[lineno - 1].strip() in _EMPTY_TARGETS:
                    broken.append(
                        f"{doc.name}: {rel}:{lineno} — пустая строка или скобка"
                    )
        self.assertGreater(checked, 0, "ни одного указателя не проверено")
        fresh = [item for item in broken if item not in _KNOWN_STALE]
        self.assertEqual(
            fresh, [], "новые указатели разъехались с кодом:\n" + "\n".join(fresh)
        )


if __name__ == "__main__":
    unittest.main()
