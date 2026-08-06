"""Guard: ядро генерации не зависит от инфраструктуры.

Правило записано в docs/ARCHITECTURE.md, но однажды уже было нарушено: модули
``app/core`` импортировали конкретный класс ``Database``. Плата за это лежала
не в архитектурной чистоте, а в коде — из-за возникшего цикла импортов
инфраструктура держала собственные копии набора пунктуации, размера контентной
n-граммы и функции построения окон, и рассинхронизация этих копий тихо развела
бы накопительный индекс цитат с тем, что видит генератор.

Правило, которое держится на внимательности, уже не удержалось однажды.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app"

# Слои, о которых ядро знать не должно: доступ к данным и всё, что выше него.
_FORBIDDEN_FOR_CORE = (
    "app.infrastructure",
    "app.repositories",
    "app.services",
    "app.handlers",
    "app.middlewares",
    "app.filters",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


class TestCoreDoesNotDependOnInfrastructure(unittest.TestCase):
    def test_core_modules_import_no_infrastructure(self) -> None:
        offenders: dict[str, set[str]] = {}
        for path in sorted((_APP / "core").rglob("*.py")):
            forbidden = {
                module
                for module in _imported_modules(path)
                if module.startswith(_FORBIDDEN_FOR_CORE)
            }
            if forbidden:
                offenders[str(path.relative_to(_APP.parent))] = forbidden

        self.assertEqual(
            offenders,
            {},
            "app/core обращается к данным через порт (app/core/markov_port.py), "
            "а не через конкретное хранилище",
        )

    def test_database_satisfies_the_read_port(self) -> None:
        """Реализация подходит порту без изменений сигнатур.

        Проверяется в рантайме, потому что Protocol без ``runtime_checkable``
        живёт только в проверке типов, а порт должен оставаться описанием
        фактического использования, а не отдельной абстракцией.
        """
        from app.core.markov_port import MarkovReadPort
        from app.infrastructure.database import Database

        for name in MarkovReadPort.__protocol_attrs__:  # type: ignore[attr-defined]
            self.assertTrue(
                callable(getattr(Database, name, None)),
                f"Database не реализует метод порта: {name}",
            )


if __name__ == "__main__":
    unittest.main()
