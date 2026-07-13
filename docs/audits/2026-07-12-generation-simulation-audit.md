# 2026-07-12 — Симуляционный аудит генерации и упрощение пайплайна

## Метод

Инструментированная симуляция диалогов через реальный хендлер
`on_text_message` (временная SQLite-БД, реальные `LearningService` /
`MarkovGenerator` / `RuntimeState` с прод-дефолтами registry, виртуальные
часы): 2 сценария × 900 сообщений (36 диалогов × 25, 3 чата), «медленный»
(8–120 с между сообщениями) и «бурный» (0.5–5 с). ~270 ответов бота и ~2000
сгенерированных кандидатов на сценарий. Дополнительно — контрфактический
анализ скорера (для каждого выбора из N кандидатов: как часто обнуление
компонента меняет argmax) и A/B прогоны эталонного eval для
prefix/stem-матчинга.

## Ключевые находки

| # | Находка | Данные | Решение |
|---|---|---|---|
| SIM-1 | Order-1 цепь мертва по конфигу, но `transitions1` пишется на каждое сообщение | 0 срабатываний из ~4000 кандидатов (дефолт `BACKOFF_MIN_ORDER=2` с 2026-07-09); единственный читатель — знаменатель биграмм hot-ngrams | Удалить (PR #70): миграция 013, знаменатель — SUM по `transitions` |
| SIM-2 | `fuzzy_context_prefix` никогда не включался; дублирует стеммер (PR #66) своей эвристикой | Замеры PR #48: prefix резолвит 22–24% контекстов; на малом корпусе душится гардом `transition_count>=2` | Переписать на `stem_token` и включить по дефолту (PR #73): stem резолвит 22% при том же качестве |
| SIM-3 | `lexical_diversity` и скорерный `repetition_penalty` меряют одно и то же | флипы argmax ~3% каждый; математически 1−diversity == доля повторов токенов | Слить в один компонент (PR #77), ранжирование не меняется, бейзлайны байт-идентичны |
| SIM-4 | `coherence_penalty` — шум после SIM-1 | флипы ~0.4%; остаток — 0.10 на ~2% кандидатов (order-2) | Удалить (PR #77) |
| SIM-5 | Три staleness-гейта почти не стреляют | echo current: 0, short anti-repeat: 0, full repeat: 1 на ~4000 | Схлопнуть в `repeats_recent_chat_content` (PR #77); verbatim-гейт отдельно (ведёт к extension) |
| SIM-6 | L1 hot-ngram и L3 rare events невидимы на дефолтных шансах | hot-ngram: 1–2 lookup на 900 сообщений; verdict/CAPS/double: ~1 на 200 ответов | Поднять шансы 0.05→0.25 / 0.005→0.03 / 0.03→0.05, кап 3/день без изменений (PR #71) |
| SIM-7 | Рабочие лошадки скоринга — verbatim_penalty и IDF context_relevance | флипы 37% и 32% соответственно; verbatim-extension на 47% принятых кандидатов (малый корпус завышает) | Без изменений |
| SIM-8 | `strip_leading_bot_vocative` не срезает обращение без разделителя | бот выучил и говорит «пепе сегодня был в банке...» (`learning.py`, `_LEADING_VOCATIVE_RE` требует `[,:;—–-]`) | Closed — PR #75 |
| SIM-9 | Hidden-context ветка (`reply_context_emit_start=false`) мертва по дефолту | влияет только на эмиссию хвоста; количество контекстных зацепок не меняет | Closed — PR #76 (ручка удалена, эмиссия всегда включена) |

Не покрыто симуляцией (оговорки): heated-настроение (в пуле сообщений нет
капса/восклицаний), casefold-матчи (пул в нижнем регистре), `/pivo`-механики.

## Session update — 2026-07-12

### Completed
- Синхронизация репозитория (fast-forward 150 коммитов, устаревшая правка
  PROJECT_AUDIT.md отброшена), CLAUDE.md переведён на конвенцию `docs/audits/`.
- Симуляционный аудит генерации (см. выше), инструмент — `sim_dialogues.py`
  (вне репозитория, scratchpad-скрипт; отчёты переданы пользователю).
- PR #70 — удаление order-1 цепи и `transitions1` (миграция 013).
- PR #71 — повышение шансов L1/L3 фич (0.25 / 0.03 / 0.05).
- PR #77 — слияние diversity→repetition, удаление coherence, схлопывание
  staleness-гейтов (stacked на #70; изначально открыт как PR #72, закрыт без
  мержа и перевыпущен как #77 при разборе стека).
- PR #73 — stem-матчинг контекстных состояний вместо prefix, включён по
  дефолту (stacked поверх той же ветки).

### Changed files
- По веткам PR #70, #71, #73, #77: `app/core/markov.py`, `candidate_scorer.py`,
  `response_generator.py`, `context_state_matcher.py`, `gen_trace_log.py`,
  `app/config/{registry,settings,runtime_state}.py`,
  `app/infrastructure/database.py`, `app/repositories/{markov_repo,
  chat_hot_ngrams_repo}.py`, `app/migrations/013_drop_transitions1.sql`,
  `app/presentation/bot_messages.py`, `.env.example`, `tools/eval_*.py`,
  фикстура `synthetic_generation_stem_context.txt`, соответствующие тесты,
  `docs/GENERATION_PIPELINE.md`, `docs/ARCHITECTURE.md`.

### Tests/checks run
- `python -m unittest discover tests` на каждой ветке: 643 / 647 / 642 / 640 OK.
- `ruff check app/ tests/ tools/` и `mypy app/` — clean на каждой ветке.
- `tests.test_eval_generation` — бейзлайны байт-стабильны на #77; на #70 и
  #73 контентные метрики без изменений (на #73 перегенерированы из-за
  переименования ключей отчёта).

### Not run / limitations
- eval_prod на реальной прод-БД не гонялся (копии БД в окружении нет) —
  рекомендуется прогнать после мержа #73 и сравнить
  `context_anchored_win_rate` / `verbatim_run`.
- Живой запуск бота не проводился.

### Remaining work
- SIM-8: фикс вокатива без разделителя.
- SIM-9: решение по hidden-context ветке (удалить флаг или оставить).
- После мержа стека: прогнать eval_prod на прод-копии.
