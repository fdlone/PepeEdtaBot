# Дифференциальное ревью PR #55–58 (2026-07-06)

**Диапазон:** `f3c536d..09f204e` (merge #54 → merge #58), ветка `main`, 19 не-merge коммитов.
**Объём:** 84 файла, ~4040 вставок; из них код приложения — 27 файлов, ~1080 строк.
**Методика:** differential-review (Trail of Bits), стратегия FOCUSED (кодовая база MEDIUM, 59 файлов в `app/`).

## Состав изменений

| PR | Ветка | Содержание |
|----|-------|-----------|
| #55 | `feat/dialogue-gen-stage3-m3-m4` | M3 emoji-канал (таблица `chat_emoji_stats`, миграция 011), M4 topic-drift jump со splice-коннективами |
| #56 | `chore/audit-followup-fixes` | Закрытие находок аудита N1–N6: `/clear` чистит /pivo-данные, анти-повтор /pivo фиксируется после доставки, throttle-notify rate-limit, mention-cooldown, hourly-cap не считает mention-ответы |
| #57 | `feat/dialogue-gen-stage4-l1` | L1 hot-ngrams (таблица `chat_hot_ngrams`, миграция 012, seed unprompted-ответов) |
| #58 | `feat/dialogue-gen-stage4-l3` | L3 rare events / false starts, `reply_humanized_sequence` для мульти-сообщений |

## Классификация риска и покрытие ревью

- **HIGH (прочитано полностью):** `filters/admin_or_owner.py`, `middlewares/throttling.py`, `handlers/admin.py`, `infrastructure/database.py`, миграции 011/012, все репозитории.
- **MEDIUM (прочитано полностью):** `handlers/learning.py`, `handlers/pivo.py`, `core/*` (emoji, hot_ngrams, markov, reply_flavor, response_generator), `services/*`, `config/*`.
- **LOW (просмотрено по diff-stat):** тесты, docs, `bot_messages.py`, `eval_generation.py`.

Покрытие: 100 % изменённых файлов приложения прочитаны построчно.

## Findings

### Подтверждённых уязвимостей: 0

Проверенные гипотезы (все отклонены):

1. **SQL-инъекции** — все новые запросы (`chat_emoji_stats_repo`, `chat_hot_ngrams_repo`, delete-методы /pivo-репозиториев) параметризованы; конкатенации пользовательского ввода нет.
2. **`message.from_user is None`** в mention-cooldown (`learning.py`) — защищено ранним guard'ом (`learning.py:173`).
3. **Деление на ноль в `get_hot`** — знаменатель `MAX(COALESCE(t.cnt, h.cnt), h.cnt) >= h.cnt >= min_count >= 1` (валидатор `_int_min(1)`), недостижимо.
4. **Рост памяти `last_mention_reply_ts` / `rare_events_today`** — оба словаря чистятся в `forget_chat`, который вызывается из `prune_inactive` (TTL + max_chats). Ограничено.
5. **Обход анти-повтора /pivo при отказе по квоте** — отказ по квоте (`return` на `pivo.py:80`) и исключение при отправке происходят до `record_pool_usage`; фиксация только после доставки. Корректно.
6. **Регрессия снятого ограничения** — удалённая строка `jump_probability = 0.0` («Disabled until a jump can splice tokens safely») сопровождается именно реализацией безопасного splice (коннектив + реальный learned start + обновление анти-циклических структур `visited_triplets`/`seen_pairs`/`seen_triplets`). Это закрытие TODO, не регрессия.
7. **Утечка контента чата в логи** — hot-ngram seed логирует только длину n-граммы, не текст (политика log-masking соблюдена); chat_id маскируется.
8. **Race check-then-act дневного капа rare events** — между `can_fire_rare_event` и `note_rare_event` нет `await`; в однопоточном asyncio безопасно.

### Замечания (LOW, не блокирующие)

- **L-1. Локальная TZ для дневного капа rare events.** `handlers/learning.py` использует `date.today()` (локальное время), тогда как остальной код (decay, /pivo-квоты) — UTC. Эффект: суточный бюджет rare events сбрасывается в локальную полночь контейнера. Косметика, но при переносе хоста поведение чуть сместится.
- **L-2. Decay только на старте.** `decay_chat_emoji_stats` / `decay_chat_hot_ngrams` выполняются один раз в `init()`. При аптайме в недели окно «горячести» перестаёт скользить (n-граммы постепенно теряют статус hot из-за роста all-time счётчика — деградация мягкая). Осознанное решение (планировщика в проекте нет), задокументировано в docstring.

> **Резолюция (2026-07-06, тот же PR):** все три замечания закрыты. L-1 — суточный кап rare events переведён на UTC-дату; L-2 — добавлен ленивый суточный ре-ран decay с learn-пути (`Database.decay_flavor_stats_if_due`); L-3 — `_TRAILING_EMOJI_RE` теперь требует хотя бы один эмодзи в хвосте.
- **L-3. `strip_trailing_emojis` срезает и голую пунктуацию** (`(?:emoji|[\s.!?…])+\Z` матчится и без единого эмодзи). Сейчас безвредно: единственный вызов — `normalize_reply_for_repeat`, где следом идёт `rstrip(" .!?…")`. Если функция будет переиспользована в другом контексте, поведение может удивить.

## Аудит-находки N1–N6: статус закрытия подтверждён

| Находка | Фикс в диффе | Проверено |
|---------|--------------|-----------|
| N1 `/clear` не чистил /pivo | `clear_pivo_chat_data` + `PivoService.clear_chat_data`, подтверждение в `/clear`-сообщении | ✅ |
| N2 анти-повтор /pivo крутился при отказе | `build_call_message` стал side-effect-free, `record_pool_usage` после доставки | ✅ |
| N3–N5 (mention flood / hourly cap) | `mention_cooldown_sec` (демоция в unprompted-путь), `note_reply_sent(unprompted=...)`, нейтральный burst-фактор для чата без ответов | ✅ |
| N6 throttle-notify flood | `notify_cooldown_sec=30`, `_last_notified` с prune | ✅ |

## Тестовое покрытие

Новые каналы покрыты целевыми тестами: `test_emoji.py` (+119), `test_hot_ngrams.py` (+52), `test_chat_hot_ngrams_repo.py` (+184), `test_chat_emoji_stats_repo.py` (+66), `test_reply_flavor.py` (+58), `test_runtime_state.py` (+45), `test_handlers.py` (+490, включая mention-cooldown, rare events, hot-ngram seed). CI гонялся на каждом PR стека.

## Вердикт

**Изменения безопасны к эксплуатации.** Подтверждённых уязвимостей нет; три LOW-замечания — на усмотрение (фиксить не обязательно). Приватностный контур сохранён: новые таблицы — пер-чатовые агрегаты без авторства, ключуются как модельные таблицы, чистятся в `clear_chat`, контент в логи не попадает.

## Ограничения ревью

- Ревью статическое: тесты/линтеры в рамках ревью не перезапускались (CI стека зелёный на момент merge).
- Adversarial-моделирование (Phase 5) не выполнялось: HIGH-risk изменения (auth/throttling) оказались комментарием + аддитивным rate-limit'ом, триггеров эскалации нет.
- Unicode-диапазоны `_EMOJI_RE` проверены по чтению, без фаззинга на полном эмодзи-корпусе (ZWJ-последовательности сознательно вне охвата — задокументировано).
