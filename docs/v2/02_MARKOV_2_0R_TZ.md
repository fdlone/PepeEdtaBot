# PepeEdtaBot Markov 2.0R — Technical Specification

**Status:** Proposed, v1.1 (после внешнего ревью)
**Version:** 2.0R-1.1
**Architecture target:** существующие `app/core`, `services`, `repositories`, SQLite
**Compatibility:** Markov 1.x остаётся доступным как feature-flagged baseline

---

# 1. Scope

Реализация ОБЯЗАНА сохранять архитектурное направление проекта:
тонкие handlers; логика в services/core; SQL в repositories; `app/core`
без зависимости от инфраструктуры БД; изменения БД через миграции;
генерация тестируема без Telegram; существующие privacy/retention-правила
авторитетны.

**Вне scope** (ADR-013, ADR-016): hierarchical scopes, conditional state,
beam search, **изменение канонической токенизации** (включая склейку
коллокаций в токены).

Ключевое системное ограничение, влияющее на многие решения ниже:
сырые сообщения не хранятся ⇒ **rebuild модели невозможен** ⇒ все
изменения представления данных forward-compatible only.

---

# 2. Терминология

- **Состояние**: `S = (w[n-order], ..., w[n-1])`.
- **Переход**: `S -> token` (прямой) или `token -> S_prev` (реверсный).
- **Слои**: long-term (без decay, сублинейная нормализация) и short-term
  (затухающие счётчики).
- **Seed-токен**: токен входящего сообщения, выбранный композитным
  seed_score как лексический якорь. Механизм — statistical lexical
  anchoring, НЕ семантическое определение темы; документация не должна
  описывать его как «бот понимает тему».
- **Пресет**: именованный набор параметров генерации (Phase 8).
- **Документ (для IDF)**: одно нормализованное сообщение чата.

---

# 3. Архитектура

```text
                    ┌──────────────────────┐
                    │   ReplyPipeline      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ GenerationContext    │
                    └──────────┬───────────┘
                               ▼
                 ┌────────────────────────────┐
                 │ Markov2Generator           │
                 │ state / order (gated 4)    │
                 │ layer blend (short/long)   │
                 │ entropy / confidence       │
                 │ sampling                   │
                 └───────┬───────────┬────────┘
                         │           │
                    forward      reverse (order-2,
                     chain        experimental)
                         └─────┬─────┘
                               ▼
              обычные кандидаты + seeded-кандидаты
                               ▼
              существующий candidate scorer
              (+ IDF context, + collocation bonus/penalty)
                               ▼
                        final response
```

Scorer остаётся финальным гейтом (ADR-008); seeded-кандидаты конкурируют
без приоритета.

---

# 4. Core-абстракции

## 4.1 `MarkovState`

```python
@dataclass(frozen=True, slots=True)
class MarkovState:
    tokens: tuple[str, ...]
```

## 4.2 `TransitionDistribution`

```python
@dataclass(frozen=True, slots=True)
class Transition:
    token: str
    weight: float          # эффективный вес после слоя/нормализации
    probability: float

@dataclass(frozen=True, slots=True)
class TransitionDistribution:
    order: int
    direction: Literal["forward", "reverse"]
    layer: Literal["short", "long", "blended"]
    transitions: tuple[Transition, ...]
    entropy_bits: float
    normalized_entropy: float
    branching_factor: int
    confidence: float
```

## 4.3 `GenerationDiagnostics`

```python
@dataclass(slots=True)
class GenerationDiagnostics:
    selected_order: int
    entropy_bits: float
    normalized_entropy: float
    branching_factor: int
    confidence: float
    short_term_alpha: float
    seeded: bool
    seed_score: float | None       # сам токен — только при opt-in trace
    collocation_bonus_applied: bool
    cycle_detected: bool
    jump_probability: float
    preset_id: str | None
    cache_hit: bool
```

Без сырого текста, если явно не включено для локальной разработки.

---

# 5. Variable-order selection *(gated — Phase 7)*

Без изменений против v1.0: порядки 4/3/2, выбор по support + confidence,
существование order-4 строки ≠ выбор order-4. Нормативен только после
гейта shadow-статистики.

```text
MARKOV_MAX_ORDER=4
MARKOV_MIN_ORDER=2
MARKOV_ORDER4_MIN_COUNT=3
MARKOV_ORDER3_MIN_COUNT=2
MARKOV_CONFIDENCE_THRESHOLD=0.35
```

---

# 6. Энтропия

`H(S) = -Σ p_i·log2(p_i)`; `H_norm = H / log2(B)` (B ≤ 1 ⇒ 0);
`C = 1 - H_norm`. Температура:
`T = T_base · (1 + GAIN·(H_norm − H_pivot))`, clamp `[T_min, T_max]`;
`RANDOMNESS_STRENGTH` остаётся масштабом `T_base`; при `GAIN=0`
поведение идентично 1.x. Энтропия не переопределяет жёсткие гейты.

---

# 7. Short-слой: экспоненциально затухающие счётчики

**Заменяет схему `count × D(now − last_seen)` из v1.0**, которая
статистически некорректна: переход со 100 старыми употреблениями и одним
вчерашним получал бы short-вес ~100 вместо ~1.

## 7.1 Представление

На переход short-слой хранит пару `(s_value REAL, s_updated_at)`.

- Наблюдение перехода в момент t:
  `s_value = s_value · 2^(−(t − s_updated_at)/hl) + 1; s_updated_at = t`
- Чтение в момент t:
  `s_eff = s_value · 2^(−(t − s_updated_at)/hl)`

Свойство: `s_eff ≡ Σ_i 2^(−age_i/hl)` по всем наблюдениям — точная
сумма индивидуально затухших вкладов при O(1) хранении, без временных
buckets и без сырых событий.

## 7.2 Half-life

`MARKOV_SHORT_HALF_LIFE_DAYS=3`, диапазон 1–14. Счётчик математически
привязан к half-life, с которым накапливался: **смена параметра
обнуляет short-слой** (`s_value = 0`), что допустимо — слой
восстанавливается за считанные дни. Смена выполняется явной командой /
миграцией с предупреждением, не тихо.

## 7.3 Long-слой

Хранит `count` (целочисленный, без затухания), `first_seen`, `last_seen`
(для GC-отчётности и телеметрии, не для веса).

---

# 8. Blend и нормализация слоёв

## 8.1 Нормализация long

Сырые исторические count'ы перед нормализацией сжимаются сублинейно:

`w_long(token) = log(1 + count)`  — либо `count^β`, `β ∈ [0.5, 0.75]`;
выбор формы — параметр `MARKOV_LONG_COMPRESSION` (`log` | `pow`).

Мотивация: без сжатия переход с count=10000 против count=20 даёт
0.998/0.002, и никакой разумный α не позволяет свежему языку влиять
на распределение — short вырождается в косметическую поправку.
Сжатие сохраняет порядок предпочтений, но сужает динамический диапазон
до рабочего.

## 8.2 Нормализация short

`w_short(token) = s_eff(token)` (сжатие не применяется: динамический
диапазон short ограничен самим затуханием).

## 8.3 Blend

`P = α·P_short + (1−α)·P_long`, нормализация по объединению токенов;
пустой слой ⇒ α эффективно 0 или 1. α по mood (дефолты):

```text
sleepy 0.20 | calm 0.30 | lively 0.50 | heated 0.70
```

## 8.4 Калибровка (обязательный эксперимент M2R-215)

Вопрос: какие (α-профиль, β/форма сжатия) позволяют свежему языку
измеримо менять распределение, не вытесняя долгоживущие паттерны.
Метод: сетка конфигураций на eval-раннере; целевые метрики —
скорость отражения свежего среза + мем-регресс-кейс одновременно.
Дефолты выше — стартовая точка сетки, не решение.

---

# 9. Statistical lexical anchoring (Phase 5, эксперимент)

## 9.1 Статус

Вся секция — **основной эксперимент Phase 5** (ADR-012 Provisional).
Промоушен в ядро — только при выполнении критериев §9.6.

## 9.2 Реверсный индекс (только order-2)

Обратные переходы хранятся для order-2: достройка головы предложения
от seed не требует order-3 точности — голова коротка и проходит общий
scorer. Ограничение режет storage-риск (~2x → существенно меньше)
и упрощает бэкфилл-миграцию. Обучение пишет прямые и реверсные агрегаты
в одной транзакции.

## 9.3 IDF

Документ = нормализованное сообщение. Поскольку сырые сообщения не
хранятся, df ведётся инкрементально: агрегат `token -> messages_seen`
(+1 на уникальное вхождение токена в сообщение) и глобальный счётчик
сообщений `N_docs`. `idf(t) = log(N_docs / (1 + df(t)))`.
Агрегат подчиняется `/clear confirm`.

## 9.4 Выбор seed: композитный score

**Не max-IDF** (уязвим к мусорным уникальным токенам вида `foobar123`):

```text
seed_score = normalized_idf(t) × support_factor(t) × branching_quality(t)
```

- `normalized_idf` — idf, нормированный по токенам сообщения;
- `support_factor` — насыщающая функция от суммарного count токена
  в модели (редкое в чате, но реально употребляемое — хорошо;
  почти не встречавшееся — плохо);
- `branching_quality` — **полоса, не порог**: forward- и
  reverse-branching оцениваются трапецией

```text
B < MARKOV_SEED_BRANCH_MIN            → 0 (unusable)
MIN ≤ B ≤ MARKOV_SEED_BRANCH_IDEAL   → 1
B > IDEAL                            → плавный спад к MARKOV_SEED_BRANCH_MAX
```

  (слишком малый branching — генерация упрётся; слишком большой —
  якорь ни о чём). Границы — параметры, калибруются eval'ом.
- Токены короче `MARKOV_SEED_MIN_TOKEN_LEN` и стоп-слова исключаются
  до скоринга.

Нет кандидата с `seed_score ≥ MARKOV_SEED_MIN_SCORE` ⇒ seeded-ветка
пропускается (прозрачный fallback).

## 9.5 Генерация и интеграция

От seed: хвост прямой цепью, голова реверсной (order-2); blend/entropy
правила §6–8 действуют в обоих направлениях. Бюджет длины — существующие
лимиты; распределение голова/хвост — конфиг.
`MARKOV_SEEDED_CANDIDATE_RATIO=0.3` (0–0.7) — доля seeded-кандидатов
в пуле best-of-N; проходят общий scorer без приоритета.

## 9.6 Критерии промоушена (числа фиксируются в eval-протоколе)

Телеметрия обязана разделять знаменатели:

- `seeded_present_rate` — доля генераций, где seeded-кандидат вообще был;
- `seeded_win_rate_given_present` — доля побед при наличии.

Промоушен ADR-012 в Accepted требует одновременно:
- `seeded_win_rate_given_present` ≥ порога при `seeded_present_rate`
  ≥ порога;
- прирост `context_affinity_without_copy` против ablation-конфигурации
  без seeded;
- latency p95 в бюджете §17; фактический storage-прирост ≤ согласованного.

Невыполнение ⇒ фича выключается флагом, реверсные таблицы замораживаются
(не читаются), решение фиксируется с цифрами.

## 9.7 IDF-компонент скоринга кандидатов

Слагаемое scorer'а: Σ idf по пересечению токенов кандидата с последними
`MARKOV_CONTEXT_WINDOW_MESSAGES` нормализованными сообщениями, вес
`MARKOV_CONTEXT_IDF_WEIGHT` (0 ⇒ поведение 1.x). Метрика качества —
`context_affinity_without_copy` (определение — док 05), парная к
`exact_context_copy_rate`: рост пересечения не должен достигаться
перефразом/копией входа.

---

# 10. Коллокации и PMI-мемы

## 10.1 Анализатор (без изменений против v1.0)

Офлайн-джоб: PMI, lift, LLR; обязательные пороги
`MARKOV_MEME_MIN_JOINT_COUNT`, `MARKOV_MEME_MIN_SUPPORT`,
`MARKOV_MEME_RECENCY_FACTOR`;
`meme_score = normalized_pmi × support_factor × recency_factor`.
Результат подаётся в существующий механизм hot n-grams.

## 10.2 Коллокации — только уровень скоринга (ADR-016)

Склейка коллокаций в единые токены **исключена из scope**: одноsided
изменение токенизации при невозможности rebuild (§1) создаёт вечно
сосуществующие несовместимые представления (`KEK_LOL` vs `кек`+`лол`)
при любом последующем retirement.

Вместо этого активные коллокации участвуют в скоринге кандидатов:

- бонус `MARKOV_COLLOCATION_BONUS` кандидату, содержащему активную
  коллокацию как непрерывную последовательность;
- штраф `MARKOV_COLLOCATION_BREAK_PENALTY` кандидату, содержащему левый
  токен активной коллокации с иным продолжением **при условии, что
  правый токен был статистически доступен** (существует переход);
- список активных коллокаций: per-chat, ≤ `MARKOV_COLLOCATION_MAX_ENTRIES`,
  статусы candidate|active|retired, виден в `/stats`, подчиняется
  `/clear confirm`. Retirement безопасен: скоринг-правило просто
  перестаёт применяться, данные модели не затронуты.

Склейка в токенизации может вернуться только отдельным proposal с
обязательным `tokenization_version` и решённой проблемой сосуществования
версий (future experiments).

---

# 11. Anti-cycle *(gated — Phase 6)*

Механика без изменений (deque transition-id maxlen 8, детект 2/3-циклов,
`cycle_continuation_multiplier=0.15`), но **гейт двумерный**:

1. `cycle_detection_rate` — частота циклов в телеметрии;
2. `cycle_harm_rate` — фактический вред: доля кандидатов, отклонённых
   финальным scorer'ом по причинам, коррелирующим с циклами, плюс ручная
   оценка выборки циклических ответов по протоколу дока 05.

Обоснование: цикл ≠ вред. «да да да» — легитимный ответ Pepe;
детектироваться будет, портить генерацию — нет. Фаза открывается только
при превышении порогов обеими метриками.

---

# 12. Динамический jump *(gated — Phase 6)*

Без изменений: `jump_probability = base · entropy_factor ·
repetition_factor · context_factor · mood_factor`, clamp
`[0, MARKOV_MAX_JUMP_PROBABILITY]`; детект вредного цикла ⇒ форсированный
jump.

---

# 13. Изменения БД

Имена согласуются с фактической схемой до кодирования миграций.

### Таблица прямых переходов

```text
+ count          (long-слой; если ещё не агрегировано)
+ first_seen
+ last_seen
+ s_value REAL   (short-слой)
+ s_updated_at
```

### Реверсные переходы (Phase 5, только order-2)

Реализовано **индексом** `idx_transitions_reverse (chat_id, w2, w3)` над
существующей таблицей прямых переходов, а не отдельной таблицей
`markov_reverse (chat_id, token, state_hash, …)`, эскизированной в v1.x:
реверсный переход — это та же строка прямой таблицы, прочитанная по последним
двум колонкам. Индекс даёт согласованность по построению (второй копии
счётчиков нет — нечему разъезжаться), кратно меньший storage (замер миграции
020: +13.9% файла против кратного у таблицы-дубликата), бэкфилл одним
`CREATE INDEX` и откат одним `DROP INDEX`. Семантика §9.2 не изменилась.
Решение: change `markov2r-phase5-reverse-index`, design D1.

### `markov_token_df` (Phase 5)

```text
chat_id, token, messages_seen
```
(+ `N_docs` в существующей таблице chat-метаданных)

### `markov_collocations`

```text
chat_id, left_token, right_token, joint_count, pmi,
status (candidate|active|retired), updated_at
```

### `generation_presets_stats` (Phase 8)

```text
chat_id, preset_id, positive_count, negative_count, trials, updated_at
```

Колонок для scope, conditional state и `tokenization_version` НЕТ.

## 13.1 Индексы

```text
forward:  (chat_id, order, state_hash)
          (chat_id, order, state_hash, token)
reverse:  (chat_id, token)
df:       (chat_id, token)
          (last_seen) — только при включённом GC
```

---

# 14. Кэш *(ранняя фаза; условие latency)*

Ключ `(chat_id, direction, order, state_hash, layer)` →
`TransitionDistribution`; LRU или TTL+LRU, безлимитный dict запрещён;
инвалидация по затронутым state при обучении, иначе TTL; hit-rate
в `/stats`. Внимание: `s_eff` зависит от времени чтения — кэшировать
следует сырые `(count, s_value, s_updated_at)` наборы или blended
распределение с коротким TTL, не «замороженный» `s_eff` навсегда.

---

# 15. GC (данные) — двухуровневый, полностью opt-in

Decay не удаляет данные никогда. Удаление — только офлайн-джоб:

## 15.1 Дефолтный режим

```text
MARKOV_GC_ENABLED=false
MARKOV_GC_MIN_AGE_DAYS=365    # валидация ≥ 365
MARKOV_GC_MAX_COUNT=1
```

Удаляет только `count == 1 AND age ≥ порога`. Обязателен dry-run отчёт.

## 15.2 Расширенный режим (отдельный, ещё более осознанный opt-in)

```text
MARKOV_GC_EXTENDED_ENABLED=false
MARKOV_GC_EXT_MAX_COUNT=3
MARKOV_GC_EXT_MIN_AGE_DAYS=1095   # валидация ≥ 730
```

Eligible: `count ≤ EXT_MAX_COUNT` И `age ≥ EXT_MIN_AGE_DAYS` И нет
рекуррентности (все наблюдения в узком историческом окне; проверяется
по `first_seen`/`last_seen`) И низкая ассоциация (переход не входит в
активные коллокации/мемы). Критерий разделяет frequency, recency и
association вместо одного магического порога count. Обязателен dry-run;
переходы, входящие в активные мемы, не eligible ни при каком count.

Формулировка гарантии проекта: «GC не удаляет переходы с count > 1
**по умолчанию**»; расширенный режим — явное информированное решение
оператора.

---

# 16. Мета-тюнинг по реакциям *(Phase 8, экспериментально)*

- Сигнал: Telegram-реакции, агрегированные per-chat per-preset;
  событий, привязанных к пользователям, нет.
- **Reward отложенный**: реакции на сообщение аккумулируются в окне
  `MARKOV_PRESET_REWARD_WINDOW_HOURS` (дефолт 24) и лишь затем
  зачисляются пресету одним агрегатом. Мгновенная реакция ≠ reward:
  она может относиться к контексту, персонажу или привычке ставить 🔥.
- Эксплуатация (exploitation) начинается только после
  `MARKOV_PRESET_MIN_TRIALS` на пресет; до того — равномерная ротация.
- Алгоритм: epsilon-greedy или Thompson; влияет только на выбор пресета
  (ADR-014).
- Закрытие: статистическая неразличимость пресетов за настроенный срок ⇒
  выключение с фиксацией вывода.

---

# 17. Производительность

- lookup распределения (с кэшем): p95 < 10 ms;
- полная генерация без Telegram I/O: p95 < 150 ms;
- overhead над 1.x: < 50% CPU;
- storage-прирост Phase 5 измеряется и входит в критерии промоушена §9.6;
- никаких full-table scan'ов на сообщение.

Цели измеряются eval-раннером.

---

# 18. Конфигурация

```text
MARKOV_V2_ENABLED=false

MARKOV_ENTROPY_ENABLED / MARKOV_ENTROPY_TEMP_GAIN
MARKOV_BRANCHING_CANDIDATES_ENABLED

MARKOV_SHORT_HALF_LIFE_DAYS
MARKOV_LONG_COMPRESSION           # log | pow
MARKOV_LONG_COMPRESSION_BETA      # для pow
MARKOV_ALPHA_SLEEPY/_CALM/_LIVELY/_HEATED

MARKOV_GC_ENABLED / MARKOV_GC_MIN_AGE_DAYS / MARKOV_GC_MAX_COUNT
MARKOV_GC_EXTENDED_ENABLED / MARKOV_GC_EXT_MAX_COUNT / MARKOV_GC_EXT_MIN_AGE_DAYS

MARKOV_SEEDED_ENABLED / MARKOV_SEEDED_CANDIDATE_RATIO
MARKOV_SEED_MIN_TOKEN_LEN / MARKOV_SEED_MIN_SCORE
MARKOV_SEED_BRANCH_MIN / _IDEAL / _MAX
MARKOV_CONTEXT_IDF_WEIGHT / MARKOV_CONTEXT_WINDOW_MESSAGES

MARKOV_MEME_MIN_JOINT_COUNT / MARKOV_MEME_MIN_SUPPORT
MARKOV_COLLOCATION_MAX_ENTRIES
MARKOV_COLLOCATION_BONUS / MARKOV_COLLOCATION_BREAK_PENALTY

MARKOV_CYCLE_ENABLED / MARKOV_MAX_JUMP_PROBABILITY      # gated
MARKOV_MAX_ORDER / MARKOV_ORDER4_MIN_COUNT / MARKOV_CONFIDENCE_THRESHOLD  # gated

MARKOV_PRESETS_ENABLED / MARKOV_PRESET_REWARD_WINDOW_HOURS / MARKOV_PRESET_MIN_TRIALS
MARKOV_CACHE_MAX_ENTRIES
MARKOV_TRACE_ENABLED
```

Все параметры — в `.env.example` с границами; изменяемые на лету —
через `app/config/registry.py` и `/set`.

---

# 19. Тестирование

## Unit

Энтропия/нормализация; затухающий счётчик (инварианты: наблюдение сейчас
даёт +1; вклад наблюдения возрастом hl равен 0.5; порядок наблюдений не
влияет на итог); сублинейное сжатие; blend; seed_score (idf, support,
трапеция branching, fallback); реверсная генерация order-2; коллокационный
бонус/штраф (включая условие «правый токен был доступен»); GC-критерии
обоих режимов; кэш с time-dependent `s_eff`; (gated) порядок, циклы, jump.

## Property / инварианты

p_i ≥ 0; Σp ≈ 1; нет NaN/inf; blend валидных распределений валиден;
`s_eff` монотонно не возрастает между наблюдениями; реверсная генерация
завершается в бюджете токенов.

## Интеграционные

Атомарность forward+reverse+df при обучении; бэкфилл-миграция реверса;
`/clear confirm` удаляет ВСЕ структуры 2.0R (реверс, df, коллокации,
пресеты); смена half-life обнуляет short-слой с предупреждением;
retention не изменён; миграции проходят на чистой и существующей БД.

## Регрессионные

Существующий набор + eval-раннер (док 05) как гейт каждой фазы, включая
ablation-конфигурации. Мем-регресс: n-граммы годовалого среза
воспроизводимы long-слоем.

---

# 20. Наблюдаемость

```text
v2=true order=3 entropy=0.42 branching=3 confidence=0.58
alpha=0.50 seeded=true seed_score=0.71 colloc_bonus=false
cycle=false jump=0.07 preset=B cache_hit=true
```

`/stats`: распределение порядков, средняя энтропия/branching,
`seeded_present_rate` и `seeded_win_rate_given_present`, hit-rate кэша,
активные коллокации, оценки пресетов, (gated) cycle rate + harm rate.
Маскирование `chat_id` — на все новые сообщения логов; сырой текст —
только opt-in.

---

# 21. Acceptance criteria

1. Все существующие тесты проходят; миграции — на чистой и
   репрезентативной БД.
2. Markov 1.x выбирается флагом.
3. Распределения валидны и нормализованы.
4. Сырые сообщения и пер-пользовательские модели не введены.
5. `/clear confirm` полон.
6. Латентность в бюджете §17.
7. Exact-copy rate не вырос; `context_affinity_without_copy` не
   деградировал.
8. Repetition/cycle-harm не регрессировали.
9. Мем-регресс-кейс проходит; GC в дефолтной конфигурации не трогает
   `count > 1`.
10. Результаты eval воспроизводимы (фиксированный seed) и включают
    ablation-разбивку по фичам.

---

# 22. Rollback

Каждая подсистема отключаема флагом (§18); миграции forward-compatible;
отключение не требует деструктивного отката. Замороженные структуры
(реверс, df, коллокации, пресеты) при выключенных фичах не читаются.
