# Полная карта конвейера генерации ответа

Дата: 2026-07-06 (обновлено в тот же день после правок «видимый контекст + анти-цитаты»).
Документ — рабочая карта для дальнейших изменений генерации: каждый этап, функция и
ручка конфигурации, влияющие на итоговый текст ответа.

> **Изменения 2026-07-06** (диагноз: 69% ответа — дословная цитата корпуса,
> контекст-попадание 0.13; после правок: 0.42 и 0.16, ответов ≥90% verbatim
> 33% → 5.5%):
> 1. Контекстный старт стал **видимым**: из матчнутого окна эмитятся последние
>    ≤2 токена без ведущих стоп-слов/пунктуации («кто гнойный пидор» →
>    «гнойный пидор …»), ручка `reply_context_emit_start=true`
>    (`context_emission_tokens`, markov.py).
> 2. Скоринг: новый компонент `verbatim_penalty` (= `verbatim_penalty_strength=1.0`
>    × доля контент-4-грамм кандидата, найденных в корпусном индексе
>    `LearningService.get_verbatim_ngram_index`). `coherence_penalty` и
>    `lexical_diversity` удалены 2026-07-12: первый стал шумом после удаления
>    order-1 цепи, второй схлопнут в `repetition_penalty` (1 − diversity ==
>    доля повторов токенов, слияние не меняет ранжирование).
> 3. Дыры verbatim-гейта закрыты: сравнение без хвостовой пунктуации
>    (`normalize_for_verbatim`), окно кэша 500 → 1000.
> 4. Хаос: `candidate_selection_temperature` 0.7 → 1.3,
>    `markov_jump_probability` 0.04 → 0.12, порог прыжка 9 → 5 токенов
>    (`JUMP_MIN_GENERATED_TOKENS`).
> 5. M4-прыжки больше **не** целятся в контекстные старты (только глобальные):
>    контекст держится стартом и биасами, контекстный прыжок вклеивал вопрос
>    в ответ по 2–4 раза (попугайство). `_select_contextual_start3` удалён.
>
> Ниже по тексту разделы 4–6 описывают состояние *до* этих правок в деталях;
> сверяйтесь с кодом при работе с изменёнными местами.

Связанные документы: [DIALOGUE_GENERATION_AUDIT.md](DIALOGUE_GENERATION_AUDIT.md)
(аудит-снапшот 2026-07-02, частично устарел),
[DIALOGUE_GENERATION_ACTION_PLAN.md](DIALOGUE_GENERATION_ACTION_PLAN.md).

---

## 0. Обзор: путь сообщения от входа до ответа

```
Telegram message (F.text)
  └─ on_text_message (app/handlers/learning.py:162)
       ├─ 1. Входные фильтры (группа, не бот, не команда)
       ├─ 2. Детект обращения к боту + mention-cooldown
       ├─ 3. Обновление настроения чата (M1) и ритма (M2)
       ├─ 4. Учёт эмодзи (M3) — до гейтов обучаемости
       ├─ 5. Гейты обучаемости и объёма модели
       ├─ 6. Решение «отвечать или нет» (reply_policy, директор M2)
       ├─ 7. Подготовка контекста ответа + hot-ngram seed (L1)
       ├─ 8. ResponseGenerator.generate → до 10 попыток → до 5 кандидатов
       │      └─ MarkovGenerator.generate_text (старт → цепь → финализация)
       ├─ 9. Скоринг кандидатов → softmax-выбор
       ├─ 10. Пост-обработка: капитализация, flavor, эмодзи (M3)
       ├─ 11. Редкие события (L3): verdict/CAPS/double/false-start
       ├─ 12. Отправка с имитацией набора (reply_humanized_sequence)
       └─ 13. finally: обучение (record_message, hot-ngrams)
```

---

## 1. Точка входа: `on_text_message` (`app/handlers/learning.py:162`)

Единственный обработчик текста (`@router.message(F.text)`). DI через aiogram:
`learning_service`, `generator` (MarkovGenerator), `runtime_state`, `bot_username`,
`bot_id`, `bot_text_aliases`.

### 1.1 Входные фильтры (ранние выходы)
- `is_group_message` — только GROUP/SUPERGROUP (`_helpers.py:18`).
- `message.from_user is None` или `.is_bot` — выход.
- Текст начинается с `/` — команда, выход.

### 1.2 Детект обращения: `bot_is_mentioned` (`reply_policy.py:35`)
Возвращает True, если:
- `@username` бота есть в тексте (подстрока, case-insensitive), или
- любой токен текста совпадает с алиасом (`pepe`/`пепе` по умолчанию,
  `DEFAULT_BOT_TEXT_ALIASES`, переопределяется через env), или
- entity типа mention указывает на `@username`, или
- сообщение — reply на сообщение бота.

**Mention-cooldown** (анти-флуд, `mention_cooldown_sec=5`): обращение в кулдауне
понижается до обычного пути (`address_reply=False`), но «сырой» `mentioned` всё
равно питает mood/rhythm. Ключ — `(chat_id, user_id)`.

### 1.3 Настроение и ритм чата (M1/M2): `update_mood_state` (`core/mood.py:185`)
Считается, если включён mood **или** директор (общие EWMA). Сигналы на каждое
сообщение:
- `rate_ewma` — мгновенная скорость `60/dt` msg/min, кламп `mood_max_rate_per_min=120`;
- `intensity_ewma` — `message_intensity`: 0.5·(плотность `!`/`?`, кап на 3) +
  0.5·(доля заглавных букв);
- `mention_ewma` — сглаженная доля обращений к боту.

Сглаживание `mood_ewma_alpha=0.3`. Классификация `classify_mood`:
интенсивность ≥ `mood_heated_intensity=0.4` → **heated**; иначе по темпу:
≤ `mood_sleepy_rate_per_min=2` → **sleepy**, ≥ `mood_lively_rate_per_min=12` →
**lively**, иначе **calm**. Первое сообщение чата → calm с baseline-темпом
(среднее sleepy/lively порогов).

`modifiers_for_mood(mood, strength)` (`mood.py:93`) даёт `MoodModifiers`
(таблица при strength=1.0, масштабируется `mood_modulation_strength=1.0`,
мультипликаторы клампятся снизу нулём):

| Mood | reply_prob × | randomness Δ | length (s,m,l) × | flavor × |
|---|---|---|---|---|
| calm | 1.0 | 0 | 1,1,1 | 1.0 |
| lively | 1.5 | +0.2 | 0.8,1.0,1.2 | 1.1 |
| heated | 1.3 | +0.5 | 1.1,1.0,0.9 | 1.4 |
| sleepy | 0.5 | −0.3 | 1.6,1.0,0.5 | 0.8 |

### 1.4 Эмодзи-канал (M3), учёт: `count_emojis` (`core/emoji.py:97`)
До гейтов обучаемости (эмодзи-only реакции тоже считаются). Regex по базовым
пиктографическим блокам; флаги склеиваются из пар региональных индикаторов,
одинокая половина отбрасывается. Пишется в `chat_emoji_stats` через
`learning_service.record_emojis`, только если `emoji_append_chance>0`.

### 1.5 Подготовка к обучению
- `strip_leading_bot_vocative` (`learning.py:74`) — срезает ведущее «Пепе, …»
  (алиас + разделитель `,:;—–-`), чтобы корпус не заучивал обращения.
- `sanitize_text` (`core/text.py:64`): удаление ссылок → редакция PII
  (`privacy_filter.redact_sensitive_data`) → удаление @mentions → сжатие
  повторов символов (3+ → 2) → нормализация пробелов.
- `tokenize` (`markov.py:97`): `\w+|[.,!?;:]`, lowercase только при
  `normalize_lower=true` (по умолчанию **false** — регистр сохраняется).
- Обучаемость: длина ≤ 500 символов **и** ≥ 2 токена.

### 1.6 Гейты объёма модели
`has_enough_model_data`: `token_volume ≥ min_tokens_for_model=200`
(объём — из `Database.get_chat_token_volume`).
- Обращение + мало данных → fallback из `NOT_ENOUGH_DATA_PHRASES` (см. §7).
- Нет обращения + мало данных → молча выход.

---

## 2. Решение «отвечать или нет» (`reply_policy.py`)

`should_reply_to_message` (`reply_policy.py:81`): обращение (`address_reply`) —
всегда True; иначе `cooldown_ok AND hourly_cap_ok AND random() < reply_prob`.

- **Cooldown**: `now − last_reply_ts ≥ min_cooldown_sec=45` (monotonic).
  Обращения обходят cooldown и капы (кроме mention-cooldown из §1.2).

### 2.1 Директор ответов M2 (`reply_director_enabled=true`)
- `conversation_momentum` (`reply_policy.py:94`): 0.55·норм. темп (по
  `lively_rate_per_min`) + 0.30·`mention_ewma` + 0.15·(это reply?), кламп [0,1].
- `burst_factor` (`reply_policy.py:124`): ≤ `reply_burst_boost_sec=180` c после
  ответа → ×`reply_burst_boost_mult=2.0`; далее до +`reply_burst_suppress_sec=600` c
  → ×`reply_burst_suppress_mult=0.5`; иначе 1.0. Чат без прежних ответов —
  нейтрален (сентинел −1).
- `effective_reply_probability` (`reply_policy.py:149`):
  `clamp01((min + momentum·(max−min)) · mood_mult · burst_mult)`,
  band = [`reply_probability_min=0.02`, `reply_probability_max=0.30`].
- `within_hourly_cap` (`reply_policy.py:166`): < `reply_max_per_hour=20`
  непрошеных ответов за скользящий час. Ответы на обращения не считаются.

При выключенном директоре — legacy: `reply_probability=0.08 × mood_mult`.

---

## 3. Подготовка входов генерации

### 3.1 Контекст: `extract_context_tokens` (`learning.py:133`)
При `use_reply_context=true`. Источники: текст reply-to-сообщения + (опц.)
текущее сообщение (`reply_context_include_current_message=true`). По умолчанию
только для реплаев (`reply_context_only_for_replies=true`), **но** прямое
обращение без reply переопределяет: контекст = само сообщение. Санитизация,
токенизация, берутся последние `reply_context_max_tokens=12` токенов.
Важно (Phase 4.1d): контекст влияет только через скрытое контекстное состояние
и биасы — литеральный seed из контекста больше не строится.

### 3.2 Hot-ngram seed (L1): `learning.py:403`
Только для **непрошеных** ответов, с шансом `hot_ngram_seed_chance=0.05`.
`learning_service.get_hot_ngrams` → n-граммы с ≥ `hot_ngram_min_count=3`
попаданиями в окне и долей окна ≥ `hot_ngram_recency_share=0.5`; случайная
становится `seed` генерации (реплика «в тему локального мема»).

---

## 4. Оркестратор: `ResponseGenerator.generate_with_result` (`core/response_generator.py:181`)

Константы: `GENERATION_ATTEMPT_BUDGET=10`, `GENERATION_ATTEMPTS_WITH_CONTEXT=5`
(после 5-й попытки контекст отбрасывается), `CANDIDATE_TARGET=5`,
`SELECTION_SCORE_MARGIN=0.3`, `SHORT_MODE_MAX_TOKENS=8`.

Порядок:
1. **Режим длины**: `sample_length_mode` по весам `length_mode_weights=0.25,0.55,0.2`
   × mood-мультипликаторы. Режим short дополнительно ограничивает саму
   генерацию: `max_tokens = min(45, 8)`.
2. **Randomness**: `randomness_strength=2.0 + mood.randomness_delta`, floor 0.
3. **До 10 попыток**, каждая — один вызов `MarkovGenerator.generate_text`
   (attempt_budget=1) с эскалацией случайности `escalated_randomness_strength`
   (линейно от базы к 3.0 по номеру попытки; `markov.py:122`).
4. **Отбраковка кандидата** (до скоринга):
   - дословная копия текущего сообщения (normalized);
   - короткий ответ, уже бывший в последних 5 коротких (`recent_short_replies`);
   - точный повтор одного из последних 20 отправленных (`recent_replies`,
     нормализация `normalize_reply_for_repeat`: lower + срез хвостовых эмодзи
     и пунктуации);
   - не-короткий кандидат, начинающийся как обучающий сэмпл —
     `learning_service.is_verbatim_copy` (кэш последних 500 normalized-текстов).
5. **Скоринг** уникальных кандидатов (`score_candidate`, §5) + штраф
   `recent_penalty = recent_reply_penalty_strength=0.5 × доля триграмм,
   совпавших с последними 20 ответами` (`recent_reply_overlap`).
6. **Выбор**: `select_scored_candidate` — softmax с
   `candidate_selection_temperature=0.7` среди кандидатов в пределах 0.3
   (`SELECTION_SCORE_MARGIN`) от лучшего скора; t=0 или один кандидат → argmax.
   Именно `SELECTION_SCORE_MARGIN`, а не температура, решает, победит ли лучший
   кандидат: окно отсекает слабых до того, как софтмаксу есть что размазывать.
7. **Пост-обработка**:
   - `capitalize_reply_sentences` (только при `auto_capitalize_replies=true`,
     по умолчанию false);
   - `apply_reply_flavor` (§6.1) с силой `reply_flavor_strength=1.0 ×
     mood.flavor_strength_mult`;
   - `append_emoji_flavor` (§6.2) с шансом `emoji_append_chance=0.15`
     (×1.5 при heated), подавляется после `?`.

Возврат `None` (все 10 попыток пустые/отбракованы) → fallback при обращении,
молчание иначе.

---

## 5. Ядро: `MarkovGenerator` (`core/markov.py:533`)

Модель: пер-чатовая цепь Маркова порядка 3 с бэкоффом 3→2→1, хранение в SQLite
(таблицы переходов n=1/2/3, стартов 2- и 3-грамм; см. §8). LRU-кэши в памяти
(`cache_limit=1024`) по (chat_id, state); инвалидация целиком по чату при каждом
записанном сообщении (`invalidate_chat_cache`).

### 5.1 Параметры сэмплинга из randomness_strength (s∈[0,3])
`_generate_text_once` (`markov.py:1519`):
- `next_explore = min(0.98, 0.12+0.18s)` — вероятность «исследующего» шага;
- `next_power = max(0.15, 0.72−0.16s)` — степень сглаживания частот;
- `start_explore = min(0.98, 0.20+0.20s)`, `start_power = max(0.15, 0.75−0.18s)`.
При исследовании температура сэмплируется треугольно (`sampled_exploration`),
частотная степень дополнительно сплющивается (`exploration_adjusted_power`,
floor 0.02); выбор — через `rng.expovariate` (`exploration_weighted_choice`).

### 5.2 Выбор стартового состояния (приоритет)
1. **Seed** (`_pick_seed_start`, `markov.py:1096`): точная 3-грамма из стартов,
   иначе 2-грамма + один взвешенный шаг. Используется только hot-ngram seed'ом
   (L1) и прямыми вызовами API.
2. **Скрытый контекстный старт** (`_pick_contextual_start` →
   `_select_contextual_state`, `markov.py:795`): срабатывает с вероятностью
   `(bias−1)/bias` от `reply_context_start_bias=2.2` (≈0.545). Каскад:
   - exact 3-граммные окна контекста с переходами (вес: transition_count^power ×
     recency-бонус до +35% для хвостовых окон);
   - exact 2-граммные (+30% recency);
   - casefold 3/2-граммы (при `fuzzy_context_casefold=true`) через
     `ContextStateMatcher`;
   - stem-матчи (при `fuzzy_context_stem=true`, дефолт): состояния модели,
     чей приблизительный стем (`stem_token`, тот же что в
     `context_start_affinity` и IDF-релевантности) совпадает со стемом
     контекстного окна («билеты стоят» ≡ «билет стоит»); сортировка по
     частоте, дубли exact/casefold исключаются.
   Гейт: контекстный старт вообще пробуется лишь с вероятностью
   `context_start_probability(2.2)≈0.545` за попытку (`use_contextual_start`);
   в остальных ~45% сразу глобальный старт. При совпадении и
   `reply_context_emit_start=true` (дефолт) в текст эмитится **хвост** совпавшего
   окна (`context_emission_tokens`) — ответ «подхватывает» контекст вслух,
   `start_source="context"` (**видимый** старт). При `reply_context_emit_start=false`
   тройка не эмитится (`start_source="hidden_context"`, «скрытый» контекст),
   генерация продолжается из состояния. Если контекстный старт пробовался, но ни
   одно окно не совпало — откат на глобальный старт со счётчиком
   `hidden_context_fallbacks` (в трейсе `context=HIDDEN_FALLBACK`).
3. **Глобальный старт** (`_pick_global_start`, `markov.py:1148`): взвешенный
   выбор из всех стартов чата (3-граммы, иначе 2-граммы + шаг). При наличии
   контекста вес старта умножается на
   `CONTEXT_START_AFFINITY^(общие стемы с информативными токенами контекста)`
   — ответы живут в стартах, а не в продолжениях вопроса: контекстный якорь
   для «кто гнойный пидор?» может только пересказать, что шло за вопросом в
   корпусе, а желаемое «слава гнойный пидор …» — это выученный старт. Старт,
   целиком состоящий из стемов контекста, не бустится (это переспрашивание
   вопроса, тот же попугай, что и в эхо-гарде скорера). 1.0 выключает.

### 5.3 Цикл генерации: `_run_generation_loop` (`markov.py:1331`)
До `max_steps=90` шагов, стоп по `max_tokens` (45; 8 в short-режиме) или
`max_chars=280`.

Каждый шаг:
- **Прыжок темы M4**: при ≥5 токенах (`JUMP_MIN_GENERATED_TOKENS`), order 3,
  с шансом `markov_jump_probability=0.12` — новый учёный глобальный старт, в
  текст вклеивается связка из `JUMP_CONNECTIVE_TOKENS` («, кстати»,
  «, короче», …). Не больше `JUMP_MAX_PER_REPLY=1` прыжка на ответ (без капа
  18% победителей несли ≥2 прыжка — это и были бессвязные длинные ответы);
  перед вклейкой `trim_splice_tail` срезает висящую запятую/союз, связка не
  повторяется и не заикается в первое слово нового старта.
- **Переход**: пул 3-граммы; кандидаты, ведущие в уже посещённую тройку,
  фильтруются (окно 40). Пусто → бэкофф на 2-грамму (`enable_backoff=true`);
  order-1 цепи больше нет (удалена вместе с ручкой `backoff_min_order` и
  таблицей `transitions1`, миграция 013): пустой order-2 пул завершает фразу.
- **`weighted_next_choice`** (`markov.py:431`) — сердце сэмплинга. Вес токена:
  - `count^frequency_power`;
  - ×`step_bias`, если токен в контекст-множестве; step_bias =
    `1+(reply_context_bias=1.8 −1)·context_decay(step)`, decay `0.92^step`,
    floor 0.25;
  - ×(1+(step_bias−1)·1.10) за контекстную пару, ×(1+(step_bias−1)·1.25) за
    контекстную тройку;
  - анти-повтор (сила `repetition_penalty_strength=1.0`): делитель
    `1+repeats·0.85·p` по окну последних 10 токенов; ×(1−0.96p) за повтор
    последнего токена, ×(1−0.70p) за предпоследний; ×(1−0.65p) за уже
    виденную пару, ×(1−0.94p) за виденную тройку (окна 80);
  - floor веса 0.01.
- Ранний стоп: `has_degraded_recent_window` — вырождение последних 8 токенов
  (run ≥4 или ≤2 уникальных с доминированием ≥0.75).

### 5.4 Финализация попытки: `_finalize_attempt` (`markov.py:1259`)
Пайплайн хвоста: `trim_repetitive_tail` (срез вырожденного хвоста) →
`trim_to_sentence_boundary` (до последней `.!?` при ≥4 контент-токенах) →
`finalize_reply_ending` (срез плохих последних слов из `BAD_ENDING_WORDS`
и нетерминальной пунктуации, добавление `.`) → `strip_leading_punctuation` →
`detokenize` (пунктуация клеится без пробела, обрез по max_chars).

Реджекты (пустой результат с причиной в trace):
- `result_too_short` — <5 символов;
- `low_diversity` — ≥8 контент-токенов, ≤2 уникальных с доминированием ≥0.8
  или run ≥5;
- `short_context_copy` — короткий (≤3 контент-токенов) ответ, целиком
  содержащийся в контексте;
- `context_heavy` — эхо контекста: все токены из контекста + локальные циклы,
  или overlap ≥0.92 при общем прогоне ≥5, или общий прогон ≥ len−1.

`GenerationTrace` пишется в debug-лог: attempts, order_used, jumps, rejection,
start_source (**global / seed / context / hidden_context**), счётчики
exact/casefold/stem матчей и фолбэков. `context` = видимый контекстный старт
(токены эмитятся, `reply_context_emit_start=true`); `hidden_context` = совпадение
было, но старт не эмитился.

---

## 6. Скоринг и «вкусовые» слои

### 6.1 `score_candidate` (`core/candidate_scorer.py:222`)
`total = completion_quality + natural_length +
context_relevance − repetition_penalty − recent_penalty − verbatim_penalty`:
- **completion_quality**: +0.35 за терминальную пунктуацию, ±0.25/−0.50 за
  (не)сбалансированные скобки/кавычки, −0.80 за плохое последнее слово
  (BAD_ENDING_WORDS) или открывающую скобку в конце;
- **natural_length**: пик 1.0 в полосе режима — short (1,4), medium (5,14),
  long (15,24); ниже — линейный подъём от 0.4, выше — спад до 0.5 за 10 токенов;
- **context_relevance**: заменяется в `response_generator` на
  `idf_context_relevance` — доля информативной IDF-массы контекста, которую
  кандидат возвращает; пересечение и ключи IDF считаются по приблизительным
  стемам (`stem_token`: «гнойному» ≡ «гнойный»), ×1.6, кап 1.6. Чистое эхо
  (стемы кандидата ⊆ стемов контекста) получает 0. Старая формула
  (overlap/|кандидат|) осталась фолбэком при пустом IDF (синтетический eval);
- **repetition_penalty**: 1.6·повторы токенов (короткие ≤3 токена: 1.0·повторы
  + фикс 0.20 — бывшая lexical_diversity вшита в веса) + 1.0·повторы биграмм +
  1.3·повторы триграмм (доли).

### 6.2 Поверхностные слои после выбора
- **`apply_reply_flavor`** (`core/reply_flavor.py:16`): один ролл на концовку —
  25% срезать финальную точку, 7% «...», 5% «!», 4% удвоить `?`/`!`
  (вероятности × strength, кап scale 2.0). Слова не трогаются.
- **`append_emoji_flavor`** (`core/emoji.py:124`): шанс 0.15 (×1.5 heated),
  сэмпл эмодзи из статистики чата с плющением `count^0.5`; не после `?`.

---

## 7. Редкие события (L3), fallback'и и отправка

### 7.1 Редкие события: `roll_rare_event` / `apply_rare_event` (`reply_flavor.py:87`)
Только для сгенерированных (не fallback) ответов, бюджет
`rare_event_daily_cap=3`/чат/день (UTC).
- `rare_event_chance=0.005` → равновероятно: **verdict** (замена ответа словом
  «база/жиза/классика/…»), **caps** (UPPERCASE), **double** (сплит по границе
  предложения на два сообщения);
- иначе `false_start_chance=0.03` → **false_start**: филлер («ну как бы...»,
  «щас», …) + настоящий ответ вторым сообщением.

### 7.2 Fallback-фразы (`presentation/fallback_phrases.py`)
Два пула по 12 фраз: `NOT_ENOUGH_DATA_PHRASES` (мало данных + обращение) и
`GENERATION_FAILED_PHRASES` (генерация не удалась + обращение). Аддитивные
расширения: ночью 00–06 (`LATE_NIGHT_FALLBACK_PHRASES`), при heated
(`HEATED_FALLBACK_PHRASES`). Анти-повтор: последние 3 фразы чата исключаются.
Fallback на обращение не считается в hourly cap.

### 7.3 Отправка: `reply_humanized_sequence` (`handlers/_helpers.py:57`)
Для каждой части: chat action «typing» + пауза
`rand(typing_min_ms=350, typing_max_ms=1100) + typing_per_char_ms=12 × len`,
потолок 4000 мс. Первая часть — reply, последующие — обычные сообщения.

### 7.4 Учёт после отправки
- `note_reply_sent` — cooldown-метка всегда; в hourly-историю только
  непрошеные;
- `note_mention_reply` — метка mention-cooldown;
- `remember_short_reply` (если ≤3 контент-токенов, окно 5) и
  `remember_recent_reply` (окно 20) — анти-повторы следующих генераций.

---

## 8. Обучение и хранение (блок `finally`)

Выполняется **всегда** после ответа/молчания, если сообщение обучаемо:
- `LearningService.record_message` → `Database.save_message_and_update_model`
  (атомарно): сохранение normalized-текста + инкремент счётчиков переходов
  n=1 (пары), n=2 (тройки), n=3 (четвёрки) и стартов starts2/starts3; ретенция
  1000 сообщений/чат (`MESSAGES_RETENTION_PER_CHAT`). Там же лениво крутится
  суточный decay эмодзи/n-грамм (половинение устаревших счётчиков).
- Инвалидация кэшей генератора и текст-кэша verbatim-проверки по чату.
- L1: `extract_content_ngrams` (`core/hot_ngrams.py:34`) — би/триграммы с
  ≥1 контент-токеном (длина ≥3, не STOPWORD, wordlike), ≤24 на сообщение →
  `record_hot_ngrams`.

Ключевые таблицы: messages (normalized), переходы/старты Маркова,
`chat_emoji_stats`, `chat_hot_ngrams`. Кэши в памяти: LRU переходов/стартов
(1024), индексы `ContextStateMatcher` (по (chat, order)), text-кэш 500.

---

## 9. Все ручки генерации (registry, runtime-изменяемые через /set)

| Ручка | Default | Влияние |
|---|---|---|
| reply_probability | 0.08 | legacy-вероятность (без директора) |
| reply_director_enabled | true | M2 momentum-директор |
| reply_probability_min / max | 0.02 / 0.30 | полоса директора |
| reply_burst_boost_sec / mult | 180 / 2.0 | буст после своего ответа |
| reply_burst_suppress_sec / mult | 600 / 0.5 | «отход» после буста |
| reply_max_per_hour | 20 | кап непрошеных ответов |
| min_cooldown_sec | 45 | кулдаун непрошеных |
| mention_cooldown_sec | 5 | анти-флуд обращений (per user) |
| min_tokens_for_model | 200 | порог «модель готова» |
| max_reply_chars / tokens | 280 / 45 | пределы длины |
| normalize_lower | false | lowercase-токенизация |
| auto_capitalize_replies | false | капитализация предложений |
| randomness_strength | 2.0 | база explore/power (§5.1) |
| candidate_selection_temperature | 1.3 | softmax выбора кандидата |
| length_mode_weights | 0.25,0.55,0.2 | веса short/medium/long |
| repetition_penalty_strength | 1.0 | локальные анти-повторы шага |
| recent_reply_penalty_strength | 0.5 | штраф пересечения с прошлыми ответами |
| reply_flavor_strength | 1.0 | вариации концовки |
| emoji_append_chance | 0.15 | M3 эмодзи-хвост |
| markov_order / enable_backoff | 3 / true | порядок цепи |
| markov_jump_probability | 0.12 | M4 прыжки темы |
| use_reply_context | true | контекст из reply |
| reply_context_max_tokens | 12 | окно контекста |
| reply_context_bias / start_bias | 1.8 / 2.2 | сила контекста в шаге/старте |
| reply_context_only_for_replies | true | контекст только для реплаев |
| reply_context_include_current_message | true | + текущее сообщение |
| fuzzy_context_casefold / stem | true / true | нечёткий матчинг контекста |
| hot_ngram_seed_chance / min_count / recency_share | 0.05 / 3 / 0.5 | L1 running jokes |
| rare_event_chance / false_start_chance / rare_event_daily_cap | 0.005 / 0.03 / 3 | L3 |
| mood_enabled / mood_modulation_strength | true / 1.0 | M1 |
| mood_ewma_alpha | 0.3 | сглаживание сигналов |
| mood_lively/sleepy_rate_per_min | 12 / 2 | пороги темпа |
| mood_heated_intensity / mood_max_rate_per_min | 0.4 / 120 | heated-порог, кламп темпа |
| typing_min/max/per_char_ms | 350 / 1100 / 12 | имитация набора |

Константы, зашитые в код (не /set): бюджеты попыток/кандидатов и margin (§4),
`SHORT_MODE_MAX_TOKENS=8`, `max_steps=90`, формулы explore/power, веса скоринга
(§6.1), веса momentum (0.55/0.30/0.15), вероятности flavor-концовок,
`EMOJI_SAMPLE_POWER=0.5`, окна анти-повторов
(5 коротких / 20 полных / 3 fallback), decay-окна эмодзи и n-грамм.

---

## 10. Инструменты для работы с генерацией

- `tools/eval_generation.py` — синтетический eval конвейера (метрики вроде
  context_token_overlap; исторически им подбирались margin, penalty, cap).
- `tools/eval_prod.py` — eval на продовой БД.
- Характеризационные тесты: `tests/test_markov_generation_characterization.py`,
  `tests/test_response_generator.py`, `tests/test_candidate_scorer.py`,
  `tests/test_eval_generation.py` и остальные `tests/test_*` по модулям выше.
- `app/core/gen_trace_log.py` — пошаговый лайв-трейс отбора кандидатов на
  логгере `chat_markov.gen` (INFO): заголовок генерации, по каждой попытке
  маршрут (`start_source`/order/тип контекст-матча/jumps) и разбивка очков,
  таблица финального softmax-выбора с весами/вероятностями. Гейтится уровнем
  INFO (`enabled()`), поведение не меняет. Для читаемой кириллицы в лог-файле
  запускать бота в UTF-8 (`PYTHONUTF8=1`).
