# Tasks — phrase-route (M3R-210)

## 1. До правок

- [x] 1.1 `hash-log.md`: снять `generation_hash` на синтетике и прод-копии 10.09 до правок и записать ожидание «сдвига нет ни на одном снимке» при дефолте 0 (design D7)

## 2. Сборка вокруг ядра

- [x] 2.1 `app/core/markov.py`: вынести рост головы/хвоста из `generate_seeded_candidate` в `_grow_around(core)`; seeded зовёт его с `[seed, second]` — проверить, что существующие тесты seeded и `generation_hash` на обоих снимках не сдвинулись
- [x] 2.2 `generate_phrase_candidate(chat_id, phrase)` через `_grow_around(list(phrase))`, `None` при отсутствии продолжений с обеих сторон; тест на живой цепи: фраза — непрерывная подпоследовательность токенов кандидата, для биграммы и триграммы (design D1)

## 3. Чтение и ранжирование

- [x] 3.1 `ChatPhraseNgramsRepo.get_phrases_containing(chat_id, tokens, min_count)` — один запрос, полный `ORDER BY cnt DESC, w1, w2, w3`; тест упорядоченности (ловушка §5) и теста, что порог поддержки отсекает; `LearningService.get_phrases_containing` — проброс
- [x] 3.2 `app/core/phrase_route.py`: `rank_phrases(rows, anchors, message_tokens, slots)` — чистая; тесты: копия входа исключена, обход по кругу якорей, срез выбранной фразы пропущен, детерминизм при равной поддержке, ГСЧ не нужен (design D3)

## 4. Маршрут

- [x] 4.1 `CandidateRoute.PHRASE`; ручки `phrase_slot_ratio` (0..0.7, дефолт 0) и `phrase_min_count` (int ≥ 2, дефолт 3) в реестре, `RuntimeTunables`, `.env.example`; тест реестра: значение ниже 2 отвергается; фикстуры `tests/test_response_generator.py` и `tests/test_handlers.py`
- [x] 4.2 `_append_anchored_candidates` принимает якорь-кортеж и выбирает `generate_phrase_candidate` (design D2); `_append_phrase_candidates` + бюджет с клампом после assoc (design D4); счётчики `note_phrase_draw` / `phrase_draws` / `phrase_empty_rate` в телеметрии и `/stats` (`bot_messages` + тест)
- [x] 4.3 Тесты уровня `ResponseGenerator` (по образцу `tests/test_assoc_route.py`): ratio 0 — ни чтения, ни розыгрыша, attempted 0; два слота — два кандидата `phrase` разных якорей, пул ≤ target, обход присутствует; пустой набор — attempted без present, `phrase_empty` 1; четыре маршрута включены при пуле 5 — обход держит слот

## 5. Замер

- [x] 5.1 `tools/eval/matrix_phrase_route.yaml`: C0, C11p2 / C11p3 / C11p5 (ratio 0.4, `verbatim_recognized_unit: true`), контроль C11v; `knob_census.GATED_BY` — `phrase_min_count` под `phrase_slot_ratio: 0.4`; README `tools/eval`; проверить `python -m tools.eval --smoke`
- [x] 5.2 Прогон грида в обоих режимах на прод-копии 10.09 (`--context-mode ctx` / `noctx`); заметка `docs/eval_reports/eval_2026-09-<дд>_phrase-route-verdict.md` по образцу `eval_2026-09-02_route-gate-verdict.md`: coverage по плечам, must-improve, must-not-worsen, ECB, p95, вклад контроля C11v; вердикт словами гейта

## 6. Приёмка

- [x] 6.1 `ruff`, `mypy app/`, полный `unittest`, покрытие ≥ храповика
- [x] 6.2 `generation_hash` на обоих снимках после правок, факт в `hash-log.md`; `tests/test_generation_hash_baseline.py` зелёный
- [x] 6.3 Документы: роадмап (M3R-210 — реализован, вердикт), `docs/GENERATION_MAP.md` §1.5/§2.1 (маршрут, ручки, счётчики), `docs/GENERATION_PIPELINE.md` §5, `docs/CLOSED.md`; `docs/OPEN.md` — только если вердикт оставляет решение владельцу
