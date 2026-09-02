# Tasks — assoc-route-pilot (M3R-200, пилот)

## 1. До правок

- [x] 1.1 `hash-log.md`: ожидание «сдвига нет» при дефолте 0
- [x] 1.2 Блок `assoc_pilot` в `eval_thresholds.yaml` — планки жизнеспособности
      с обоснованием, до прогона (design D5)

## 2. Маршрут

- [x] 2.1 `CandidateRoute.ASSOC`; ручка `assoc_slot_ratio` (реестр,
      `RuntimeTunables`, `.env.example`); фикстуры тестов
- [x] 2.2 Порт и репозиторий: `get_seed_backward`, `get_model_volume` в
      `MarkovReadPort`; тест упорядоченности чтения (ловушка §5)
- [x] 2.3 `MarkovGenerator.rank_associates` (design D2) + тесты на живой цепи:
      сосед по PMI находится с обеих сторон, якоря и токены входа исключены,
      порог поддержки, обход по кругу якорей
- [x] 2.4 `_append_anchored_candidates` — общий helper seeded/assoc (D1);
      бюджет с гарантией слота обходу (D3); атрибуция `assoc`; `attempted`
      при бюджете > 0; счётчики розыгрыша в телеметрии и `/stats`
- [x] 2.5 Тесты: ratio 0 — ни чтения, ни розыгрыша; два слота — два ассоциата
      разных якорей; пул ≤ target и обход присутствует при трёх маршрутах;
      пустой набор — attempted без present; RNG-нейтральность при 0

## 3. Замер

- [x] 3.1 `tools/eval/report.py`: вердикт пилота (префикс C10, четыре вопроса,
      `viable` / `not viable` / `insufficient data`), тесты
- [x] 3.2 `matrix_assoc_pilot.yaml` (C0, C10a20, C10a40); прогон в обоих
      режимах на прод-копии 01.09; заметка `eval_<дата>_assoc-pilot.md`

## 4. Приёмка

- [x] 4.1 `ruff`, `mypy app/`, полный сьют, покрытие
- [x] 4.2 `generation_hash` на обоих снимках, факт в `hash-log.md`
- [x] 4.3 Роадмап (M3R-200 пилот), карта §1.5/§2.1, пайплайн §5, CLOSED,
      README `tools/eval`, `knob_census` (новая ручка)
