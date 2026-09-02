# Tasks — selection-knobs (M3R-100, шаг 1)

## 1. До правок

- [x] 1.1 `hash-log.md`: ожидание «сдвига нет» (дефолты равны константам, бонус 0)

## 2. Ручки

- [x] 2.1 Реестр, `RuntimeTunables`, `.env.example`: `selection_score_margin`,
      `context_relevance_weight`, `context_relevance_cap`,
      `selection_diversity_bonus`
- [x] 2.2 Скорер: именованные параметры weight/cap; `select_scored_candidate`
      принимает margin; `ResponseGenerator` передаёт ручки; трасса печатает
      действующий margin
- [x] 2.3 `app/core/trajectory.py` (рёбра, перекрытие), `tools/eval/metrics.py`
      импортирует оттуда

## 3. Бонус различности

- [x] 3.1 `CandidateScore.diversity_bonus` в `total`; `apply_diversity_bonus`
      перед отбором (D2, D3)
- [x] 3.2 Тесты: дефолты реестра == константы; порог == `edge_overlap_similar`;
      отличный кандидат поднят, обрезок лучшего не тронут, лучший не тронут;
      бонус 0 — тот же список; margin 0.8 расширяет окно розыгрыша

## 4. Приёмка

- [x] 4.1 `ruff`, `mypy app/`, полный сьют, покрытие
- [x] 4.2 `generation_hash` на обоих снимках, факт в `hash-log.md`
- [x] 4.3 Карта §2.1/§2.2, пайплайн §5–6, `.env.example`
