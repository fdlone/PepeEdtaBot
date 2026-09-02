## Why

M3R-100 (ревизия скорера) стоит в роадмапе как P2 «фоном», а измерения двух
дней сходятся в одной точке: пул разнообразен (ECB ~4.5 из 5), окно отбора в
ctx держит ~2 траектории, и у 36% входов альтернативы нет вовсе (M3R-011);
каждая контекстная ручка, поднимавшая тематичность, сужала окно (M3R-110).
Узкое место — отбор, а не цепь. При этом три величины, которые окно и
определяют, — `SELECTION_SCORE_MARGIN`, `CONTEXT_RELEVANCE_WEIGHT`,
`CONTEXT_RELEVANCE_CAP` — константы модулей: матрица eval их не видит, а
`eval_prod` подменяет их манки-патчем. Ни свипа, ни переписи на них не
построить. Решение владельца 2026-09-02: делать M3R-100.

## What Changes

- Три константы становятся ручками реестра с прежними дефолтами:
  `selection_score_margin` (0.3), `context_relevance_weight` (1.6),
  `context_relevance_cap` (1.6). Функции скорера принимают их именованными
  параметрами с прежними значениями по умолчанию — вызовы без параметров не
  меняются.
- Новый механизм за нейтральной ручкой `selection_diversity_bonus` (дефолт 0):
  после скоринга кандидат, чья траектория **существенно отлична** от лучшей
  (перекрытие рёбер ниже порога `structural_escape.edge_overlap_similar`),
  получает бонус `bonus × (1 − перекрытие)`. Это candidate-level: он не
  меняет прогулку, он меняет, кто попадает в окно отбора, — ровно то место,
  где M3R-011 нашёл схлопывание. При 0 ни одного вычисления, ни одного
  обращения к ГСЧ.
- `trajectory_edges` / `edge_overlap` переезжают из `tools/eval/metrics.py` в
  `app/core/trajectory.py`; харнесс импортирует их оттуда — одно определение
  «той же траектории» для гейта и для механизма.
- Трасса и телеметрия печатают действующий margin (из ручки).
- Дефолты не меняются: `generation_hash` без сдвига (ожидание до правок).
  Замер — отдельный change `selection-window-grid` с пре-регистрированным
  гейтом.

## Capabilities

### New Capabilities
- `generation-selection-window`: окно отбора и его ручки; бонус различности
  как candidate-level механизм; нейтральность при дефолтах.

### Modified Capabilities
<!-- нет -->

## Impact

- `app/config/registry.py`, `app/config/settings.py`, `.env.example`;
  `app/core/candidate_scorer.py` (параметры, компонент `diversity_bonus`),
  `app/core/response_generator.py` (margin из ручки, бонус перед отбором),
  новый `app/core/trajectory.py`; `tools/eval/metrics.py` (импорт); тесты.
- `docs/GENERATION_MAP.md` §2.1/§2.2, `docs/GENERATION_PIPELINE.md` §5–6.
