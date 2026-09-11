## REMOVED Requirements

### Requirement: Shadow order-4 selection is measured without an order-4 index

**Reason**: Фаза 7 закрыта 2026-08-12 отрицательным вердиктом (0 выборов
order-4 на 5937 подходящих шагов, `eval_2026-08-12_phase7-verdict.md`); тень
после этого ничего не питала, а её оконный индекс строился и пополнялся на
каждом сообщении. Перепись ручек 2026-09-11 — inert. Решение владельца (O22,
группа 1).

**Migration**: Ручка `markov_shadow_order4_enabled`, счётчики
`shadow_order4_*` в телеметрии и `/stats`, модуль `shadow_order.py` и метод
`get_order4_shadow_index` удалены. Переоткрытие фазы 7 — только с новыми
данными и заново спроектированным инструментом; старый код — в истории git
до 2026-09-11.
