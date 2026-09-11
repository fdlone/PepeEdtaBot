## REMOVED Requirements

### Requirement: The shadow order-4 gate renders a verdict once its sample suffices

**Reason**: Гейт вынес вердикт 2026-08-12 (`fail`: 0.0% при пороге 10% на
5937 шагах) и фазу 7 закрыл; с удалением теневого селектора
(`remove-dead-knobs`, 2026-09-11) у гейта не осталось источника данных.
Порог `phase7_order4` снят из `eval_thresholds.yaml` вместе с инструментом —
это не смена порога после прогона, а вывод из строя закрытого гейта.

**Migration**: Отчёт больше не печатает строку `phase7_order4`; вердикт фазы
7 остаётся в `docs/v2/00_STATUS.md` и `eval_2026-08-12_phase7-verdict.md`.
