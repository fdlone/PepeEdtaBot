## REMOVED Requirements

### Requirement: The L1 seed draw is modelled in the harness and gated

**Reason**: Легаси-розыгрыш одной затравки на ответ удалён из конвейера
(`hot-channel-write-gate`, 2026-09-11): при включённом hot-маршруте он до
прогулки не доходил, замер без него побайтно равен базлайну
(`eval_2026-09-11_legacy-seed-check_noctx.md`). Горячий канал доходит до
генерации только через маршрут, который харнесс исполняет через сам
`ResponseGenerator`, — моделировать в харнессе больше нечего.

**Migration**: `draw_hot_seed`, `HOT_SEED_RNG_OFFSET` и поле `seed_drawn`
записи удалены; coverage гейта `l1_hot_channel` по-прежнему читается из
`start_source` победителя.
