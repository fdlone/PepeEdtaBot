# Перепись ручек (M3R-151) — 2026-09-11

Change `knob-census`. Правило классов — `eval_thresholds.yaml` → `knob_census`, зарегистрировано до прогона. Каждая ручка — на экстремумах домена (булева — инверсия) против C0, парные дельты, оба режима; ручки с родителем — ещё и при включённом родителе. Метрики классификации: `context_affinity_without_copy`, `exact_context_copy_rate`, `repetition_rate`, `historical_meme_rate`, `structural_window_escape`, `structural_pool_ecb`, `mean_response_length`.

- C0 `ctx`: 1500 записей, версия промптов `bababb4b7693`, p95 64.3 мс
- C0 `noctx`: 1500 записей, версия промптов `bababb4b7693`, p95 47.6 мс

Латентность в таблице справочная: армы считались параллельно и между собой по ней не сравнимы. Классы: dead — не читается; gated — двигает только при включённом родителе; inert — интервалы всех дельт внутри полосы допуска на всех экстремумах; strong — значимая дельта не ниже планки силы; weak — остальное.

## Сводка по ручкам

| ручка | класс | предложение |
|---|---|---|
| `max_reply_chars` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `max_reply_tokens` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `auto_capitalize_replies` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `randomness_strength` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `candidate_selection_temperature` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `selection_score_margin` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `context_relevance_weight` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `context_relevance_cap` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `selection_diversity_bonus` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `selection_diversity_bonus_noctx` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `reply_flavor_strength` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `emoji_append_chance` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `repetition_penalty_strength` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `recent_reply_penalty_strength` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `verbatim_penalty_strength` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `verbatim_recognized_unit` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `intonation_profile_strength` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `length_context_adaptation` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `markov_order` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_cache_incremental` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `markov_shadow_order4_enabled` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `markov_entropy_temp_gain` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `markov_entropy_pivot` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_entropy_temp_min` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_entropy_temp_max` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_branching_degenerate_max` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_branching_candidate_floor` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_short_half_life_days` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_long_compression_beta` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_alpha_calm` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `markov_interp_order2_weight` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `markov_collocation_bonus` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_collocation_break_penalty` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_hot_ngram_meme_ordering` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `markov_seeded_candidate_ratio` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `hot_ngram_slot_ratio` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `assoc_slot_ratio` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `phrase_slot_ratio` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `phrase_min_count` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_seed_branch_min` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_branch_ideal` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_branch_max` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_min_support` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_min_score` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_min_token_len` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_seed_head_share` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `enable_backoff` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_jump_probability` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `context_jump_boost` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `verbatim_extension_share` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `order_mix_probability` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `slot_mutation_probability` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `hot_ngram_min_count` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `hot_ngram_recency_share` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `fuzzy_context_casefold` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `reply_context_bias` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `reply_context_start_bias` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `generation_attempts_with_context` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `context_start_affinity` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `context_anchor_splice_probability` | **strong** | keep; check the domain ceiling (an extreme may break form) |

Итого: gated 11, inert 7, strong 30, weak 12.

## Не свипуются (читаются ядром)

- `normalize_lower` — tokenization of the LEARNED corpus, not a generation knob; the copy is lowercased already
- `length_mode_weights` — composite value (three weights); sweep needs a grid of its own
- `markov_long_compression` — string enum (log|pow); measured as the parent of its beta
- `markov_alpha_sleepy` — mood-gated: the harness runs in the neutral mood, which reads alpha_calm
- `markov_alpha_lively` — mood-gated: the harness runs in the neutral mood, which reads alpha_calm
- `markov_alpha_heated` — mood-gated: the harness runs in the neutral mood, which reads alpha_calm

## Вне оффлайн-замера (не читаются ядром генерации)

| ручка | где читается | статический класс |
|---|---|---|
| `reply_probability` | app/core/reply_policy.py, app/handlers/admin.py, app/services/reply_pipeline.py | outside |
| `min_cooldown_sec` | app/core/reply_policy.py, app/services/reply_pipeline.py | outside |
| `min_tokens_for_model` | app/core/reply_policy.py, app/services/reply_pipeline.py | outside |
| `typing_min_ms` | app/handlers/_helpers.py | outside |
| `typing_max_ms` | app/handlers/_helpers.py | outside |
| `typing_per_char_ms` | app/handlers/_helpers.py | outside |
| `markov_meme_min_joint_count` | app/services/reply_pipeline.py | outside |
| `markov_meme_min_support` | app/services/reply_pipeline.py | outside |
| `markov_meme_recency_days` | app/services/reply_pipeline.py | outside |
| `markov_collocation_max_entries` | app/services/reply_pipeline.py | outside |
| `hot_ngram_seed_chance` | app/services/reply_pipeline.py | outside |
| `rare_event_chance` | app/services/reply_pipeline.py | outside |
| `false_start_chance` | app/core/reply_flavor.py, app/services/reply_pipeline.py | outside |
| `rare_event_daily_cap` | app/config/runtime_state.py | outside |
| `user_quirk_chance` | app/services/reply_pipeline.py | outside |
| `user_quirk_min_interactions` | app/handlers/admin.py, app/services/reply_pipeline.py | outside |
| `user_quirk_name_share` | app/services/reply_pipeline.py | outside |
| `use_reply_context` | app/services/reply_pipeline.py | outside |
| `reply_context_max_tokens` | app/services/reply_pipeline.py | outside |
| `reply_context_only_for_replies` | app/services/reply_pipeline.py | outside |
| `reply_context_include_current_message` | app/services/reply_pipeline.py | outside |
| `pivo_recent_pool_window` | app/handlers/pivo.py | outside |
| `pivo_temporal_flavor_chance` | app/handlers/pivo.py | outside |
| `pivo_mention_by_id` | app/handlers/pivo.py | outside |
| `pivo_report_to_owner` | app/handlers/pivo.py | outside |
| `mood_enabled` | app/services/reply_pipeline.py | outside |
| `mood_modulation_strength` | app/core/mood.py, app/services/reply_pipeline.py | outside |
| `mood_ewma_alpha` | app/config/runtime_state.py | outside |
| `mood_lively_rate_per_min` | app/config/runtime_state.py, app/services/reply_pipeline.py | outside |
| `mood_sleepy_rate_per_min` | app/config/runtime_state.py | outside |
| `mood_heated_intensity` | app/config/runtime_state.py | outside |
| `mood_mention_heated_share` | app/config/runtime_state.py | outside |
| `mood_max_rate_per_min` | app/config/runtime_state.py | outside |
| `reply_director_enabled` | app/handlers/admin.py, app/services/reply_pipeline.py | outside |
| `reply_probability_min` | app/handlers/admin.py, app/services/reply_pipeline.py | outside |
| `reply_probability_max` | app/handlers/admin.py, app/services/reply_pipeline.py | outside |
| `reply_burst_boost_sec` | app/services/reply_pipeline.py | outside |
| `reply_burst_boost_mult` | app/services/reply_pipeline.py | outside |
| `reply_burst_suppress_sec` | app/services/reply_pipeline.py | outside |
| `reply_burst_suppress_mult` | app/services/reply_pipeline.py | outside |
| `reply_max_per_hour` | app/services/reply_pipeline.py | outside |
| `mention_cooldown_sec` | app/core/generation_telemetry.py, app/services/reply_pipeline.py | outside |
| `mention_max_per_hour` | app/core/generation_telemetry.py, app/services/reply_pipeline.py | outside |

## Разбор по ручкам

### `max_reply_chars` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=20 | ctx | strong | -0.130 [-0.142, -0.115]* | -0.143 [-0.163, -0.123]* | -0.001 [-0.002, +0.000] | -0.157 [-0.208, -0.107]* | +0.087 [+0.008, +0.161]* | +0.257 [+0.227, +0.285]* | -7.598 [-7.861, -7.334]* |
| min=20 | noctx | strong | -0.048 [-0.057, -0.038]* | +0.011 [+0.005, +0.019]* | +0.000 [+0.000, +0.000] | -0.101 [-0.144, -0.061]* | -0.293 [-0.360, -0.225]* | +0.259 [+0.235, +0.287]* | -6.974 [-7.209, -6.739]* |
| max=4000 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4000 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `max_reply_tokens` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1 | ctx | strong | -0.268 [-0.286, -0.251]* | -0.149 [-0.169, -0.129]* | -0.001 [-0.002, +0.000] | -0.260 [-0.305, -0.215]* | -1.584 [-1.644, -1.525]* | -3.699 [-3.727, -3.669]* | -10.192 [-10.511, -9.901]* |
| min=1 | noctx | strong | -0.082 [-0.093, -0.072]* | +0.112 [+0.094, +0.129]* | +0.000 [+0.000, +0.000] | -0.161 [-0.199, -0.125]* | -2.349 [-2.392, -2.299]* | -3.691 [-3.717, -3.663]* | -9.446 [-9.681, -9.222]* |
| max=300 | ctx | weak | +0.000 [-0.006, +0.007] | -0.003 [-0.017, +0.011] | +0.001 [+0.000, +0.002] | +0.000 [-0.035, +0.035] | -0.011 [-0.047, +0.023] | +0.004 [-0.014, +0.020] | +0.008 [-0.195, +0.235] |
| max=300 | noctx | weak | -0.001 [-0.006, +0.003] | -0.001 [-0.003, +0.000] | +0.001 [+0.000, +0.002] | +0.011 [-0.021, +0.040] | -0.013 [-0.042, +0.015] | +0.004 [-0.012, +0.020] | +0.051 [-0.127, +0.247] |

### `auto_capitalize_replies` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=True | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flip=True | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `randomness_strength` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | +0.010 [-0.004, +0.026] | +0.005 [-0.017, +0.029] | +0.003 [+0.000, +0.006] | +0.011 [-0.043, +0.064] | -0.053 [-0.127, +0.015] | +0.007 [-0.025, +0.045] | +0.074 [-0.245, +0.407] |
| min=0.0 | noctx | strong | -0.004 [-0.015, +0.008] | -0.002 [-0.006, +0.001] | +0.000 [+0.000, +0.000] | +0.053 [+0.005, +0.099]* | +0.047 [-0.009, +0.105] | +0.020 [-0.013, +0.054] | +0.092 [-0.225, +0.421] |
| max=3.0 | ctx | weak | +0.005 [-0.011, +0.019] | -0.009 [-0.031, +0.014] | +0.001 [-0.001, +0.003] | +0.016 [-0.037, +0.069] | -0.028 [-0.101, +0.044] | +0.033 [-0.003, +0.069] | -0.054 [-0.379, +0.265] |
| max=3.0 | noctx | weak | +0.003 [-0.007, +0.014] | -0.001 [-0.005, +0.002] | +0.000 [+0.000, +0.000] | +0.011 [-0.032, +0.059] | +0.016 [-0.042, +0.077] | -0.018 [-0.052, +0.017] | +0.057 [-0.236, +0.369] |

### `candidate_selection_temperature` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | +0.013 [+0.006, +0.020]* | +0.024 [+0.013, +0.036]* | +0.000 [+0.000, +0.000] | +0.000 [-0.037, +0.035] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.032 [-0.260, +0.204] |
| min=0.0 | noctx | weak | +0.001 [-0.008, +0.011] | -0.001 [-0.005, +0.003] | +0.000 [+0.000, +0.000] | +0.013 [-0.029, +0.056] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.025 [-0.238, +0.219] |
| max=3.0 | ctx | weak | +0.000 [-0.002, +0.002] | -0.001 [-0.005, +0.002] | +0.000 [+0.000, +0.000] | +0.003 [-0.005, +0.011] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.055 [-0.009, +0.131] |
| max=3.0 | noctx | weak | +0.002 [-0.001, +0.004] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.003 [-0.013, +0.019] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.037 [-0.041, +0.122] |

### `selection_score_margin` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | +0.013 [+0.006, +0.021]* | +0.024 [+0.013, +0.036]* | +0.000 [+0.000, +0.000] | +0.000 [-0.037, +0.035] | -1.470 [-1.530, -1.415]* | +0.000 [+0.000, +0.000] | -0.050 [-0.279, +0.189] |
| min=0.0 | noctx | strong | -0.004 [-0.014, +0.005] | -0.001 [-0.005, +0.003] | +0.000 [+0.000, +0.000] | +0.016 [-0.024, +0.061] | -2.095 [-2.140, -2.048]* | +0.000 [+0.000, +0.000] | -0.038 [-0.255, +0.199] |
| max=3.0 | ctx | strong | -0.046 [-0.056, -0.037]* | -0.053 [-0.067, -0.041]* | +0.001 [+0.000, +0.003] | -0.056 [-0.088, -0.021]* | +2.103 [+2.043, +2.159]* | +0.000 [+0.000, +0.000] | +0.196 [-0.040, +0.435] |
| max=3.0 | noctx | strong | +0.002 [-0.004, +0.007] | +0.005 [+0.002, +0.009]* | +0.003 [+0.001, +0.007]* | +0.019 [-0.016, +0.051] | +1.333 [+1.287, +1.380]* | +0.000 [+0.000, +0.000] | +0.687 [+0.448, +0.919]* |

### `context_relevance_weight` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.106 [-0.121, -0.093]* | -0.091 [-0.105, -0.076]* | -0.001 [-0.002, +0.000] | -0.072 [-0.109, -0.032]* | +0.810 [+0.749, +0.866]* | +0.000 [+0.000, +0.000] | -0.507 [-0.727, -0.297]* |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | ctx | strong | +0.011 [+0.006, +0.017]* | +0.028 [+0.018, +0.039]* | +0.001 [+0.000, +0.002] | -0.019 [-0.048, +0.008] | -0.332 [-0.381, -0.286]* | +0.000 [+0.000, +0.000] | +0.377 [+0.213, +0.544]* |
| max=4.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `context_relevance_cap` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.106 [-0.121, -0.093]* | -0.091 [-0.105, -0.076]* | -0.001 [-0.002, +0.000] | -0.072 [-0.109, -0.032]* | +0.810 [+0.749, +0.866]* | +0.000 [+0.000, +0.000] | -0.507 [-0.727, -0.297]* |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `selection_diversity_bonus` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | +0.015 [+0.011, +0.020]* | +0.018 [+0.011, +0.027]* | +0.000 [+0.000, +0.000] | +0.008 [-0.016, +0.032] | -0.425 [-0.463, -0.387]* | +0.000 [+0.000, +0.000] | -0.015 [-0.193, +0.155] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | ctx | strong | -0.063 [-0.073, -0.053]* | -0.094 [-0.110, -0.078]* | +0.001 [+0.000, +0.003] | -0.067 [-0.107, -0.029]* | -0.321 [-0.382, -0.266]* | +0.000 [+0.000, +0.000] | +0.136 [-0.106, +0.400] |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `selection_diversity_bonus_noctx` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.003 [-0.006, +0.000] | +0.000 [+0.000, +0.000] | -0.002 [-0.006, +0.000] |
| min=0.0 | noctx | strong | +0.007 [+0.003, +0.011]* | -0.002 [-0.005, +0.000] | +0.000 [+0.000, +0.000] | +0.008 [-0.016, +0.032] | -0.326 [-0.363, -0.295]* | +0.000 [+0.000, +0.000] | -0.099 [-0.247, +0.057] |
| max=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.007 [-0.012, -0.001]* | +0.000 [+0.000, +0.000] | -0.001 [-0.027, +0.024] |
| max=1.0 | noctx | strong | -0.011 [-0.019, -0.002]* | +0.004 [+0.000, +0.009] | +0.000 [+0.000, +0.000] | +0.037 [+0.000, +0.075] | -0.949 [-0.971, -0.928]* | +0.000 [+0.000, +0.000] | +0.359 [+0.157, +0.582]* |

### `reply_flavor_strength` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=2.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=2.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `emoji_append_chance` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `repetition_penalty_strength` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | +0.003 [-0.003, +0.009] | -0.006 [-0.021, +0.007] | +0.001 [+0.000, +0.003] | +0.019 [-0.011, +0.051] | +0.017 [-0.020, +0.054] | +0.009 [-0.007, +0.025] | +0.059 [-0.141, +0.285] |
| min=0.0 | noctx | weak | +0.002 [-0.003, +0.008] | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [-0.027, +0.027] | +0.005 [-0.023, +0.032] | -0.007 [-0.023, +0.007] | +0.005 [-0.161, +0.163] |
| max=3.0 | ctx | weak | +0.002 [-0.002, +0.007] | +0.002 [-0.008, +0.012] | +0.001 [+0.000, +0.002] | -0.008 [-0.035, +0.016] | +0.001 [-0.025, +0.026] | +0.013 [-0.001, +0.029] | -0.028 [-0.195, +0.134] |
| max=3.0 | noctx | weak | +0.001 [-0.003, +0.005] | -0.001 [-0.002, +0.000] | +0.000 [+0.000, +0.000] | +0.016 [-0.003, +0.035] | -0.001 [-0.024, +0.023] | -0.002 [-0.013, +0.009] | -0.013 [-0.139, +0.121] |

### `recent_reply_penalty_strength` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=3.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=3.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `verbatim_penalty_strength` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | +0.001 [-0.008, +0.010] | -0.003 [-0.015, +0.011] | +0.000 [-0.002, +0.002] | +0.011 [-0.024, +0.048] | +0.171 [+0.125, +0.219]* | +0.012 [-0.011, +0.035] | -0.528 [-0.763, -0.287]* |
| min=0.0 | noctx | strong | -0.005 [-0.013, +0.006] | -0.002 [-0.005, +0.001] | +0.000 [+0.000, +0.000] | +0.029 [-0.016, +0.069] | +0.483 [+0.436, +0.528]* | +0.018 [-0.006, +0.044] | -1.353 [-1.609, -1.102]* |
| max=3.0 | ctx | weak | -0.004 [-0.008, -0.001]* | -0.005 [-0.009, +0.000] | +0.000 [+0.000, +0.000] | -0.011 [-0.024, +0.000] | -0.069 [-0.087, -0.051]* | +0.000 [+0.000, +0.000] | -0.101 [-0.205, -0.005]* |
| max=3.0 | noctx | weak | -0.002 [-0.004, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.003 [-0.019, +0.011] | -0.115 [-0.134, -0.097]* | +0.000 [+0.000, +0.000] | -0.073 [-0.151, -0.004]* |

### `verbatim_recognized_unit` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=False | ctx | weak | -0.001 [-0.004, +0.001] | -0.001 [-0.004, +0.003] | +0.000 [+0.000, +0.000] | -0.003 [-0.016, +0.011] | -0.048 [-0.070, -0.027]* | +0.000 [+0.000, +0.000] | +0.127 [+0.045, +0.215]* |
| flip=False | noctx | weak | -0.003 [-0.006, +0.001] | +0.001 [+0.000, +0.002] | +0.000 [+0.000, +0.000] | +0.027 [+0.011, +0.045]* | -0.067 [-0.091, -0.041]* | +0.000 [+0.000, +0.000] | +0.267 [+0.165, +0.367]* |

### `intonation_profile_strength` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=1.0 | ctx | weak | +0.002 [-0.005, +0.009] | -0.009 [-0.019, +0.000] | -0.001 [-0.002, +0.000] | -0.021 [-0.045, +0.000] | +0.041 [+0.010, +0.075]* | -0.002 [-0.015, +0.011] | -0.885 [-1.036, -0.743]* |
| max=1.0 | noctx | weak | +0.003 [-0.002, +0.008] | +0.003 [+0.001, +0.006]* | +0.000 [+0.000, +0.000] | -0.003 [-0.027, +0.019] | +0.029 [+0.000, +0.060] | -0.011 [-0.025, +0.002] | -0.648 [-0.808, -0.514]* |

### `length_context_adaptation` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | -0.001 [-0.006, +0.005] | +0.004 [-0.004, +0.013] | +0.001 [+0.000, +0.002] | +0.005 [-0.013, +0.024] | -0.003 [-0.035, +0.025] | -0.003 [-0.017, +0.009] | +0.253 [+0.111, +0.417]* |
| min=0.0 | noctx | weak | -0.004 [-0.009, +0.000] | +0.001 [-0.001, +0.004] | +0.000 [+0.000, +0.000] | -0.011 [-0.032, +0.011] | -0.031 [-0.058, -0.003]* | +0.016 [+0.003, +0.030]* | +0.265 [+0.125, +0.412]* |
| max=3.0 | ctx | weak | +0.001 [-0.005, +0.008] | -0.005 [-0.011, +0.002] | -0.001 [-0.002, +0.000] | +0.003 [-0.013, +0.019] | +0.025 [-0.002, +0.056] | +0.004 [-0.007, +0.017] | -0.273 [-0.415, -0.123]* |
| max=3.0 | noctx | weak | +0.004 [-0.001, +0.009] | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [-0.019, +0.019] | +0.038 [+0.011, +0.065]* | -0.013 [-0.025, +0.000] | -0.160 [-0.295, -0.031]* |

### `markov_order` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| alt=2 | ctx | strong | -0.009 [-0.023, +0.006] | -0.032 [-0.051, -0.011]* | +0.001 [+0.000, +0.002] | +0.011 [-0.040, +0.067] | -0.013 [-0.081, +0.054] | -0.015 [-0.052, +0.020] | -0.197 [-0.506, +0.123] |
| alt=2 | noctx | weak | -0.005 [-0.016, +0.005] | +0.001 [-0.003, +0.005] | +0.000 [+0.000, +0.000] | -0.040 [-0.083, +0.005] | -0.039 [-0.097, +0.009] | -0.010 [-0.045, +0.026] | -0.191 [-0.477, +0.111] |

### `markov_cache_incremental` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=False | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flip=False | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `markov_shadow_order4_enabled` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=False | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flip=False | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `markov_entropy_temp_gain` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=-2.0 | ctx | weak | +0.007 [-0.005, +0.019] | -0.009 [-0.029, +0.011] | +0.001 [-0.001, +0.003] | +0.005 [-0.048, +0.056] | +0.015 [-0.044, +0.072] | +0.020 [-0.007, +0.049] | +0.244 [-0.064, +0.560] |
| min=-2.0 | noctx | weak | +0.002 [-0.007, +0.012] | -0.001 [-0.003, +0.002] | +0.000 [+0.000, +0.000] | +0.005 [-0.043, +0.051] | -0.020 [-0.065, +0.029] | -0.015 [-0.043, +0.012] | +0.258 [-0.001, +0.553] |
| max=2.0 | ctx | weak | +0.002 [-0.006, +0.012] | -0.007 [-0.025, +0.011] | +0.000 [-0.002, +0.002] | -0.024 [-0.069, +0.021] | -0.005 [-0.061, +0.044] | +0.020 [-0.004, +0.043] | +0.053 [-0.208, +0.297] |
| max=2.0 | noctx | weak | +0.003 [-0.005, +0.010] | -0.002 [-0.005, +0.000] | +0.000 [+0.000, +0.000] | +0.021 [-0.021, +0.061] | -0.004 [-0.045, +0.034] | -0.009 [-0.031, +0.013] | -0.151 [-0.380, +0.082] |

### `markov_entropy_pivot` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 (parent on) | ctx | weak | +0.003 [-0.007, +0.011] | -0.006 [-0.023, +0.010] | +0.000 [-0.002, +0.002] | -0.011 [-0.056, +0.035] | +0.021 [-0.025, +0.067] | +0.020 [-0.003, +0.042] | +0.160 [-0.087, +0.415] |
| min=0.0 (parent on) | noctx | weak | -0.002 [-0.008, +0.005] | -0.001 [-0.004, +0.001] | +0.000 [+0.000, +0.000] | +0.011 [-0.027, +0.048] | -0.007 [-0.045, +0.029] | -0.017 [-0.035, +0.001] | -0.067 [-0.271, +0.136] |
| max=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 (parent on) | ctx | weak | -0.002 [-0.007, +0.003] | -0.008 [-0.017, +0.001] | +0.001 [+0.000, +0.002] | +0.021 [-0.003, +0.045] | +0.021 [-0.003, +0.046] | -0.002 [-0.014, +0.011] | +0.029 [-0.095, +0.158] |
| max=1.0 (parent on) | noctx | weak | -0.004 [-0.008, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.003 [-0.013, +0.019] | -0.005 [-0.023, +0.015] | -0.003 [-0.013, +0.008] | +0.051 [-0.062, +0.172] |

### `markov_entropy_temp_min` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.05 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.05 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.05 (parent on) | ctx | weak | +0.001 [-0.008, +0.009] | -0.005 [-0.021, +0.011] | -0.001 [-0.002, +0.000] | -0.021 [-0.061, +0.019] | +0.012 [-0.029, +0.057] | +0.017 [-0.005, +0.039] | +0.087 [-0.156, +0.338] |
| min=0.05 (parent on) | noctx | weak | +0.003 [-0.003, +0.009] | -0.001 [-0.004, +0.003] | +0.000 [+0.000, +0.000] | +0.021 [-0.013, +0.056] | -0.007 [-0.043, +0.031] | -0.018 [-0.037, +0.001] | -0.022 [-0.217, +0.174] |
| max=50.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=50.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=50.0 (parent on) | ctx | weak | +0.000 [-0.010, +0.011] | +0.007 [-0.013, +0.027] | +0.001 [-0.001, +0.003] | -0.027 [-0.075, +0.027] | -0.017 [-0.071, +0.040] | +0.016 [-0.009, +0.039] | +0.075 [-0.211, +0.329] |
| max=50.0 (parent on) | noctx | weak | +0.004 [-0.004, +0.014] | -0.001 [-0.003, +0.000] | +0.001 [+0.000, +0.002] | +0.013 [-0.032, +0.053] | +0.012 [-0.031, +0.059] | -0.009 [-0.031, +0.016] | +0.048 [-0.214, +0.310] |

### `markov_entropy_temp_max` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.05 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.05 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.05 (parent on) | ctx | weak | +0.007 [-0.005, +0.019] | -0.009 [-0.029, +0.011] | +0.001 [-0.001, +0.003] | +0.008 [-0.048, +0.061] | +0.013 [-0.047, +0.069] | +0.022 [-0.005, +0.050] | +0.223 [-0.095, +0.538] |
| min=0.05 (parent on) | noctx | weak | +0.000 [-0.009, +0.011] | -0.001 [-0.003, +0.002] | +0.000 [+0.000, +0.000] | +0.005 [-0.043, +0.051] | -0.020 [-0.065, +0.031] | -0.016 [-0.043, +0.012] | +0.289 [+0.023, +0.587]* |
| max=50.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=50.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=50.0 (parent on) | ctx | weak | +0.001 [-0.008, +0.009] | -0.005 [-0.021, +0.011] | -0.001 [-0.002, +0.000] | -0.021 [-0.061, +0.019] | +0.012 [-0.029, +0.057] | +0.017 [-0.005, +0.039] | +0.087 [-0.156, +0.338] |
| max=50.0 (parent on) | noctx | weak | +0.003 [-0.003, +0.009] | -0.001 [-0.004, +0.003] | +0.000 [+0.000, +0.000] | +0.021 [-0.013, +0.056] | -0.007 [-0.043, +0.031] | -0.018 [-0.037, +0.001] | -0.022 [-0.217, +0.174] |

### `markov_branching_degenerate_max` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=20.0 | ctx | strong | -0.031 [-0.041, -0.021]* | -0.060 [-0.075, -0.044]* | +0.002 [+0.000, +0.005] | -0.029 [-0.069, +0.011] | -0.688 [-0.743, -0.643]* | -1.919 [-1.959, -1.881]* | +0.362 [+0.068, +0.671]* |
| max=20.0 | noctx | strong | +0.037 [+0.028, +0.047]* | -0.001 [-0.005, +0.003] | +0.003 [+0.001, +0.006]* | +0.035 [-0.011, +0.077] | -1.253 [-1.296, -1.210]* | -1.973 [-2.007, -1.937]* | +0.706 [+0.426, +0.983]* |

### `markov_branching_candidate_floor` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 (parent on) | ctx | strong | -0.021 [-0.030, -0.012]* | -0.037 [-0.050, -0.025]* | +0.001 [+0.000, +0.003] | -0.027 [-0.056, +0.003] | -0.568 [-0.624, -0.514]* | -1.444 [-1.511, -1.382]* | +0.254 [+0.004, +0.487]* |
| min=1 (parent on) | noctx | strong | +0.029 [+0.020, +0.039]* | -0.002 [-0.005, +0.001] | +0.006 [+0.003, +0.010]* | +0.024 [-0.016, +0.061] | -1.090 [-1.143, -1.043]* | -1.661 [-1.723, -1.599]* | +0.552 [+0.303, +0.805]* |
| max=5 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5 (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `markov_short_half_life_days` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | weak | +0.008 [-0.005, +0.020] | -0.002 [-0.024, +0.018] | +0.001 [-0.001, +0.003] | -0.027 [-0.072, +0.019] | +0.022 [-0.035, +0.083] | -0.007 [-0.037, +0.022] | -0.020 [-0.316, +0.271] |
| min=1.0 (parent on) | noctx | weak | +0.001 [-0.008, +0.012] | +0.000 [-0.003, +0.003] | +0.000 [+0.000, +0.000] | +0.011 [-0.040, +0.056] | +0.001 [-0.049, +0.049] | -0.027 [-0.053, +0.000] | -0.001 [-0.262, +0.280] |
| max=14.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=14.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=14.0 (parent on) | ctx | weak | +0.008 [-0.003, +0.020] | -0.007 [-0.029, +0.013] | +0.001 [-0.001, +0.004] | +0.011 [-0.037, +0.061] | +0.021 [-0.037, +0.081] | +0.005 [-0.022, +0.032] | -0.067 [-0.381, +0.215] |
| max=14.0 (parent on) | noctx | weak | +0.003 [-0.007, +0.012] | +0.000 [-0.003, +0.003] | +0.000 [+0.000, +0.000] | +0.005 [-0.043, +0.053] | -0.026 [-0.076, +0.025] | -0.039 [-0.065, -0.012]* | +0.075 [-0.186, +0.345] |

### `markov_long_compression_beta` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.5 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.5 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.5 (parent on) | ctx | weak | +0.008 [-0.005, +0.019] | +0.004 [-0.019, +0.025] | +0.001 [-0.001, +0.004] | -0.003 [-0.051, +0.048] | +0.012 [-0.050, +0.069] | -0.007 [-0.036, +0.019] | +0.005 [-0.287, +0.287] |
| min=0.5 (parent on) | noctx | weak | +0.002 [-0.007, +0.011] | +0.000 [-0.003, +0.003] | +0.000 [+0.000, +0.000] | -0.008 [-0.056, +0.040] | -0.023 [-0.072, +0.028] | -0.032 [-0.059, -0.006]* | +0.095 [-0.192, +0.371] |
| max=0.75 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=0.75 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=0.75 (parent on) | ctx | weak | +0.013 [+0.001, +0.025]* | -0.004 [-0.025, +0.017] | +0.001 [-0.001, +0.003] | -0.003 [-0.053, +0.051] | +0.001 [-0.065, +0.057] | -0.005 [-0.033, +0.023] | +0.057 [-0.242, +0.341] |
| max=0.75 (parent on) | noctx | weak | +0.003 [-0.007, +0.013] | +0.001 [-0.002, +0.004] | +0.000 [+0.000, +0.000] | +0.008 [-0.040, +0.051] | -0.026 [-0.073, +0.022] | -0.049 [-0.077, -0.020]* | +0.139 [-0.127, +0.411] |

### `markov_alpha_calm` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=1.0 | ctx | weak | +0.007 [-0.007, +0.021] | +0.002 [-0.018, +0.023] | -0.001 [-0.002, +0.000] | +0.027 [-0.027, +0.080] | -0.117 [-0.175, -0.058]* | -0.047 [-0.079, -0.015]* | -0.257 [-0.546, +0.038] |
| max=1.0 | noctx | weak | -0.004 [-0.014, +0.007] | -0.002 [-0.005, +0.000] | +0.000 [+0.000, +0.000] | +0.051 [+0.000, +0.101] | -0.138 [-0.191, -0.080]* | -0.026 [-0.055, +0.005] | -0.221 [-0.513, +0.063] |

### `markov_interp_order2_weight` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=1.0 | ctx | weak | +0.005 [-0.004, +0.015] | +0.003 [-0.013, +0.019] | +0.001 [+0.000, +0.002] | -0.005 [-0.048, +0.032] | +0.007 [-0.042, +0.053] | +0.003 [-0.022, +0.029] | -0.039 [-0.299, +0.216] |
| max=1.0 | noctx | weak | -0.002 [-0.010, +0.007] | -0.001 [-0.003, +0.002] | +0.001 [+0.000, +0.002] | +0.011 [-0.027, +0.048] | +0.095 [+0.055, +0.136]* | +0.013 [-0.009, +0.037] | +0.037 [-0.186, +0.276] |

### `markov_collocation_bonus` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=2.0 | ctx | strong | -0.045 [-0.056, -0.033]* | -0.081 [-0.099, -0.065]* | +0.011 [+0.006, +0.017]* | +0.069 [+0.021, +0.120]* | -1.101 [-1.159, -1.042]* | +0.000 [+0.000, +0.000] | +3.457 [+3.109, +3.803]* |
| max=2.0 | noctx | strong | +0.049 [+0.038, +0.061]* | -0.001 [-0.004, +0.002] | +0.019 [+0.012, +0.025]* | +0.192 [+0.141, +0.237]* | -1.788 [-1.844, -1.729]* | +0.000 [+0.000, +0.000] | +4.181 [+3.765, +4.605]* |

### `markov_collocation_break_penalty` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=2.0 | ctx | strong | -0.028 [-0.037, -0.019]* | -0.043 [-0.056, -0.029]* | +0.001 [+0.000, +0.002] | -0.040 [-0.072, -0.005]* | -0.486 [-0.531, -0.443]* | +0.000 [+0.000, +0.000] | -0.625 [-0.859, -0.383]* |
| max=2.0 | noctx | strong | -0.008 [-0.015, -0.001]* | +0.003 [+0.001, +0.005]* | +0.001 [+0.000, +0.002] | +0.003 [-0.029, +0.035] | -0.830 [-0.871, -0.787]* | +0.000 [+0.000, +0.000] | -0.619 [-0.833, -0.399]* |

### `markov_hot_ngram_meme_ordering` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=True | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.002 [-0.003, +0.009] | -0.001 [-0.003, +0.001] | -0.004 [-0.024, +0.017] |
| flip=True | noctx | weak | +0.001 [-0.009, +0.011] | -0.001 [-0.004, +0.003] | +0.001 [+0.000, +0.003] | +0.005 [-0.043, +0.053] | +0.044 [-0.013, +0.096] | -0.018 [-0.049, +0.015] | -0.041 [-0.323, +0.246] |
| flip=True (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.002 [-0.003, +0.009] | -0.001 [-0.003, +0.001] | -0.004 [-0.024, +0.017] |
| flip=True (parent on) | noctx | weak | +0.001 [-0.009, +0.011] | -0.001 [-0.004, +0.003] | +0.001 [+0.000, +0.003] | +0.005 [-0.043, +0.053] | +0.044 [-0.013, +0.096] | -0.018 [-0.049, +0.015] | -0.041 [-0.323, +0.246] |

### `markov_seeded_candidate_ratio` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=0.7 | ctx | strong | +0.020 [+0.006, +0.034]* | -0.036 [-0.059, -0.013]* | +0.000 [-0.002, +0.002] | +0.011 [-0.043, +0.069] | +0.047 [-0.023, +0.113] | +0.143 [+0.115, +0.173]* | +0.022 [-0.281, +0.335] |
| max=0.7 | noctx | weak | -0.018 [-0.028, -0.007]* | +0.015 [+0.008, +0.022]* | +0.000 [+0.000, +0.000] | +0.035 [-0.016, +0.083] | -0.097 [-0.155, -0.033]* | -0.001 [-0.032, +0.032] | +0.177 [-0.120, +0.465] |

### `hot_ngram_slot_ratio` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.001 [-0.005, +0.005] | -0.002 [-0.007, +0.002] | +0.000 [-0.029, +0.026] |
| min=0.0 | noctx | weak | +0.002 [-0.009, +0.012] | -0.002 [-0.006, +0.001] | +0.000 [+0.000, +0.000] | -0.029 [-0.080, +0.021] | -0.044 [-0.100, +0.015] | -0.016 [-0.053, +0.020] | +0.197 [-0.113, +0.519] |
| max=0.7 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=0.7 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `assoc_slot_ratio` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=0.7 | ctx | strong | -0.012 [-0.024, +0.001] | -0.054 [-0.074, -0.034]* | +0.001 [-0.001, +0.003] | -0.003 [-0.056, +0.053] | +0.078 [+0.017, +0.139]* | +0.117 [+0.091, +0.143]* | -0.073 [-0.380, +0.244] |
| max=0.7 | noctx | strong | -0.046 [-0.056, -0.037]* | +0.002 [-0.001, +0.006] | +0.001 [+0.000, +0.002] | -0.045 [-0.091, +0.000] | -0.066 [-0.119, -0.015]* | -0.023 [-0.051, +0.004] | +0.168 [-0.094, +0.437] |

### `phrase_slot_ratio` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.052 [-0.069, -0.034]* | +0.023 [+0.002, +0.044]* | +0.001 [-0.001, +0.004] | -0.008 [-0.064, +0.043] | +0.006 [-0.062, +0.073] | -0.158 [-0.193, -0.125]* | -0.166 [-0.449, +0.119] |
| min=0.0 | noctx | strong | -0.062 [-0.071, -0.053]* | +0.001 [-0.003, +0.004] | +0.001 [+0.000, +0.003] | -0.045 [-0.096, +0.003] | +0.108 [+0.055, +0.164]* | -0.135 [-0.165, -0.103]* | +0.129 [-0.141, +0.401] |
| max=0.7 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=0.7 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `phrase_min_count` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=10000 | ctx | strong | -0.052 [-0.069, -0.034]* | +0.023 [+0.002, +0.044]* | +0.001 [-0.001, +0.004] | -0.008 [-0.064, +0.043] | +0.006 [-0.062, +0.073] | -0.158 [-0.193, -0.125]* | -0.166 [-0.449, +0.119] |
| max=10000 | noctx | strong | -0.062 [-0.071, -0.053]* | +0.001 [-0.003, +0.004] | +0.001 [+0.000, +0.003] | -0.045 [-0.096, +0.003] | +0.108 [+0.055, +0.164]* | -0.135 [-0.165, -0.103]* | +0.129 [-0.141, +0.401] |
| max=10000 (parent on) | ctx | strong | -0.052 [-0.069, -0.034]* | +0.023 [+0.002, +0.044]* | +0.001 [-0.001, +0.004] | -0.008 [-0.064, +0.043] | +0.006 [-0.062, +0.073] | -0.158 [-0.193, -0.125]* | -0.166 [-0.449, +0.119] |
| max=10000 (parent on) | noctx | strong | -0.062 [-0.071, -0.053]* | +0.001 [-0.003, +0.004] | +0.001 [+0.000, +0.003] | -0.045 [-0.096, +0.003] | +0.108 [+0.055, +0.164]* | -0.135 [-0.165, -0.103]* | +0.129 [-0.141, +0.401] |

### `markov_seed_branch_min` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | strong | +0.055 [+0.040, +0.071]* | -0.009 [-0.033, +0.015] | +0.000 [-0.002, +0.002] | -0.005 [-0.056, +0.051] | -0.039 [-0.118, +0.030] | +0.156 [+0.127, +0.185]* | -0.017 [-0.341, +0.297] |
| min=1.0 (parent on) | noctx | weak | -0.011 [-0.022, +0.001] | +0.026 [+0.018, +0.035]* | +0.000 [+0.000, +0.000] | +0.032 [-0.019, +0.077] | -0.078 [-0.135, -0.016]* | +0.018 [-0.011, +0.053] | +0.201 [-0.108, +0.490] |
| max=1000.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 (parent on) | noctx | strong | -0.062 [-0.071, -0.053]* | +0.001 [-0.003, +0.004] | +0.001 [+0.000, +0.003] | -0.045 [-0.096, +0.003] | +0.108 [+0.055, +0.164]* | -0.135 [-0.165, -0.103]* | +0.129 [-0.141, +0.401] |

### `markov_seed_branch_ideal` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | strong | +0.020 [+0.007, +0.033]* | -0.043 [-0.064, -0.019]* | +0.000 [-0.002, +0.002] | +0.016 [-0.037, +0.075] | +0.051 [-0.019, +0.115] | +0.145 [+0.117, +0.174]* | +0.047 [-0.253, +0.345] |
| min=1.0 (parent on) | noctx | weak | -0.018 [-0.028, -0.006]* | +0.015 [+0.008, +0.023]* | +0.000 [+0.000, +0.000] | +0.016 [-0.037, +0.064] | -0.101 [-0.157, -0.037]* | -0.012 [-0.043, +0.021] | +0.141 [-0.154, +0.427] |
| max=1000.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 (parent on) | ctx | strong | +0.020 [+0.007, +0.035]* | -0.036 [-0.058, -0.013]* | +0.001 [-0.001, +0.004] | +0.000 [-0.053, +0.056] | +0.054 [-0.015, +0.123] | +0.149 [+0.120, +0.179]* | +0.099 [-0.235, +0.405] |
| max=1000.0 (parent on) | noctx | weak | -0.016 [-0.027, -0.006]* | +0.016 [+0.009, +0.023]* | +0.000 [+0.000, +0.000] | -0.003 [-0.053, +0.043] | -0.077 [-0.135, -0.011]* | +0.006 [-0.023, +0.038] | +0.229 [-0.064, +0.509] |

### `markov_seed_branch_max` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | noctx | strong | -0.062 [-0.071, -0.053]* | +0.001 [-0.003, +0.004] | +0.001 [+0.000, +0.003] | -0.045 [-0.096, +0.003] | +0.108 [+0.055, +0.164]* | -0.135 [-0.165, -0.103]* | +0.129 [-0.141, +0.401] |
| max=5000.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5000.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5000.0 (parent on) | ctx | strong | +0.026 [+0.010, +0.041]* | -0.052 [-0.074, -0.029]* | +0.001 [-0.001, +0.003] | -0.011 [-0.067, +0.048] | +0.097 [+0.024, +0.170]* | +0.161 [+0.131, +0.189]* | +0.178 [-0.170, +0.483] |
| max=5000.0 (parent on) | noctx | weak | -0.015 [-0.026, -0.004]* | +0.011 [+0.005, +0.018]* | +0.000 [+0.000, +0.000] | +0.008 [-0.043, +0.056] | -0.105 [-0.163, -0.039]* | +0.012 [-0.017, +0.045] | +0.165 [-0.135, +0.479] |

### `markov_seed_min_support` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | weak | +0.022 [+0.008, +0.036]* | +0.007 [-0.014, +0.029] | +0.000 [-0.002, +0.002] | +0.051 [-0.005, +0.109] | -0.031 [-0.099, +0.035] | +0.136 [+0.106, +0.166]* | +0.111 [-0.185, +0.423] |
| min=1.0 (parent on) | noctx | weak | -0.019 [-0.031, -0.009]* | +0.024 [+0.016, +0.033]* | +0.000 [+0.000, +0.000] | +0.000 [-0.045, +0.045] | -0.117 [-0.176, -0.052]* | -0.004 [-0.035, +0.029] | +0.175 [-0.113, +0.473] |
| max=500.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=500.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=500.0 (parent on) | ctx | inert | -0.001 [-0.005, +0.002] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.001 [-0.007, +0.009] | +0.004 [+0.000, +0.009] | -0.015 [-0.053, +0.017] |
| max=500.0 (parent on) | noctx | strong | -0.060 [-0.070, -0.050]* | +0.001 [-0.003, +0.004] | +0.001 [+0.000, +0.003] | -0.045 [-0.096, +0.003] | +0.102 [+0.047, +0.157]* | -0.130 [-0.160, -0.098]* | +0.098 [-0.171, +0.367] |

### `markov_seed_min_score` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 (parent on) | ctx | strong | +0.059 [+0.042, +0.075]* | -0.037 [-0.060, -0.014]* | +0.001 [-0.001, +0.003] | +0.003 [-0.051, +0.064] | +0.016 [-0.061, +0.086] | +0.160 [+0.129, +0.189]* | +0.037 [-0.309, +0.380] |
| min=0.0 (parent on) | noctx | weak | -0.008 [-0.018, +0.005] | +0.021 [+0.013, +0.029]* | +0.000 [+0.000, +0.000] | +0.037 [-0.013, +0.085] | -0.095 [-0.157, -0.029]* | +0.019 [-0.013, +0.055] | +0.118 [-0.187, +0.426] |
| max=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 (parent on) | ctx | weak | +0.011 [+0.005, +0.017]* | +0.005 [-0.007, +0.018] | +0.000 [+0.000, +0.000] | +0.019 [-0.011, +0.048] | -0.027 [-0.059, +0.004] | +0.018 [+0.004, +0.033]* | +0.019 [-0.125, +0.169] |
| max=1.0 (parent on) | noctx | strong | -0.056 [-0.066, -0.047]* | +0.007 [+0.003, +0.013]* | +0.001 [+0.000, +0.002] | -0.037 [-0.088, +0.011] | +0.065 [+0.010, +0.124]* | -0.111 [-0.141, -0.076]* | +0.261 [-0.019, +0.532] |

### `markov_seed_min_token_len` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1 | ctx | weak | -0.003 [-0.010, +0.005] | -0.017 [-0.031, -0.003]* | +0.001 [+0.000, +0.002] | -0.011 [-0.040, +0.016] | +0.020 [-0.024, +0.061] | +0.000 [-0.019, +0.019] | -0.022 [-0.211, +0.169] |
| min=1 | noctx | weak | +0.000 [-0.005, +0.006] | +0.000 [-0.002, +0.002] | +0.000 [+0.000, +0.000] | +0.016 [-0.008, +0.040] | +0.001 [-0.033, +0.038] | +0.007 [-0.010, +0.023] | +0.035 [-0.145, +0.211] |
| min=1 (parent on) | ctx | strong | +0.037 [+0.022, +0.053]* | -0.041 [-0.063, -0.019]* | +0.001 [-0.001, +0.003] | -0.003 [-0.056, +0.048] | +0.034 [-0.039, +0.106] | +0.169 [+0.139, +0.199]* | -0.017 [-0.347, +0.290] |
| min=1 (parent on) | noctx | weak | -0.016 [-0.027, -0.004]* | +0.016 [+0.009, +0.023]* | +0.000 [+0.000, +0.000] | +0.037 [-0.016, +0.085] | -0.122 [-0.182, -0.055]* | +0.008 [-0.023, +0.042] | +0.206 [-0.099, +0.505] |
| max=20 | ctx | strong | -0.052 [-0.069, -0.034]* | +0.023 [+0.002, +0.044]* | +0.001 [-0.001, +0.004] | -0.008 [-0.064, +0.043] | +0.006 [-0.062, +0.073] | -0.158 [-0.193, -0.125]* | -0.166 [-0.449, +0.119] |
| max=20 | noctx | strong | -0.062 [-0.071, -0.053]* | +0.001 [-0.003, +0.004] | +0.001 [+0.000, +0.003] | -0.045 [-0.096, +0.003] | +0.108 [+0.055, +0.164]* | -0.135 [-0.165, -0.103]* | +0.129 [-0.141, +0.401] |
| max=20 (parent on) | ctx | strong | -0.052 [-0.069, -0.034]* | +0.023 [+0.002, +0.044]* | +0.001 [-0.001, +0.004] | -0.008 [-0.064, +0.043] | +0.006 [-0.062, +0.073] | -0.158 [-0.193, -0.125]* | -0.166 [-0.449, +0.119] |
| max=20 (parent on) | noctx | strong | -0.062 [-0.071, -0.053]* | +0.001 [-0.003, +0.004] | +0.001 [+0.000, +0.003] | -0.045 [-0.096, +0.003] | +0.108 [+0.055, +0.164]* | -0.135 [-0.165, -0.103]* | +0.129 [-0.141, +0.401] |

### `markov_seed_head_share` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | -0.006 [-0.019, +0.007] | +0.003 [-0.019, +0.025] | +0.001 [+0.000, +0.002] | +0.008 [-0.040, +0.059] | -0.111 [-0.171, -0.055]* | -0.025 [-0.057, +0.007] | -0.588 [-0.887, -0.306]* |
| min=0.0 | noctx | strong | -0.014 [-0.023, -0.004]* | +0.002 [-0.002, +0.006] | +0.000 [+0.000, +0.000] | -0.008 [-0.056, +0.037] | -0.191 [-0.243, -0.136]* | -0.023 [-0.049, +0.004] | -0.338 [-0.621, -0.063]* |
| min=0.0 (parent on) | ctx | strong | +0.011 [-0.003, +0.027] | -0.047 [-0.071, -0.025]* | +0.001 [-0.001, +0.003] | -0.051 [-0.104, +0.003] | -0.115 [-0.178, -0.051]* | +0.121 [+0.090, +0.151]* | -1.547 [-1.893, -1.233]* |
| min=0.0 (parent on) | noctx | strong | -0.020 [-0.031, -0.010]* | +0.011 [+0.005, +0.018]* | +0.001 [+0.000, +0.003] | -0.032 [-0.083, +0.016] | -0.157 [-0.213, -0.097]* | +0.019 [-0.013, +0.051] | -0.247 [-0.523, +0.052] |
| max=1.0 | ctx | weak | -0.009 [-0.023, +0.006] | -0.014 [-0.035, +0.009] | +0.003 [+0.000, +0.007] | -0.029 [-0.083, +0.021] | -0.015 [-0.080, +0.045] | -0.003 [-0.035, +0.029] | -0.535 [-0.829, -0.232]* |
| max=1.0 | noctx | weak | -0.016 [-0.026, -0.006]* | +0.000 [-0.003, +0.003] | +0.001 [+0.000, +0.002] | -0.024 [-0.072, +0.021] | +0.004 [-0.048, +0.059] | -0.007 [-0.035, +0.021] | -0.219 [-0.483, +0.053] |
| max=1.0 (parent on) | ctx | strong | +0.008 [-0.006, +0.023] | -0.054 [-0.075, -0.031]* | +0.001 [-0.001, +0.003] | -0.029 [-0.085, +0.024] | -0.023 [-0.094, +0.045] | +0.136 [+0.106, +0.169]* | -1.041 [-1.355, -0.709]* |
| max=1.0 (parent on) | noctx | strong | -0.020 [-0.031, -0.009]* | +0.010 [+0.004, +0.017]* | +0.001 [+0.000, +0.003] | -0.021 [-0.072, +0.032] | -0.162 [-0.223, -0.101]* | -0.013 [-0.042, +0.020] | -0.175 [-0.459, +0.115] |

### `enable_backoff` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=False | ctx | weak | -0.004 [-0.018, +0.011] | +0.022 [+0.000, +0.045] | +0.001 [+0.000, +0.002] | -0.027 [-0.077, +0.024] | -0.096 [-0.165, -0.031]* | +0.013 [-0.025, +0.047] | -0.097 [-0.401, +0.215] |
| flip=False | noctx | strong | +0.008 [-0.004, +0.018] | +0.001 [-0.004, +0.005] | +0.000 [+0.000, +0.000] | +0.011 [-0.040, +0.059] | -0.178 [-0.238, -0.127]* | -0.005 [-0.038, +0.029] | +0.611 [+0.346, +0.879]* |

### `markov_jump_probability` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | +0.001 [-0.009, +0.010] | -0.037 [-0.056, -0.019]* | +0.001 [+0.000, +0.003] | +0.021 [-0.021, +0.067] | -0.047 [-0.094, +0.000] | -0.021 [-0.047, +0.005] | -0.049 [-0.320, +0.219] |
| min=0.0 | noctx | weak | +0.009 [-0.000, +0.018] | -0.001 [-0.005, +0.003] | +0.001 [+0.000, +0.003] | +0.000 [-0.040, +0.037] | -0.107 [-0.149, -0.064]* | +0.011 [-0.016, +0.035] | +0.105 [-0.152, +0.364] |
| max=1.0 | ctx | strong | +0.004 [-0.008, +0.017] | +0.046 [+0.026, +0.068]* | +0.001 [+0.000, +0.002] | -0.013 [-0.067, +0.032] | +0.043 [-0.020, +0.100] | +0.015 [-0.018, +0.047] | -0.241 [-0.525, +0.034] |
| max=1.0 | noctx | strong | -0.007 [-0.018, +0.003] | -0.001 [-0.004, +0.003] | +0.000 [+0.000, +0.000] | -0.027 [-0.077, +0.016] | +0.219 [+0.163, +0.266]* | +0.012 [-0.019, +0.044] | +0.172 [-0.119, +0.461] |

### `context_jump_boost` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | weak | -0.003 [-0.005, +0.000] | +0.001 [-0.010, +0.014] | +0.000 [+0.000, +0.000] | +0.005 [-0.016, +0.027] | +0.009 [-0.019, +0.033] | -0.001 [-0.014, +0.012] | -0.011 [-0.161, +0.142] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=10.0 | ctx | strong | +0.000 [-0.005, +0.004] | +0.037 [+0.021, +0.055]* | +0.000 [+0.000, +0.000] | -0.019 [-0.056, +0.016] | -0.026 [-0.063, +0.014] | +0.012 [-0.005, +0.031] | -0.123 [-0.308, +0.073] |
| max=10.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `verbatim_extension_share` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.007 [-0.014, +0.002] | +0.005 [-0.007, +0.017] | +0.001 [+0.000, +0.003] | +0.005 [-0.029, +0.040] | -0.261 [-0.303, -0.218]* | +0.012 [-0.011, +0.035] | -0.320 [-0.518, -0.111]* |
| min=0.0 | noctx | strong | +0.012 [+0.003, +0.020]* | +0.000 [-0.004, +0.003] | +0.000 [+0.000, +0.000] | +0.053 [+0.013, +0.096]* | -0.545 [-0.592, -0.497]* | +0.018 [-0.006, +0.044] | -0.698 [-0.947, -0.449]* |
| max=1.0 | ctx | weak | -0.003 [-0.008, +0.001] | +0.001 [-0.005, +0.007] | +0.000 [+0.000, +0.000] | -0.003 [-0.021, +0.013] | -0.033 [-0.050, -0.015]* | -0.001 [-0.012, +0.009] | -0.098 [-0.197, +0.002] |
| max=1.0 | noctx | weak | -0.006 [-0.011, -0.001]* | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | +0.008 [-0.013, +0.029] | -0.031 [-0.051, -0.011]* | -0.001 [-0.013, +0.010] | -0.150 [-0.271, -0.035]* |

### `order_mix_probability` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.001 [-0.015, +0.014] | +0.035 [+0.015, +0.057]* | +0.003 [+0.001, +0.005]* | -0.013 [-0.061, +0.037] | -0.131 [-0.195, -0.069]* | +0.005 [-0.031, +0.039] | +0.042 [-0.266, +0.330] |
| min=0.0 | noctx | strong | +0.012 [+0.001, +0.023]* | +0.001 [-0.003, +0.006] | +0.001 [+0.000, +0.002] | -0.003 [-0.048, +0.037] | -0.193 [-0.248, -0.137]* | -0.008 [-0.043, +0.027] | +0.663 [+0.384, +0.963]* |
| max=1.0 | ctx | weak | +0.005 [-0.004, +0.015] | +0.003 [-0.013, +0.019] | +0.001 [+0.000, +0.002] | -0.005 [-0.048, +0.032] | +0.006 [-0.043, +0.052] | +0.003 [-0.022, +0.029] | -0.039 [-0.299, +0.216] |
| max=1.0 | noctx | weak | -0.002 [-0.010, +0.007] | -0.001 [-0.003, +0.002] | +0.001 [+0.000, +0.002] | +0.011 [-0.027, +0.048] | +0.094 [+0.055, +0.136]* | +0.013 [-0.010, +0.036] | +0.037 [-0.186, +0.276] |

### `slot_mutation_probability` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | +0.005 [-0.008, +0.018] | +0.003 [-0.017, +0.023] | +0.001 [+0.000, +0.003] | +0.008 [-0.040, +0.053] | +0.101 [+0.042, +0.164]* | +0.287 [+0.258, +0.315]* | -0.160 [-0.456, +0.132] |
| min=0.0 | noctx | strong | -0.006 [-0.016, +0.004] | +0.000 [-0.003, +0.003] | +0.001 [+0.000, +0.002] | -0.003 [-0.045, +0.043] | +0.196 [+0.149, +0.241]* | +0.307 [+0.283, +0.332]* | +0.127 [-0.147, +0.427] |
| max=1.0 | ctx | strong | -0.010 [-0.020, +0.002] | -0.012 [-0.029, +0.005] | +0.000 [+0.000, +0.000] | +0.003 [-0.037, +0.043] | -0.317 [-0.377, -0.264]* | -0.897 [-0.933, -0.867]* | +0.147 [-0.157, +0.433] |
| max=1.0 | noctx | strong | -0.002 [-0.012, +0.007] | -0.001 [-0.005, +0.003] | +0.000 [+0.000, +0.000] | +0.021 [-0.024, +0.067] | -0.521 [-0.565, -0.477]* | -0.915 [-0.946, -0.883]* | +0.273 [+0.028, +0.529]* |

### `hot_ngram_min_count` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000 | ctx | inert | -0.000 [-0.003, +0.002] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [-0.008, +0.008] | +0.005 [-0.001, +0.013] | -0.003 [-0.008, +0.002] | +0.006 [-0.026, +0.035] |
| max=1000 | noctx | weak | +0.005 [-0.006, +0.015] | -0.002 [-0.006, +0.001] | +0.000 [+0.000, +0.000] | -0.035 [-0.088, +0.013] | -0.064 [-0.123, -0.007]* | -0.016 [-0.053, +0.019] | +0.106 [-0.198, +0.416] |
| max=1000 (parent on) | ctx | inert | -0.000 [-0.003, +0.002] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [-0.008, +0.008] | +0.005 [-0.001, +0.013] | -0.003 [-0.008, +0.002] | +0.006 [-0.026, +0.035] |
| max=1000 (parent on) | noctx | weak | +0.005 [-0.006, +0.015] | -0.002 [-0.006, +0.001] | +0.000 [+0.000, +0.000] | -0.035 [-0.088, +0.013] | -0.064 [-0.123, -0.007]* | -0.016 [-0.053, +0.019] | +0.106 [-0.198, +0.416] |

### `hot_ngram_recency_share` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | -0.000 [-0.001, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.003 [+0.000, +0.008] | +0.006 [+0.000, +0.013] | -0.004 [-0.007, -0.001]* | -0.024 [-0.047, -0.003]* |
| min=0.0 | noctx | strong | +0.005 [-0.005, +0.016] | +0.004 [-0.001, +0.009] | +0.001 [+0.000, +0.003] | +0.131 [+0.077, +0.179]* | +0.010 [-0.044, +0.059] | -0.035 [-0.065, -0.007]* | +0.344 [+0.099, +0.611]* |
| min=0.0 (parent on) | ctx | inert | -0.000 [-0.001, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.003 [+0.000, +0.008] | +0.006 [+0.000, +0.013] | -0.004 [-0.007, -0.001]* | -0.024 [-0.047, -0.003]* |
| min=0.0 (parent on) | noctx | strong | +0.005 [-0.005, +0.016] | +0.004 [-0.001, +0.009] | +0.001 [+0.000, +0.003] | +0.131 [+0.077, +0.179]* | +0.010 [-0.044, +0.059] | -0.035 [-0.065, -0.007]* | +0.344 [+0.099, +0.611]* |
| max=1.0 | ctx | inert | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [-0.008, +0.008] | +0.003 [-0.003, +0.011] | -0.003 [-0.007, +0.001] | +0.014 [-0.010, +0.045] |
| max=1.0 | noctx | weak | -0.000 [-0.010, +0.009] | +0.000 [-0.003, +0.003] | +0.001 [+0.000, +0.002] | -0.027 [-0.077, +0.021] | -0.024 [-0.077, +0.028] | +0.018 [-0.013, +0.050] | +0.013 [-0.277, +0.315] |
| max=1.0 (parent on) | ctx | inert | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [-0.008, +0.008] | +0.003 [-0.003, +0.011] | -0.003 [-0.007, +0.001] | +0.014 [-0.010, +0.045] |
| max=1.0 (parent on) | noctx | weak | -0.000 [-0.010, +0.009] | +0.000 [-0.003, +0.003] | +0.001 [+0.000, +0.002] | -0.027 [-0.077, +0.021] | -0.024 [-0.077, +0.028] | +0.018 [-0.013, +0.050] | +0.013 [-0.277, +0.315] |

### `fuzzy_context_casefold` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=False | ctx | inert | -0.001 [-0.003, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.005 [-0.011, +0.003] | +0.000 [-0.005, +0.005] | +0.016 [-0.017, +0.057] |
| flip=False | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `reply_context_bias` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | weak | -0.002 [-0.007, +0.003] | -0.017 [-0.029, -0.005]* | +0.001 [+0.000, +0.002] | -0.016 [-0.048, +0.016] | +0.033 [-0.001, +0.063] | -0.003 [-0.021, +0.015] | -0.212 [-0.386, -0.031]* |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | ctx | strong | -0.003 [-0.009, +0.003] | +0.039 [+0.025, +0.055]* | +0.001 [+0.000, +0.002] | +0.019 [-0.016, +0.053] | -0.043 [-0.082, -0.006]* | +0.011 [-0.005, +0.027] | +0.044 [-0.135, +0.229] |
| max=4.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `reply_context_start_bias` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | strong | -0.069 [-0.082, -0.058]* | -0.152 [-0.169, -0.134]* | +0.000 [+0.000, +0.000] | -0.088 [-0.133, -0.045]* | +0.324 [+0.263, +0.380]* | +0.037 [+0.007, +0.067]* | -0.281 [-0.559, +0.023] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | ctx | strong | +0.030 [+0.020, +0.041]* | +0.063 [+0.045, +0.079]* | +0.000 [+0.000, +0.000] | +0.040 [+0.000, +0.083] | -0.085 [-0.132, -0.037]* | -0.008 [-0.028, +0.015] | +0.016 [-0.204, +0.225] |
| max=4.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `generation_attempts_with_context` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0 | ctx | strong | -0.083 [-0.095, -0.071]* | -0.153 [-0.171, -0.136]* | +0.001 [+0.000, +0.002] | -0.112 [-0.163, -0.064]* | +0.333 [+0.265, +0.398]* | +0.028 [-0.005, +0.059] | -0.194 [-0.489, +0.125] |
| min=0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=10 | ctx | weak | +0.013 [+0.008, +0.017]* | +0.009 [+0.003, +0.015]* | +0.000 [+0.000, +0.000] | -0.008 [-0.024, +0.008] | -0.043 [-0.065, -0.022]* | -0.012 [-0.021, -0.003]* | +0.047 [-0.048, +0.135] |
| max=10 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `context_start_affinity` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | weak | -0.003 [-0.013, +0.008] | -0.007 [-0.023, +0.009] | +0.001 [+0.000, +0.002] | -0.011 [-0.053, +0.032] | -0.045 [-0.102, +0.005] | -0.013 [-0.038, +0.011] | -0.075 [-0.333, +0.161] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=10.0 | ctx | weak | +0.007 [-0.002, +0.018] | +0.000 [-0.016, +0.017] | +0.003 [+0.001, +0.005]* | -0.008 [-0.051, +0.037] | +0.019 [-0.037, +0.074] | -0.005 [-0.028, +0.021] | -0.014 [-0.264, +0.243] |
| max=10.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `context_anchor_splice_probability` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.047 [-0.059, -0.035]* | -0.015 [-0.036, +0.007] | +0.002 [+0.000, +0.005] | +0.024 [-0.027, +0.075] | +0.101 [+0.041, +0.155]* | +0.001 [-0.025, +0.029] | +0.006 [-0.271, +0.271] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | ctx | strong | +0.062 [+0.052, +0.075]* | +0.025 [+0.007, +0.043]* | +0.000 [+0.000, +0.000] | +0.011 [-0.029, +0.053] | -0.037 [-0.092, +0.018] | +0.013 [-0.009, +0.039] | +0.328 [+0.065, +0.581]* |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

---

## Вердикт (владельцу; дописан вручную 2026-09-11)

Вторая перепись, на новом C0 (после промоушена `promote-hot-and-phrase-routes`:
hot-маршрут, фразовый маршрут, бонус различности, исключение M3R-120 в
дефолтах). Копия прода 10.09, 135 армов × 2 режима, 10 воркеров, 34 минуты.
Итого **strong 30 / weak 12 / gated 11 / inert 7** (02.09: 23 / 12 / 13 / 8);
43 ручки читаются вне ядра генерации и оффлайн не меряются, 6 не свипуются
по форме.

### Что изменилось против 02.09 (14 ручек)

| ручка | 02.09 → 11.09 | почему |
|---|---|---|
| `phrase_slot_ratio`, `phrase_min_count`, `selection_diversity_bonus_noctx`, `assoc_slot_ratio` | — → strong | новые или впервые в переписи; `phrase_slot_ratio` min=0 в ctx: affinity −0.052\*, copy +0.023\*, ECB −0.158\* — снятие маршрута теперь стоит тематичности |
| `markov_seed_head_share`, `markov_seed_min_token_len` | gated → strong | их читает и фразовый маршрут (общий сборщик), родитель больше не нужен: `min_token_len` 20 в ctx — affinity −0.052\*, ECB −0.158\* |
| `hot_ngram_slot_ratio` | inert → weak | при порогах 2 / 0.25 горячий пул не пуст; min=0 двигает только noctx |
| `fuzzy_context_casefold` | weak → inert | все интервалы в допуске в обоих режимах |
| `markov_alpha_calm`, `markov_interp_order2_weight` | strong → weak | на новом C0 экстремумы двигают только escape (−0.12 / +0.10) |
| `markov_hot_ngram_meme_ordering` | inert → weak | родитель включён по умолчанию, флип виден в noctx |
| `context_jump_boost`, `randomness_strength`, `reply_context_bias` | weak → strong | экстремумы пересекли планку на новом C0 (copy +0.037\*, meme +0.053\*, copy +0.039\*) |

### Границы доменов, которые перепись показала мёртвыми

- **`phrase_slot_ratio` / `hot_ngram_slot_ratio` / `assoc_slot_ratio` выше 0.5 —
  инертны по построению:** `route_slot_budget` держит потолок в половину пула
  (2 слота из 5), поэтому max=0.7 ≡ 0.4. Домен 0..0.7 обещает то, чего нет.
- **`selection_diversity_bonus` выше запаса окна сужает окно:** max=1.0 в ctx —
  escape −0.321\*, affinity −0.063\*, copy −0.094\*; при 0.3 запаса потолок
  домена 1.0 бессмыслен (вердикт 02.09, d40, подтверждён).
- **`hot_ngram_min_count` до 1000** — при таком пороге горячий пул пуст, ручка
  превращается в выключатель маршрута через данные (esc −0.064\* в noctx).

### Скрытая связь, найденная по пути

`hot_ngram_seed_chance` — не только шанс легаси-затравки (при включённом
маршруте она не подаётся, `l1-hot-route` D4), но и **гейт записи** горячего
окна на пути обучения (`reply_pipeline.py`, «Gated on the channel knob so a
zero chance keeps the learn path write-free»). Обнулить её «как мёртвую» —
значит перестать писать окно и выключить hot-маршрут данными. Чистить —
только с переносом гейта записи на `hot_ngram_slot_ratio > 0 or seed_chance > 0`
либо на явный выключатель канала.

### Кандидаты для решения владельца (O22)

Каждый пункт — отдельное решение; при сомнении оставляем. Все правки
хеш-нейтральны, если значение сохраняется константой.

| группа | ручки | предложение | цена |
|---|---|---|---|
| inert, мёртвые по коду | `recent_reply_penalty_strength` (+ `build_recent_reply_trigrams`, `recent_reply_overlap`), `fuzzy_context_casefold` (+ casefold-тир каскада, живёт только при `normalize_lower=false`), `markov_shadow_order4_enabled` (фаза 7 закрыта: 0 выборов на 5937 шагов), `auto_capitalize_replies` | удалить с кодом или заморозить константой | ~40 / ~90 / ~30 / 5 строк |
| inert по конвенции харнесса, живые в проде | `reply_flavor_strength`, `emoji_append_chance` | **не трогать**: eval глушит их намеренно, класс ничего не говорит о проде | 0 |
| kill-switch | `markov_cache_incremental` | оставить, ярус internal | 0 |
| gated дети закрытых фаз | фаза 2: `markov_entropy_pivot/temp_min/temp_max` (+ weak-родитель `markov_entropy_temp_gain`), `markov_branching_candidate_floor`; фаза 3: `markov_long_compression_beta`, `markov_short_half_life_days` (M2R-215 ждёт данных — оставить); фаза 5: `markov_seed_branch_min/ideal/max`, `markov_seed_min_support`, `markov_seed_min_score` (M2R-430 ждёт `n_docs` ≥ 500 — оставить до вердикта) | фаза 2 — заморозить константами (гейт провален всеми плечами); фазы 3 и 5 — оставить до их вердиктов | 5 ручек сейчас |
| домены | три `*_slot_ratio` → 0..0.5; `selection_diversity_bonus*` → потолок 0.3 (= запас) либо кросс-полевой инвариант «бонус ≤ запас»; `hot_ngram_min_count` → потолок 100 | сузить домены, тест реестра | малые |
| связь | `hot_ngram_seed_chance` | заменить на выключатель канала или перенести гейт записи | средняя |
| константа | `JUMP_MIN_TOKENS_BETWEEN` (мёртвая при `JUMP_MAX_PER_REPLY=1`) | удалить | 6 строк |
| weak (12) | `candidate_selection_temperature`, `context_start_affinity`, `hot_ngram_min_count`, `intonation_profile_strength`, `length_context_adaptation`, `markov_alpha_calm`, `markov_interp_order2_weight`, `repetition_penalty_strength`, `verbatim_recognized_unit`, … | оставить; ярус tuning при введении ярусов | 0 |

Пороги переписи не двигались. Латентность в таблицах справочная (параллельный
прогон). Сырые семплы — JSON рядом (`knob-census-2026-09-11.json`), числа
только.
