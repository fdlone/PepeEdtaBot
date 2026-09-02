# Перепись ручек (M3R-151) — 2026-09-02

Change `knob-census`. Правило классов — `eval_thresholds.yaml` → `knob_census`, зарегистрировано до прогона. Каждая ручка — на экстремумах домена (булева — инверсия) против C0, парные дельты, оба режима; ручки с родителем — ещё и при включённом родителе. Метрики классификации: `context_affinity_without_copy`, `exact_context_copy_rate`, `repetition_rate`, `historical_meme_rate`, `structural_window_escape`, `structural_pool_ecb`, `mean_response_length`.

- C0 `ctx`: 1500 записей, версия промптов `bababb4b7693`, p95 71.4 мс
- C0 `noctx`: 1500 записей, версия промптов `bababb4b7693`, p95 47.8 мс

Латентность в таблице справочная: армы считались параллельно и между собой по ней не сравнимы. Классы: dead — не читается; gated — двигает только при включённом родителе; inert — интервалы всех дельт внутри полосы допуска на всех экстремумах; strong — значимая дельта не ниже планки силы; weak — остальное.

## Сводка по ручкам

| ручка | класс | предложение |
|---|---|---|
| `max_reply_chars` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `max_reply_tokens` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `auto_capitalize_replies` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `randomness_strength` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `candidate_selection_temperature` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `selection_score_margin` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `context_relevance_weight` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `context_relevance_cap` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `selection_diversity_bonus` | **strong** | keep; check the domain ceiling (an extreme may break form) |
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
| `markov_alpha_calm` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_interp_order2_weight` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_collocation_bonus` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_collocation_break_penalty` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_hot_ngram_meme_ordering` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `markov_seeded_candidate_ratio` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `hot_ngram_slot_ratio` | **inert** | remove or reduce to a constant: extremes move nothing measurable |
| `markov_seed_branch_min` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_branch_ideal` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_branch_max` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_min_support` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_min_score` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_min_token_len` | **gated** | decide together with the parent knob; alone it is a no-op |
| `markov_seed_head_share` | **gated** | decide together with the parent knob; alone it is a no-op |
| `enable_backoff` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `markov_jump_probability` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `context_jump_boost` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `verbatim_extension_share` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `order_mix_probability` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `slot_mutation_probability` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `hot_ngram_min_count` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `hot_ngram_recency_share` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `fuzzy_context_casefold` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `reply_context_bias` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `reply_context_start_bias` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `generation_attempts_with_context` | **strong** | keep; check the domain ceiling (an extreme may break form) |
| `context_start_affinity` | **weak** | candidate to merge or narrow: effect below the strength bar |
| `context_anchor_splice_probability` | **strong** | keep; check the domain ceiling (an extreme may break form) |

Итого: gated 13, inert 8, strong 23, weak 12.

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

## Разбор по ручкам

### `max_reply_chars` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=20 | ctx | strong | -0.143 [-0.160, -0.127]* | -0.202 [-0.224, -0.181]* | -0.002 [-0.005, +0.000] | -0.200 [-0.245, -0.157]* | +0.129 [+0.051, +0.205]* | +0.381 [+0.348, +0.414]* | -7.330 [-7.580, -7.059]* |
| min=20 | noctx | strong | -0.014 [-0.018, -0.009]* | +0.027 [+0.019, +0.036]* | +0.000 [+0.000, +0.000] | -0.024 [-0.053, +0.005] | -0.569 [-0.643, -0.495]* | +0.356 [+0.325, +0.388]* | -6.890 [-7.130, -6.655]* |
| max=4000 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4000 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `max_reply_tokens` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1 | ctx | strong | -0.247 [-0.263, -0.228]* | -0.221 [-0.244, -0.198]* | -0.002 [-0.005, +0.000] | -0.271 [-0.319, -0.225]* | -1.089 [-1.138, -1.036]* | -3.536 [-3.569, -3.502]* | -9.830 [-10.107, -9.567]* |
| min=1 | noctx | strong | -0.024 [-0.029, -0.020]* | +0.005 [+0.000, +0.010] | +0.000 [+0.000, +0.000] | -0.052 [-0.077, -0.030]* | -1.889 [-1.939, -1.841]* | -3.549 [-3.578, -3.519]* | -9.479 [-9.706, -9.235]* |
| max=300 | ctx | weak | +0.001 [-0.002, +0.005] | +0.000 [-0.007, +0.006] | +0.000 [+0.000, +0.000] | -0.003 [-0.016, +0.008] | +0.011 [-0.005, +0.030] | -0.008 [-0.019, +0.003] | +0.111 [-0.012, +0.238] |
| max=300 | noctx | weak | +0.001 [-0.000, +0.002] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.005 [+0.000, +0.013] | +0.001 [-0.006, +0.009] | -0.003 [-0.007, +0.000] | +0.001 [-0.049, +0.049] |

### `auto_capitalize_replies` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=True | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flip=True | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `randomness_strength` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | +0.004 [-0.015, +0.023] | -0.007 [-0.029, +0.017] | +0.000 [-0.003, +0.003] | +0.000 [-0.053, +0.053] | +0.063 [-0.011, +0.131] | +0.001 [-0.045, +0.044] | +0.239 [-0.055, +0.583] |
| min=0.0 | noctx | weak | -0.000 [-0.006, +0.005] | +0.002 [-0.002, +0.006] | +0.000 [+0.000, +0.000] | +0.027 [-0.011, +0.061] | -0.003 [-0.069, +0.063] | +0.047 [+0.009, +0.089]* | -0.073 [-0.355, +0.227] |
| max=3.0 | ctx | weak | -0.012 [-0.030, +0.007] | -0.016 [-0.038, +0.005] | -0.002 [-0.005, +0.000] | -0.024 [-0.069, +0.024] | +0.071 [+0.003, +0.145]* | +0.038 [-0.003, +0.079] | -0.040 [-0.353, +0.287] |
| max=3.0 | noctx | weak | -0.000 [-0.006, +0.006] | +0.005 [+0.000, +0.010] | +0.000 [+0.000, +0.000] | +0.029 [-0.008, +0.064] | +0.039 [-0.031, +0.109] | +0.033 [-0.007, +0.071] | -0.069 [-0.337, +0.226] |

### `candidate_selection_temperature` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | +0.011 [+0.007, +0.015]* | +0.015 [+0.005, +0.025]* | +0.000 [+0.000, +0.000] | -0.005 [-0.029, +0.021] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.073 [-0.250, +0.127] |
| min=0.0 | noctx | weak | -0.001 [-0.005, +0.004] | +0.001 [-0.003, +0.005] | +0.000 [+0.000, +0.000] | +0.013 [-0.013, +0.037] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.411 [-0.611, -0.186]* |
| max=3.0 | ctx | weak | -0.000 [-0.002, +0.001] | -0.007 [-0.011, -0.003]* | +0.000 [+0.000, +0.000] | -0.005 [-0.016, +0.005] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.031 [-0.096, +0.031] |
| max=3.0 | noctx | weak | +0.000 [-0.001, +0.001] | +0.001 [+0.000, +0.002] | +0.000 [+0.000, +0.000] | +0.005 [+0.000, +0.013] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.033 [-0.035, +0.111] |

### `selection_score_margin` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | +0.010 [+0.007, +0.013]* | +0.014 [+0.005, +0.024]* | +0.000 [+0.000, +0.000] | -0.008 [-0.032, +0.016] | -0.891 [-0.939, -0.843]* | +0.000 [+0.000, +0.000] | -0.075 [-0.243, +0.111] |
| min=0.0 | noctx | strong | -0.000 [-0.005, +0.004] | -0.001 [-0.004, +0.002] | +0.000 [+0.000, +0.000] | +0.016 [-0.005, +0.040] | -1.431 [-1.482, -1.382]* | +0.000 [+0.000, +0.000] | -0.443 [-0.634, -0.225]* |
| max=3.0 | ctx | strong | -0.066 [-0.078, -0.056]* | -0.072 [-0.087, -0.058]* | -0.001 [-0.002, +0.000] | -0.059 [-0.096, -0.024]* | +2.441 [+2.387, +2.500]* | +0.000 [+0.000, +0.000] | +0.154 [-0.105, +0.398] |
| max=3.0 | noctx | strong | +0.002 [-0.001, +0.005] | +0.013 [+0.007, +0.020]* | +0.003 [+0.001, +0.007]* | +0.005 [-0.011, +0.021] | +1.657 [+1.606, +1.705]* | +0.000 [+0.000, +0.000] | +0.207 [-0.023, +0.465] |

### `context_relevance_weight` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.121 [-0.136, -0.106]* | -0.133 [-0.151, -0.119]* | -0.002 [-0.005, +0.000] | -0.128 [-0.173, -0.085]* | +0.925 [+0.861, +0.985]* | +0.000 [+0.000, +0.000] | -0.458 [-0.658, -0.246]* |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | ctx | strong | +0.002 [-0.003, +0.007] | +0.016 [+0.007, +0.027]* | -0.001 [-0.002, +0.000] | -0.021 [-0.045, +0.003] | -0.155 [-0.191, -0.117]* | +0.000 [+0.000, +0.000] | +0.312 [+0.152, +0.478]* |
| max=4.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `context_relevance_cap` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.121 [-0.136, -0.106]* | -0.133 [-0.151, -0.119]* | -0.002 [-0.005, +0.000] | -0.128 [-0.173, -0.085]* | +0.925 [+0.861, +0.985]* | +0.000 [+0.000, +0.000] | -0.458 [-0.658, -0.246]* |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `selection_diversity_bonus` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=1.0 | ctx | strong | -0.085 [-0.098, -0.073]* | -0.121 [-0.139, -0.104]* | -0.001 [-0.003, +0.001] | -0.109 [-0.155, -0.067]* | +0.177 [+0.117, +0.240]* | +0.000 [+0.000, +0.000] | +0.141 [-0.149, +0.433] |
| max=1.0 | noctx | strong | +0.005 [-0.000, +0.010] | +0.003 [-0.002, +0.007] | +0.000 [+0.000, +0.000] | +0.016 [-0.013, +0.043] | -0.479 [-0.523, -0.433]* | +0.000 [+0.000, +0.000] | +0.547 [+0.321, +0.809]* |

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
| min=0.0 | ctx | weak | +0.001 [-0.003, +0.005] | -0.012 [-0.021, -0.003]* | +0.001 [+0.000, +0.003] | -0.019 [-0.043, +0.005] | +0.010 [-0.014, +0.036] | -0.003 [-0.019, +0.013] | +0.015 [-0.117, +0.151] |
| min=0.0 | noctx | weak | -0.001 [-0.003, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.003 [-0.013, +0.008] | +0.019 [-0.007, +0.046] | +0.017 [+0.003, +0.031]* | -0.049 [-0.187, +0.079] |
| max=3.0 | ctx | weak | +0.001 [-0.002, +0.003] | +0.004 [-0.004, +0.012] | -0.001 [-0.003, +0.000] | -0.019 [-0.035, -0.003]* | -0.003 [-0.021, +0.017] | +0.007 [-0.005, +0.019] | -0.033 [-0.135, +0.067] |
| max=3.0 | noctx | weak | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.003 [-0.005, +0.013] | +0.023 [+0.007, +0.037]* | +0.006 [-0.003, +0.015] | +0.003 [-0.093, +0.103] |

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
| min=0.0 | ctx | strong | -0.013 [-0.026, -0.000]* | +0.009 [-0.007, +0.024] | +0.000 [-0.002, +0.002] | +0.013 [-0.019, +0.045] | +0.271 [+0.215, +0.331]* | +0.027 [-0.003, +0.056] | -0.686 [-0.942, -0.432]* |
| min=0.0 | noctx | strong | -0.005 [-0.010, +0.000] | +0.001 [-0.003, +0.004] | +0.000 [+0.000, +0.000] | +0.021 [-0.011, +0.053] | +0.482 [+0.422, +0.543]* | +0.060 [+0.031, +0.091]* | -1.912 [-2.135, -1.643]* |
| max=3.0 | ctx | weak | -0.002 [-0.004, -0.000]* | -0.009 [-0.014, -0.004]* | -0.001 [-0.002, +0.000] | -0.013 [-0.027, -0.003]* | -0.063 [-0.081, -0.045]* | +0.000 [+0.000, +0.000] | -0.105 [-0.205, -0.016]* |
| max=3.0 | noctx | weak | +0.000 [-0.000, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.003 [-0.008, +0.000] | -0.146 [-0.166, -0.126]* | +0.000 [+0.000, +0.000] | +0.034 [-0.025, +0.089] |

### `verbatim_recognized_unit` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=True | ctx | weak | -0.000 [-0.001, +0.001] | -0.002 [-0.005, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [-0.008, +0.008] | +0.044 [+0.028, +0.059]* | +0.000 [+0.000, +0.000] | -0.074 [-0.138, -0.020]* |
| flip=True | noctx | weak | -0.000 [-0.002, +0.002] | -0.001 [-0.002, +0.000] | +0.000 [+0.000, +0.000] | -0.005 [-0.013, +0.000] | +0.087 [+0.064, +0.109]* | +0.000 [+0.000, +0.000] | -0.113 [-0.185, -0.035]* |

### `intonation_profile_strength` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=1.0 | ctx | weak | +0.003 [-0.004, +0.009] | -0.026 [-0.037, -0.016]* | +0.000 [+0.000, +0.000] | -0.008 [-0.029, +0.013] | +0.016 [-0.013, +0.047] | +0.007 [-0.009, +0.022] | -0.759 [-0.901, -0.621]* |
| max=1.0 | noctx | weak | -0.000 [-0.004, +0.003] | +0.001 [+0.000, +0.003] | +0.000 [+0.000, +0.000] | -0.005 [-0.021, +0.011] | +0.005 [-0.024, +0.033] | +0.008 [-0.005, +0.022] | -0.874 [-1.031, -0.719]* |

### `length_context_adaptation` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | +0.003 [-0.003, +0.010] | -0.009 [-0.017, -0.002]* | +0.001 [+0.000, +0.002] | -0.008 [-0.024, +0.005] | +0.022 [-0.008, +0.051] | +0.001 [-0.011, +0.014] | +0.209 [+0.052, +0.369]* |
| min=0.0 | noctx | weak | +0.002 [-0.000, +0.005] | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | -0.003 [-0.016, +0.011] | +0.006 [-0.022, +0.037] | +0.006 [-0.006, +0.018] | +0.225 [+0.079, +0.379]* |
| max=3.0 | ctx | weak | +0.001 [-0.006, +0.008] | +0.002 [-0.006, +0.010] | +0.000 [+0.000, +0.000] | +0.005 [-0.013, +0.024] | -0.005 [-0.033, +0.022] | +0.001 [-0.013, +0.013] | -0.144 [-0.285, -0.004]* |
| max=3.0 | noctx | weak | +0.001 [-0.002, +0.004] | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | -0.005 [-0.013, +0.000] | -0.011 [-0.039, +0.017] | -0.003 [-0.015, +0.009] | -0.289 [-0.444, -0.149]* |

### `markov_order` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| alt=2 | ctx | strong | -0.046 [-0.065, -0.025]* | -0.058 [-0.080, -0.035]* | -0.001 [-0.004, +0.001] | -0.051 [-0.099, +0.000] | +0.074 [+0.008, +0.150]* | -0.021 [-0.065, +0.021] | -0.466 [-0.768, -0.143]* |
| alt=2 | noctx | strong | -0.001 [-0.007, +0.005] | +0.002 [-0.002, +0.006] | +0.000 [+0.000, +0.000] | +0.003 [-0.029, +0.037] | -0.173 [-0.240, -0.107]* | +0.035 [-0.003, +0.075] | -0.446 [-0.695, -0.149]* |

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
| min=-2.0 | ctx | weak | +0.001 [-0.009, +0.009] | -0.005 [-0.021, +0.011] | -0.001 [-0.003, +0.001] | -0.016 [-0.056, +0.024] | +0.021 [-0.023, +0.064] | -0.002 [-0.028, +0.024] | +0.253 [+0.012, +0.505]* |
| min=-2.0 | noctx | weak | +0.004 [-0.002, +0.009] | +0.002 [+0.000, +0.005] | +0.000 [+0.000, +0.000] | +0.024 [+0.000, +0.051] | -0.002 [-0.047, +0.043] | +0.021 [-0.004, +0.047] | +0.086 [-0.111, +0.291] |
| max=2.0 | ctx | weak | -0.001 [-0.006, +0.004] | -0.004 [-0.013, +0.005] | -0.001 [-0.003, +0.000] | -0.019 [-0.043, +0.003] | +0.001 [-0.027, +0.028] | +0.005 [-0.012, +0.021] | +0.016 [-0.121, +0.165] |
| max=2.0 | noctx | weak | -0.000 [-0.004, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.008 [-0.027, +0.011] | +0.005 [-0.019, +0.029] | -0.006 [-0.019, +0.008] | +0.047 [-0.091, +0.183] |

### `markov_entropy_pivot` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 (parent on) | ctx | weak | -0.001 [-0.006, +0.004] | -0.001 [-0.009, +0.006] | -0.001 [-0.002, +0.000] | -0.016 [-0.037, +0.003] | -0.002 [-0.025, +0.022] | +0.000 [-0.015, +0.014] | +0.035 [-0.073, +0.143] |
| min=0.0 (parent on) | noctx | weak | -0.001 [-0.003, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.003 [-0.016, +0.013] | +0.004 [-0.015, +0.023] | -0.006 [-0.016, +0.005] | +0.071 [-0.055, +0.190] |
| max=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 (parent on) | ctx | weak | -0.001 [-0.002, +0.000] | -0.001 [-0.004, +0.002] | +0.000 [+0.000, +0.000] | -0.008 [-0.019, +0.000] | -0.006 [-0.015, +0.004] | +0.002 [-0.003, +0.007] | -0.020 [-0.063, +0.022] |
| max=1.0 (parent on) | noctx | inert | -0.001 [-0.003, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.003 [-0.013, +0.005] | +0.003 [-0.001, +0.007] | -0.013 [-0.062, +0.037] |

### `markov_entropy_temp_min` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.05 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.05 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.05 (parent on) | ctx | weak | -0.001 [-0.006, +0.004] | +0.000 [-0.007, +0.007] | -0.001 [-0.002, +0.000] | -0.011 [-0.032, +0.011] | +0.001 [-0.023, +0.025] | -0.001 [-0.015, +0.013] | +0.022 [-0.077, +0.125] |
| min=0.05 (parent on) | noctx | weak | -0.000 [-0.001, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [-0.013, +0.013] | +0.011 [-0.007, +0.028] | -0.004 [-0.013, +0.005] | +0.081 [-0.035, +0.194] |
| max=50.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=50.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=50.0 (parent on) | ctx | weak | -0.002 [-0.007, +0.004] | -0.006 [-0.016, +0.004] | -0.001 [-0.003, +0.000] | -0.024 [-0.048, -0.003]* | +0.008 [-0.021, +0.038] | +0.005 [-0.013, +0.025] | +0.060 [-0.087, +0.218] |
| max=50.0 (parent on) | noctx | weak | +0.001 [-0.003, +0.004] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.008 [-0.027, +0.011] | +0.001 [-0.025, +0.027] | -0.003 [-0.017, +0.013] | +0.092 [-0.049, +0.241] |

### `markov_entropy_temp_max` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.05 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.05 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.05 (parent on) | ctx | weak | +0.001 [-0.009, +0.009] | -0.005 [-0.021, +0.011] | -0.001 [-0.003, +0.001] | -0.016 [-0.056, +0.024] | +0.021 [-0.023, +0.064] | -0.003 [-0.029, +0.023] | +0.259 [+0.014, +0.505]* |
| min=0.05 (parent on) | noctx | weak | +0.004 [-0.002, +0.009] | +0.002 [+0.000, +0.005] | +0.000 [+0.000, +0.000] | +0.029 [+0.003, +0.059]* | +0.001 [-0.043, +0.047] | +0.021 [-0.005, +0.046] | +0.076 [-0.113, +0.281] |
| max=50.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=50.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=50.0 (parent on) | ctx | weak | -0.001 [-0.006, +0.004] | +0.000 [-0.007, +0.007] | -0.001 [-0.002, +0.000] | -0.011 [-0.032, +0.011] | +0.001 [-0.023, +0.025] | -0.001 [-0.015, +0.013] | +0.022 [-0.077, +0.125] |
| max=50.0 (parent on) | noctx | weak | -0.000 [-0.001, +0.001] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [-0.013, +0.013] | +0.011 [-0.007, +0.028] | -0.004 [-0.013, +0.005] | +0.081 [-0.035, +0.194] |

### `markov_branching_degenerate_max` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=20.0 | ctx | strong | -0.066 [-0.079, -0.054]* | -0.056 [-0.071, -0.041]* | +0.000 [-0.003, +0.003] | -0.072 [-0.109, -0.035]* | -0.691 [-0.739, -0.643]* | -2.595 [-2.623, -2.565]* | +0.337 [+0.051, +0.628]* |
| max=20.0 | noctx | strong | +0.002 [-0.003, +0.008] | +0.005 [+0.001, +0.010]* | +0.002 [+0.000, +0.005] | +0.024 [-0.008, +0.059] | -1.393 [-1.439, -1.343]* | -2.630 [-2.655, -2.604]* | +0.451 [+0.181, +0.749]* |

### `markov_branching_candidate_floor` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 (parent on) | ctx | strong | -0.076 [-0.091, -0.062]* | -0.067 [-0.083, -0.053]* | +0.001 [-0.002, +0.004] | -0.067 [-0.104, -0.032]* | -0.691 [-0.741, -0.639]* | -2.375 [-2.455, -2.292]* | +0.477 [+0.193, +0.751]* |
| min=1 (parent on) | noctx | strong | +0.003 [-0.002, +0.007] | +0.009 [+0.003, +0.015]* | +0.003 [+0.001, +0.007]* | +0.000 [-0.027, +0.027] | -1.315 [-1.372, -1.253]* | -2.422 [-2.501, -2.343]* | +0.710 [+0.455, +0.988]* |
| max=5 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5 (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `markov_short_half_life_days` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | weak | -0.002 [-0.009, +0.005] | -0.001 [-0.015, +0.011] | -0.001 [-0.003, +0.001] | -0.027 [-0.061, +0.005] | +0.019 [-0.015, +0.056] | +0.021 [+0.001, +0.043]* | +0.065 [-0.123, +0.253] |
| min=1.0 (parent on) | noctx | weak | -0.001 [-0.004, +0.003] | +0.000 [-0.002, +0.002] | +0.000 [+0.000, +0.000] | +0.008 [-0.008, +0.027] | +0.009 [-0.025, +0.039] | +0.005 [-0.014, +0.025] | -0.049 [-0.224, +0.117] |
| max=14.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=14.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=14.0 (parent on) | ctx | weak | -0.002 [-0.010, +0.005] | -0.003 [-0.015, +0.009] | -0.001 [-0.003, +0.001] | -0.016 [-0.048, +0.016] | +0.021 [-0.015, +0.058] | +0.008 [-0.013, +0.031] | +0.131 [-0.069, +0.324] |
| max=14.0 (parent on) | noctx | weak | -0.002 [-0.006, +0.002] | +0.000 [-0.002, +0.002] | +0.000 [+0.000, +0.000] | +0.005 [-0.011, +0.024] | +0.017 [-0.019, +0.049] | +0.011 [-0.009, +0.030] | -0.007 [-0.177, +0.154] |

### `markov_long_compression_beta` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.5 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.5 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.5 (parent on) | ctx | strong | -0.003 [-0.010, +0.004] | -0.003 [-0.016, +0.009] | -0.001 [-0.003, +0.001] | -0.035 [-0.067, -0.003]* | +0.022 [-0.013, +0.059] | +0.009 [-0.012, +0.031] | +0.062 [-0.128, +0.253] |
| min=0.5 (parent on) | noctx | weak | -0.002 [-0.005, +0.002] | +0.000 [-0.002, +0.002] | +0.000 [+0.000, +0.000] | +0.003 [-0.013, +0.021] | +0.015 [-0.017, +0.048] | +0.001 [-0.021, +0.020] | -0.023 [-0.197, +0.144] |
| max=0.75 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=0.75 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=0.75 (parent on) | ctx | weak | -0.003 [-0.010, +0.004] | +0.001 [-0.012, +0.013] | -0.001 [-0.003, +0.001] | -0.029 [-0.064, +0.005] | +0.021 [-0.013, +0.059] | +0.013 [-0.007, +0.033] | +0.099 [-0.095, +0.270] |
| max=0.75 (parent on) | noctx | weak | -0.001 [-0.005, +0.002] | +0.000 [-0.002, +0.002] | +0.000 [+0.000, +0.000] | +0.005 [-0.011, +0.024] | +0.009 [-0.022, +0.042] | +0.013 [-0.005, +0.031] | -0.061 [-0.238, +0.103] |

### `markov_alpha_calm` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=1.0 | ctx | weak | -0.005 [-0.017, +0.006] | -0.002 [-0.021, +0.016] | +0.000 [-0.003, +0.003] | -0.016 [-0.061, +0.029] | +0.025 [-0.028, +0.078] | -0.040 [-0.075, -0.006]* | -0.294 [-0.557, -0.016]* |
| max=1.0 | noctx | strong | +0.000 [-0.005, +0.006] | +0.001 [-0.001, +0.004] | +0.000 [+0.000, +0.000] | +0.048 [+0.016, +0.080]* | +0.000 [-0.053, +0.049] | +0.026 [-0.001, +0.053] | -0.046 [-0.283, +0.198] |

### `markov_interp_order2_weight` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=1.0 | ctx | strong | +0.003 [-0.010, +0.015] | -0.012 [-0.031, +0.006] | -0.001 [-0.004, +0.001] | -0.053 [-0.096, -0.013]* | +0.083 [+0.027, +0.135]* | +0.012 [-0.025, +0.049] | -0.007 [-0.265, +0.263] |
| max=1.0 | noctx | weak | -0.000 [-0.006, +0.005] | +0.000 [-0.003, +0.003] | +0.000 [+0.000, +0.000] | +0.016 [-0.016, +0.048] | +0.096 [+0.045, +0.149]* | +0.020 [-0.009, +0.048] | -0.006 [-0.229, +0.245] |

### `markov_collocation_bonus` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=2.0 | ctx | strong | -0.081 [-0.095, -0.068]* | -0.073 [-0.090, -0.057]* | +0.005 [+0.001, +0.009]* | +0.064 [+0.021, +0.104]* | -0.744 [-0.795, -0.692]* | +0.000 [+0.000, +0.000] | +3.056 [+2.697, +3.412]* |
| max=2.0 | noctx | strong | +0.006 [+0.001, +0.011]* | +0.001 [-0.003, +0.004] | +0.007 [+0.003, +0.012]* | +0.123 [+0.088, +0.160]* | -1.365 [-1.418, -1.312]* | +0.000 [+0.000, +0.000] | +2.998 [+2.680, +3.327]* |

### `markov_collocation_break_penalty` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=2.0 | ctx | strong | -0.046 [-0.057, -0.035]* | -0.068 [-0.083, -0.054]* | -0.001 [-0.003, +0.000] | -0.093 [-0.128, -0.056]* | -0.345 [-0.388, -0.301]* | +0.000 [+0.000, +0.000] | -0.758 [-0.980, -0.499]* |
| max=2.0 | noctx | strong | -0.004 [-0.008, -0.001]* | +0.004 [+0.001, +0.008]* | +0.001 [+0.000, +0.002] | +0.003 [-0.019, +0.024] | -0.678 [-0.721, -0.636]* | +0.000 [+0.000, +0.000] | -0.790 [-0.987, -0.585]* |

### `markov_hot_ngram_meme_ordering` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=True | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flip=True | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flip=True (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| flip=True (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `markov_seeded_candidate_ratio` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=0.7 | ctx | strong | +0.075 [+0.056, +0.094]* | -0.030 [-0.055, -0.007]* | -0.001 [-0.004, +0.001] | +0.043 [-0.011, +0.096] | -0.111 [-0.174, -0.047]* | +0.129 [+0.092, +0.167]* | +0.103 [-0.187, +0.395] |
| max=0.7 | noctx | strong | +0.048 [+0.040, +0.056]* | +0.018 [+0.011, +0.025]* | +0.000 [+0.000, +0.000] | +0.075 [+0.035, +0.120]* | -0.197 [-0.262, -0.137]* | +0.159 [+0.125, +0.195]* | +0.067 [-0.175, +0.335] |

### `hot_ngram_slot_ratio` — **inert**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| max=0.7 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=0.7 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `markov_seed_branch_min` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | strong | +0.119 [+0.098, +0.141]* | -0.006 [-0.031, +0.017] | -0.001 [-0.004, +0.001] | +0.029 [-0.029, +0.085] | -0.195 [-0.257, -0.129]* | +0.113 [+0.073, +0.152]* | +0.127 [-0.155, +0.411] |
| min=1.0 (parent on) | noctx | strong | +0.062 [+0.053, +0.072]* | +0.031 [+0.022, +0.040]* | +0.000 [+0.000, +0.000] | +0.067 [+0.029, +0.107]* | -0.174 [-0.241, -0.115]* | +0.165 [+0.132, +0.201]* | -0.054 [-0.306, +0.229] |
| max=1000.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `markov_seed_branch_ideal` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | strong | +0.076 [+0.057, +0.095]* | -0.027 [-0.052, -0.005]* | -0.001 [-0.003, +0.002] | +0.032 [-0.019, +0.080] | -0.099 [-0.163, -0.037]* | +0.137 [+0.101, +0.179]* | +0.129 [-0.149, +0.435] |
| min=1.0 (parent on) | noctx | strong | +0.048 [+0.040, +0.055]* | +0.017 [+0.010, +0.023]* | +0.000 [+0.000, +0.000] | +0.064 [+0.027, +0.104]* | -0.198 [-0.264, -0.138]* | +0.159 [+0.125, +0.196]* | +0.091 [-0.156, +0.358] |
| max=1000.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000.0 (parent on) | ctx | strong | +0.072 [+0.052, +0.092]* | -0.035 [-0.057, -0.013]* | -0.002 [-0.005, +0.000] | +0.024 [-0.029, +0.077] | -0.117 [-0.177, -0.055]* | +0.136 [+0.099, +0.171]* | +0.147 [-0.130, +0.448] |
| max=1000.0 (parent on) | noctx | strong | +0.050 [+0.042, +0.058]* | +0.015 [+0.009, +0.021]* | +0.000 [+0.000, +0.000] | +0.051 [+0.013, +0.091]* | -0.198 [-0.263, -0.134]* | +0.172 [+0.139, +0.209]* | +0.060 [-0.173, +0.346] |

### `markov_seed_branch_max` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5000.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5000.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=5000.0 (parent on) | ctx | strong | +0.079 [+0.057, +0.100]* | -0.039 [-0.062, -0.018]* | -0.002 [-0.005, +0.000] | +0.024 [-0.032, +0.077] | -0.107 [-0.170, -0.041]* | +0.151 [+0.113, +0.188]* | +0.189 [-0.107, +0.501] |
| max=5000.0 (parent on) | noctx | strong | +0.057 [+0.049, +0.067]* | +0.012 [+0.007, +0.018]* | +0.000 [+0.000, +0.000] | +0.053 [+0.016, +0.093]* | -0.187 [-0.256, -0.123]* | +0.174 [+0.141, +0.212]* | +0.005 [-0.241, +0.284] |

### `markov_seed_min_support` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1.0 (parent on) | ctx | strong | +0.077 [+0.056, +0.097]* | +0.013 [-0.012, +0.035] | -0.001 [-0.004, +0.001] | +0.043 [-0.008, +0.099] | -0.157 [-0.221, -0.095]* | +0.123 [+0.085, +0.159]* | +0.429 [+0.132, +0.742]* |
| min=1.0 (parent on) | noctx | strong | +0.046 [+0.038, +0.055]* | +0.025 [+0.017, +0.033]* | +0.000 [+0.000, +0.000] | +0.059 [+0.019, +0.101]* | -0.214 [-0.280, -0.151]* | +0.162 [+0.130, +0.197]* | +0.027 [-0.248, +0.307] |
| max=500.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=500.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=500.0 (parent on) | ctx | inert | -0.000 [-0.004, +0.003] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.002 [-0.004, +0.008] | +0.003 [-0.001, +0.009] | -0.002 [-0.053, +0.055] |
| max=500.0 (parent on) | noctx | inert | +0.001 [-0.001, +0.004] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.009 [+0.001, +0.017]* | +0.004 [-0.001, +0.009] | -0.006 [-0.039, +0.028] |

### `markov_seed_min_score` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 (parent on) | ctx | strong | +0.119 [+0.096, +0.142]* | -0.027 [-0.051, -0.003]* | -0.001 [-0.004, +0.001] | +0.040 [-0.019, +0.099] | -0.173 [-0.239, -0.107]* | +0.139 [+0.100, +0.179]* | +0.197 [-0.106, +0.519] |
| min=0.0 (parent on) | noctx | strong | +0.070 [+0.061, +0.081]* | +0.023 [+0.015, +0.030]* | +0.000 [+0.000, +0.000] | +0.075 [+0.035, +0.117]* | -0.189 [-0.257, -0.131]* | +0.173 [+0.139, +0.211]* | -0.047 [-0.305, +0.243] |
| max=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 (parent on) | ctx | weak | +0.014 [+0.007, +0.021]* | -0.001 [-0.013, +0.012] | +0.001 [-0.001, +0.003] | +0.005 [-0.024, +0.035] | -0.008 [-0.035, +0.019] | +0.015 [-0.002, +0.031] | +0.005 [-0.128, +0.136] |
| max=1.0 (parent on) | noctx | weak | +0.003 [+0.001, +0.006]* | +0.002 [+0.000, +0.005] | +0.000 [+0.000, +0.000] | +0.008 [-0.016, +0.032] | -0.028 [-0.055, -0.003]* | +0.014 [-0.001, +0.029] | +0.065 [-0.051, +0.182] |

### `markov_seed_min_token_len` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=1 (parent on) | ctx | strong | +0.094 [+0.074, +0.116]* | -0.035 [-0.059, -0.012]* | -0.001 [-0.003, +0.002] | +0.035 [-0.024, +0.091] | -0.135 [-0.199, -0.067]* | +0.147 [+0.109, +0.187]* | +0.070 [-0.220, +0.369] |
| min=1 (parent on) | noctx | strong | +0.054 [+0.045, +0.062]* | +0.019 [+0.012, +0.026]* | +0.000 [+0.000, +0.000] | +0.080 [+0.040, +0.125]* | -0.197 [-0.268, -0.135]* | +0.168 [+0.135, +0.205]* | +0.045 [-0.218, +0.318] |
| max=20 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=20 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=20 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=20 (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `markov_seed_head_share` — **gated**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| min=0.0 (parent on) | ctx | strong | +0.069 [+0.049, +0.087]* | -0.029 [-0.053, -0.008]* | -0.001 [-0.004, +0.001] | -0.024 [-0.077, +0.029] | -0.115 [-0.173, -0.055]* | +0.134 [+0.097, +0.170]* | -0.369 [-0.667, -0.060]* |
| min=0.0 (parent on) | noctx | strong | +0.035 [+0.028, +0.042]* | +0.004 [+0.000, +0.008] | +0.001 [+0.000, +0.002] | +0.024 [-0.011, +0.056] | -0.219 [-0.283, -0.158]* | +0.151 [+0.117, +0.185]* | -0.341 [-0.611, -0.048]* |
| max=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 (parent on) | ctx | strong | +0.077 [+0.058, +0.095]* | -0.035 [-0.059, -0.012]* | -0.001 [-0.004, +0.001] | +0.032 [-0.024, +0.083] | -0.083 [-0.145, -0.021]* | +0.129 [+0.093, +0.165]* | -0.403 [-0.723, -0.092]* |
| max=1.0 (parent on) | noctx | strong | +0.045 [+0.037, +0.053]* | +0.013 [+0.007, +0.019]* | +0.001 [+0.000, +0.002] | +0.080 [+0.037, +0.120]* | -0.159 [-0.223, -0.097]* | +0.148 [+0.116, +0.182]* | -0.365 [-0.601, -0.095]* |

### `enable_backoff` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=False | ctx | strong | -0.024 [-0.044, -0.004]* | +0.031 [+0.007, +0.052]* | -0.001 [-0.004, +0.001] | -0.064 [-0.117, -0.011]* | -0.068 [-0.131, -0.001]* | +0.037 [-0.003, +0.081] | -0.014 [-0.283, +0.285] |
| flip=False | noctx | weak | +0.003 [-0.003, +0.010] | +0.003 [-0.001, +0.009] | +0.000 [+0.000, +0.000] | +0.037 [+0.000, +0.077] | -0.110 [-0.178, -0.045]* | +0.025 [-0.017, +0.066] | +0.349 [+0.097, +0.621]* |

### `markov_jump_probability` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | +0.001 [-0.011, +0.013] | -0.057 [-0.077, -0.039]* | +0.001 [-0.001, +0.003] | +0.000 [-0.045, +0.045] | +0.002 [-0.052, +0.057] | -0.039 [-0.075, -0.004]* | +0.081 [-0.181, +0.375] |
| min=0.0 | noctx | strong | +0.000 [-0.005, +0.006] | +0.001 [-0.002, +0.003] | +0.000 [+0.000, +0.000] | -0.013 [-0.040, +0.011] | -0.193 [-0.248, -0.134]* | +0.003 [-0.027, +0.033] | -0.255 [-0.497, +0.006] |
| max=1.0 | ctx | strong | -0.015 [-0.033, +0.003] | +0.033 [+0.013, +0.055]* | -0.002 [-0.005, +0.000] | -0.029 [-0.075, +0.016] | +0.161 [+0.089, +0.227]* | -0.009 [-0.051, +0.031] | -0.047 [-0.304, +0.261] |
| max=1.0 | noctx | strong | +0.000 [-0.005, +0.006] | -0.001 [-0.005, +0.002] | +0.000 [+0.000, +0.000] | +0.016 [-0.019, +0.048] | +0.404 [+0.339, +0.472]* | +0.025 [-0.017, +0.065] | -0.364 [-0.611, -0.087]* |

### `context_jump_boost` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | weak | -0.001 [-0.004, +0.002] | -0.021 [-0.034, -0.008]* | -0.001 [-0.002, +0.000] | -0.005 [-0.032, +0.024] | +0.031 [+0.006, +0.057]* | -0.006 [-0.026, +0.014] | -0.066 [-0.217, +0.089] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=10.0 | ctx | weak | -0.002 [-0.007, +0.004] | +0.015 [-0.002, +0.033] | -0.001 [-0.003, +0.000] | -0.016 [-0.045, +0.016] | +0.049 [+0.010, +0.089]* | +0.006 [-0.017, +0.033] | -0.205 [-0.395, +0.014] |
| max=10.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `verbatim_extension_share` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.015 [-0.029, -0.002]* | +0.012 [-0.005, +0.027] | +0.001 [-0.001, +0.003] | +0.005 [-0.024, +0.032] | -0.275 [-0.325, -0.225]* | +0.027 [-0.003, +0.056] | -0.451 [-0.704, -0.215]* |
| min=0.0 | noctx | strong | -0.000 [-0.005, +0.005] | +0.005 [+0.001, +0.010]* | +0.000 [+0.000, +0.000] | +0.013 [-0.016, +0.045] | -0.717 [-0.771, -0.667]* | +0.060 [+0.031, +0.091]* | -0.877 [-1.133, -0.598]* |
| max=1.0 | ctx | weak | +0.002 [-0.002, +0.006] | -0.003 [-0.009, +0.004] | +0.000 [+0.000, +0.000] | +0.000 [-0.013, +0.013] | -0.029 [-0.051, -0.008]* | +0.001 [-0.014, +0.016] | -0.067 [-0.173, +0.029] |
| max=1.0 | noctx | weak | -0.000 [-0.002, +0.001] | +0.001 [-0.001, +0.003] | +0.000 [+0.000, +0.000] | -0.005 [-0.021, +0.011] | -0.069 [-0.095, -0.043]* | +0.010 [-0.003, +0.023] | -0.127 [-0.242, +0.002] |

### `order_mix_probability` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | weak | -0.012 [-0.033, +0.006] | +0.027 [+0.005, +0.048]* | +0.000 [-0.003, +0.003] | -0.016 [-0.067, +0.035] | -0.046 [-0.109, +0.020] | +0.002 [-0.041, +0.047] | +0.429 [+0.151, +0.749]* |
| min=0.0 | noctx | weak | +0.004 [-0.002, +0.009] | +0.005 [+0.000, +0.009] | +0.000 [+0.000, +0.000] | +0.011 [-0.024, +0.045] | -0.107 [-0.173, -0.047]* | -0.009 [-0.051, +0.035] | +0.469 [+0.203, +0.759]* |
| max=1.0 | ctx | strong | +0.003 [-0.010, +0.015] | -0.012 [-0.031, +0.006] | -0.001 [-0.004, +0.001] | -0.053 [-0.096, -0.013]* | +0.084 [+0.029, +0.138]* | +0.013 [-0.023, +0.049] | -0.006 [-0.264, +0.264] |
| max=1.0 | noctx | weak | -0.000 [-0.006, +0.005] | +0.000 [-0.003, +0.003] | +0.000 [+0.000, +0.000] | +0.016 [-0.016, +0.048] | +0.096 [+0.045, +0.149]* | +0.020 [-0.009, +0.048] | -0.006 [-0.229, +0.245] |

### `slot_mutation_probability` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | +0.001 [-0.017, +0.018] | +0.007 [-0.016, +0.029] | -0.001 [-0.004, +0.001] | -0.051 [-0.101, -0.003]* | +0.159 [+0.091, +0.233]* | +0.452 [+0.419, +0.484]* | +0.157 [-0.131, +0.447] |
| min=0.0 | noctx | strong | +0.002 [-0.005, +0.007] | -0.001 [-0.004, +0.003] | +0.000 [+0.000, +0.000] | +0.011 [-0.021, +0.043] | +0.252 [+0.186, +0.317]* | +0.469 [+0.441, +0.499]* | +0.017 [-0.227, +0.304] |
| max=1.0 | ctx | strong | -0.013 [-0.032, +0.003] | -0.028 [-0.047, -0.009]* | -0.001 [-0.003, +0.001] | -0.075 [-0.123, -0.029]* | -0.263 [-0.320, -0.199]* | -1.344 [-1.380, -1.309]* | +0.142 [-0.128, +0.452] |
| max=1.0 | noctx | strong | +0.001 [-0.005, +0.006] | +0.002 [-0.002, +0.007] | +0.000 [+0.000, +0.000] | +0.045 [+0.011, +0.083]* | -0.643 [-0.705, -0.587]* | -1.328 [-1.361, -1.292]* | +0.037 [-0.213, +0.333] |

### `hot_ngram_min_count` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1 | ctx | inert | -0.000 [-0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.001 [-0.001, +0.004] | +0.000 [+0.000, +0.000] | -0.001 [-0.003, +0.000] |
| min=1 | noctx | weak | -0.000 [-0.001, +0.000] | +0.002 [+0.000, +0.005] | +0.000 [+0.000, +0.000] | +0.019 [+0.003, +0.037]* | +0.012 [-0.007, +0.030] | -0.003 [-0.014, +0.006] | +0.061 [-0.006, +0.131] |
| min=1 (parent on) | ctx | inert | -0.000 [-0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.005 [-0.001, +0.011] | +0.000 [-0.003, +0.003] | -0.006 [-0.034, +0.023] |
| min=1 (parent on) | noctx | strong | +0.002 [-0.004, +0.008] | +0.002 [-0.002, +0.006] | +0.001 [+0.000, +0.002] | +0.112 [+0.069, +0.157]* | +0.066 [-0.003, +0.136] | -0.003 [-0.045, +0.035] | -0.050 [-0.317, +0.243] |
| max=1000 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1000 (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `hot_ngram_recency_share` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | inert | +0.000 [-0.001, +0.002] | +0.001 [-0.001, +0.003] | +0.001 [+0.000, +0.002] | +0.003 [+0.000, +0.008] | +0.003 [-0.003, +0.009] | -0.004 [-0.007, -0.001]* | -0.015 [-0.042, +0.011] |
| min=0.0 | noctx | strong | +0.001 [-0.002, +0.005] | +0.001 [+0.000, +0.003] | +0.000 [+0.000, +0.000] | +0.064 [+0.037, +0.093]* | -0.023 [-0.059, +0.013] | +0.001 [-0.016, +0.020] | -0.053 [-0.191, +0.083] |
| min=0.0 (parent on) | ctx | inert | +0.000 [-0.001, +0.002] | +0.001 [-0.001, +0.003] | +0.001 [+0.000, +0.002] | +0.003 [+0.000, +0.008] | +0.007 [-0.001, +0.015] | -0.003 [-0.008, +0.003] | +0.007 [-0.029, +0.043] |
| min=0.0 (parent on) | noctx | strong | +0.002 [-0.004, +0.009] | +0.000 [-0.003, +0.004] | +0.000 [+0.000, +0.000] | +0.467 [+0.413, +0.523]* | +0.085 [+0.010, +0.150]* | -0.006 [-0.047, +0.034] | +0.031 [-0.245, +0.323] |
| max=1.0 | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 (parent on) | ctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 (parent on) | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `fuzzy_context_casefold` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| flip=False | ctx | weak | -0.006 [-0.011, -0.001]* | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.002 [-0.010, +0.013] | +0.007 [+0.000, +0.014] | +0.025 [-0.025, +0.068] |
| flip=False | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `reply_context_bias` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | weak | +0.006 [-0.001, +0.014] | -0.027 [-0.041, -0.013]* | -0.001 [-0.002, +0.000] | -0.008 [-0.040, +0.021] | +0.054 [+0.019, +0.091]* | +0.005 [-0.017, +0.026] | -0.013 [-0.197, +0.155] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | ctx | weak | -0.011 [-0.019, -0.003]* | +0.018 [+0.003, +0.035]* | +0.000 [-0.001, +0.002] | +0.003 [-0.040, +0.043] | +0.055 [+0.015, +0.091]* | +0.017 [-0.007, +0.041] | +0.058 [-0.123, +0.273] |
| max=4.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `reply_context_start_bias` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | strong | -0.153 [-0.171, -0.135]* | -0.214 [-0.235, -0.194]* | -0.002 [-0.005, +0.000] | -0.176 [-0.232, -0.125]* | +0.633 [+0.565, +0.704]* | +0.049 [+0.011, +0.086]* | -0.407 [-0.709, -0.069]* |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=4.0 | ctx | strong | +0.029 [+0.014, +0.044]* | +0.053 [+0.036, +0.069]* | +0.000 [-0.001, +0.002] | +0.013 [-0.021, +0.048] | -0.049 [-0.098, -0.001]* | -0.035 [-0.068, -0.004]* | +0.233 [-0.025, +0.497] |
| max=4.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `generation_attempts_with_context` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0 | ctx | strong | -0.183 [-0.201, -0.165]* | -0.222 [-0.242, -0.203]* | -0.002 [-0.005, +0.000] | -0.208 [-0.261, -0.157]* | +0.648 [+0.582, +0.721]* | +0.052 [+0.012, +0.091]* | -0.252 [-0.546, +0.035] |
| min=0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=10 | ctx | strong | +0.030 [+0.023, +0.038]* | +0.023 [+0.013, +0.032]* | -0.001 [-0.002, +0.000] | +0.021 [-0.008, +0.051] | -0.046 [-0.079, -0.018]* | -0.074 [-0.094, -0.055]* | +0.140 [-0.006, +0.269] |
| max=10 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `context_start_affinity` — **weak**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=1.0 | ctx | weak | -0.026 [-0.041, -0.012]* | +0.014 [-0.005, +0.031] | -0.001 [-0.002, +0.000] | -0.021 [-0.061, +0.016] | -0.020 [-0.077, +0.037] | +0.007 [-0.025, +0.040] | -0.029 [-0.286, +0.237] |
| min=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=10.0 | ctx | weak | +0.009 [-0.008, +0.026] | +0.009 [-0.009, +0.027] | +0.000 [-0.002, +0.002] | -0.019 [-0.067, +0.029] | -0.002 [-0.053, +0.053] | -0.028 [-0.059, +0.005] | +0.164 [-0.095, +0.435] |
| max=10.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

### `context_anchor_splice_probability` — **strong**

| extreme | mode | class | context_affinity_without_copy | exact_context_copy_rate | repetition_rate | historical_meme_rate | structural_window_escape | structural_pool_ecb | mean_response_length |
|---|---|---|---|---|---|---|---|---|---|
| min=0.0 | ctx | strong | -0.087 [-0.102, -0.071]* | -0.019 [-0.041, +0.003] | +0.000 [-0.003, +0.003] | -0.016 [-0.067, +0.035] | +0.257 [+0.193, +0.313]* | +0.021 [-0.014, +0.059] | -0.102 [-0.377, +0.187] |
| min=0.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |
| max=1.0 | ctx | strong | +0.097 [+0.082, +0.114]* | +0.025 [+0.005, +0.045]* | -0.001 [-0.004, +0.002] | -0.005 [-0.051, +0.040] | +0.077 [+0.019, +0.134]* | +0.012 [-0.021, +0.045] | +0.635 [+0.396, +0.902]* |
| max=1.0 | noctx | inert | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] |

## Вердикт (владельцу; текст ниже — не выход инструмента)

**Прогон.** 10 отсоединённых воркеров, копия прода 01.09, 500 генераций × 3 сида на арм и режим, 126 армов × 2 режима, 34 минуты стены. Ни один арм не упал. Оба C0 совпали с базлайном версии `bababb4b7693` (`baseline-2026-09-01-{ctx,noctx}.json`) по всем 14 метрикам с точностью до 1e-9, значит перепись сравнима с гридами Track A/B этой сессии.

**Что считать результатом.** Из 56 свипнутых ручек: strong 23, weak 12, gated 13, inert 8. Ни одна ручка не «мертва» статически: первый вариант отчёта показал шесть «dead» (`mood_*`, `rare_event_daily_cap`), это была ошибка скана, который пропускал `runtime_state.py` целиком; починено в этой же заявке, тест `test_runtime_state_reads_count_as_sites`.

**Inert 8 — по строкам, потому что «inert» здесь трёх разных сортов:**

| ручка | почему ноль | предложение |
|---|---|---|
| `hot_ngram_slot_ratio`, `markov_hot_ngram_meme_ordering` | **смерть по данным при дефолтных порогах**: `get_hot` для чата харнесса даёт 0 фраз при `min_count` 3 / `recency_share` 0.5, 8 при 2 / 0.25, 2 при 3 / 0.0 (замер на копии 01.09 после штатного decay в `init`). То же в проде: «горячие фразы: пусто в 100%» в `/stats` 02.09. Подтверждение от соседей: `hot_ngram_recency_share` 0.0 при включённом маршруте даёт meme rate +0.467*, `hot_ngram_min_count` 1 — +0.112* | не трогать; дефолты порогов решает O18 |
| `markov_cache_incremental` | тёплый путь обязан совпадать с холодным (hash-контракт, §2 CLAUDE.md); нулевая дельта — ожидаемое доказательство, не мёртвая ручка | оставить, ручка производительности |
| `markov_shadow_order4_enabled` | тень order-4 только пишет телеметрию и по построению не влияет на ответ | оставить до закрытия эксперимента с тенью; удалять вместе с ним |
| `recent_reply_penalty_strength` | харнесс стартует с пустой историей ответов (`run.py:361`), штрафу нечего штрафовать; слепота харнесса, не ручки | оставить; поведение закрыто юнит-тестами анти-повтора |
| `auto_capitalize_replies`, `reply_flavor_strength`, `emoji_append_chance` | слой формы: метрики считаются по casefold-контентным токенам и не видят регистр, концовки и эмодзи | перепись не судья; решение по глазам владельца в проде |

**Weak 12** (`randomness_strength`, `candidate_selection_temperature`, `repetition_penalty_strength`, `verbatim_recognized_unit`, `intonation_profile_strength`, `length_context_adaptation`, `markov_entropy_temp_gain`, `context_jump_boost`, `hot_ngram_min_count`, `fuzzy_context_casefold`, `reply_context_bias`, `context_start_affinity`). Три наблюдения. Температура отбора и сила случайности «weak» на своих экстремумах: это согласуется с M3R-100 (у 36% ctx-входов окно из одной траектории, крутить температуру нечего). `markov_entropy_temp_gain` weak — ещё одно подтверждение §1.2 конституции: token-level механика инертна. Полоса «weak» при 1500 записей на арм ограничена мощностью, поэтому «weak» = «кандидат на сужение домена», не «удалить».

**Gated 13** — тройка энтропии, пол ветвления, полураспад и бета компрессии, семейство `markov_seed_*` (7). Все ведут себя как задумано: без родителя ноль, с родителем двигают (`markov_seed_min_score` 0.0 при родителе: affinity +0.119*, escape −0.173*). Не мёртвые.

**Strong 23** — оставить; у ручек длины и запаса окна экстремум ломает форму (ожидаемо), потолки доменов пересматривать не надо.

**Вход для M3R-150.** По цифрам переписи удалять нечего: кандидатов на удаление ноль, на сужение домена — 12 weak, на решение по глазам — 3 ручки формы, на решение O18 — 2 ручки горячего пула. 42 ручки конвейера ответа (`reply_*`, `mood_*`, `pivo_*`, причуды, темп) переписью не измеряются по построению и остаются вне её вердикта.
