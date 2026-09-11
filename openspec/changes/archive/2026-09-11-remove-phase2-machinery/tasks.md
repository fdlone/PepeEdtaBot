# Tasks — remove-phase2-machinery (O22, группа 2)

## 1. Код

- [x] 1.1 `EntropySampling` и протяжка через `markov.py` / `response_generator.py`; `_step_power` — только телеметрия; `branching_aware_target` и ранняя остановка
- [x] 1.2 Шесть ручек: реестр, `RuntimeTunables`, `.env.example`; фикстуры тестов; `tests/test_markov2r_phase2.py` и классы фазы 2 в других тестах сняты
- [x] 1.3 Харнесс: гейт `phase2_entropy` в отчёте, блок порогов, `matrix_phase2_grid.yaml`, `knob_census.GATED_BY`, оверрайды `eval_generation`
- [x] 1.4 `generation_hash` на обоих снимках совпал (`hash-log.md`)

## 2. Приёмка

- [x] 2.1 `ruff`, `mypy app/`, полный `unittest`, покрытие, смок, гарды документов
- [x] 2.2 Документы: карта §2.1/§4, пайплайн §4/§5.1, STATUS; спека `generation-entropy-sampling` снята при архиве
