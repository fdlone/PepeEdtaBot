import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

P = Path(__file__).with_name("sweep_2026-07-20_results.jsonl")
rows = [json.loads(line) for line in P.read_text(encoding="utf-8").splitlines()]

by_config = defaultdict(list)
for r in rows:
    by_config[r["config"]].append(r)

METRICS = [
    "context_anchored_win_rate", "context_top_rate", "context_win_given_top",
    "novel_ngram_share_mean", "pure_corpus_reply_rate",
    "verbatim_run_ratio_mean", "reply_ge90pct_verbatim_rate",
    "distinct_1", "distinct_2", "repeated_bigram_ratio",
    "connective_reply_rate", "winner_multijump_rate",
    "empty_result_rate", "avg_length_tokens", "length_mirror_gap",
    "context_token_overlap_mean", "avg_generation_latency_ms",
    "slot_mutation_win_rate", "verbatim_extension_win_rate",
]

BASELINES = ("baseline_registry", "baseline_live")
order = list(BASELINES) + [c for c in by_config if c not in BASELINES]

print(f"{'config':22s}", *[f"{m[:14]:>15s}" for m in METRICS])
for cfg in order:
    rs = by_config[cfg]
    vals = []
    for m in METRICS:
        v = [r.get(m, 0.0) for r in rs]
        vals.append(mean(v))
    print(f"{cfg:22s}", *[f"{v:15.4f}" for v in vals])
