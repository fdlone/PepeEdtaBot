"""Prompt set: loading and deterministic generation (doc 05 §1.2).

Four categories — generic, topical, meme-bait, short/degenerate — of at least
30 prompts each. Topical and meme-bait prompts are derived from the snapshot by
a seeded script, never written by hand. Until a temporal snapshot exists
(audit §10.1, approved workaround) the source is ``snapshot_full``: topical
prompts are built around low-df tokens with adequate chain support (the
seed-score precursors of TZ §9.4), meme-bait around the top hot / verbatim
n-grams as the proxy for the chat's known memes.

The generated file also carries the meme list used by ``meme_regression_pass``
(doc 05 §3.5): the fixed n-grams whose reproducibility the gate checks.

The file's ``version`` is a content hash: regenerating with the same snapshot
and seed yields the identical file (spec: prompt provenance).
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.lexicon import STOPWORDS  # noqa: E402
from app.core.markov import content_tokens, tokenize  # noqa: E402

PROMPTS_PATH = Path(__file__).with_name("eval_prompts.yaml")
CATEGORIES = ("generic", "topical", "meme-bait", "short-degenerate")
MIN_PER_CATEGORY = 30
# Fixed degenerate inputs (doc 05 §1.2: односложные и шумовые входы). These are
# literals of the protocol, not chat-derived data.
DEGENERATE_LITERALS = ("?", "ну", "и", "ааа", "да", "нет", "лол", "!", "…", "ок")


@dataclass(frozen=True, slots=True)
class PromptSet:
    version: str
    snapshot_label: str
    seed: int
    categories: dict[str, list[str]]
    memes: list[list[str]]  # casefolded content n-grams for the §3.5 gate


def _content_cf(text: str) -> list[str]:
    return [token.casefold() for token in content_tokens(tokenize(text))]


def _version_of(categories: dict[str, list[str]], memes: list[list[str]]) -> str:
    payload = json.dumps({"categories": categories, "memes": memes}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_prompts(path: Path = PROMPTS_PATH) -> PromptSet:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    categories = {name: list(raw["categories"][name]) for name in CATEGORIES}
    memes = [list(meme) for meme in raw.get("memes", [])]
    for name, prompts in categories.items():
        if len(prompts) < MIN_PER_CATEGORY:
            raise ValueError(f"category {name}: {len(prompts)} prompts < {MIN_PER_CATEGORY}")
    expected = _version_of(categories, memes)
    if raw.get("version") != expected:
        raise ValueError(
            f"prompt set version mismatch: file says {raw.get('version')!r}, "
            f"content hashes to {expected!r} — the file was hand-edited"
        )
    return PromptSet(
        version=raw["version"],
        snapshot_label=str(raw.get("snapshot_label", "unknown")),
        seed=int(raw.get("seed", 0)),
        categories=categories,
        memes=memes,
    )


def generate_prompts(
    db_path: Path,
    *,
    chat_id: int,
    seed: int,
    snapshot_label: str,
    per_category: int = MIN_PER_CATEGORY,
) -> PromptSet:
    """Build the prompt set from ``snapshot_full`` deterministically."""
    rng = random.Random(seed)
    con = sqlite3.connect(db_path)
    try:
        messages = [
            row[0]
            for row in con.execute(
                "SELECT normalized_text FROM messages WHERE chat_id = ? "
                "AND normalized_text != '' ORDER BY id",
                (chat_id,),
            )
        ]
        support = {
            row[0]: int(row[1])
            for row in con.execute(
                "SELECT w3, SUM(cnt) FROM transitions WHERE chat_id = ? GROUP BY w3",
                (chat_id,),
            )
        }
        hot_ngrams = [
            tuple(token for token in row[:3] if token)
            for row in con.execute(
                "SELECT w1, w2, w3 FROM chat_hot_ngrams WHERE chat_id = ? "
                "ORDER BY cnt DESC, w1, w2, w3 LIMIT ?",
                (chat_id, per_category * 3),
            )
        ]
    finally:
        con.close()
    if len(messages) < per_category:
        raise ValueError(f"snapshot has {len(messages)} messages; need >= {per_category}")

    contents = [_content_cf(message) for message in messages]

    # generic: ordinary messages with enough content to react to.
    generic_pool = [m for m, c in zip(messages, contents) if len(c) >= 3]
    generic = rng.sample(generic_pool, min(per_category, len(generic_pool)))

    # topical: one message per rare-but-supported token (TZ §9.4 precursors:
    # low df, adequate chain support, content-like, not a stopword).
    df: dict[str, int] = {}
    for content in contents:
        for token in set(content):
            df[token] = df.get(token, 0) + 1
    rare_tokens = sorted(
        token
        for token, count in df.items()
        if count <= 2 and len(token) >= 4 and token not in STOPWORDS
        and support.get(token, 0) >= 3
    )
    rng.shuffle(rare_tokens)
    topical: list[str] = []
    used_messages: set[str] = set()
    for token in rare_tokens:
        if len(topical) >= per_category:
            break
        for message, content in zip(messages, contents):
            if token in content and message not in used_messages:
                topical.append(message)
                used_messages.add(message)
                break
    if len(topical) < per_category:  # fall back to generic-like messages
        filler = [m for m in generic_pool if m not in used_messages and m not in topical]
        topical.extend(rng.sample(filler, min(per_category - len(topical), len(filler))))

    # meme-bait: messages containing a top hot n-gram; the n-grams themselves
    # form the meme list for the §3.5 gate. Bare n-gram text is the fallback
    # bait when no retained message contains it.
    memes: list[list[str]] = []
    meme_bait: list[str] = []
    for ngram in hot_ngrams:
        if len(meme_bait) >= per_category:
            break
        ngram_cf = [token.casefold() for token in ngram]
        if any(token in STOPWORDS or len(token) < 3 for token in ngram_cf):
            continue
        bait = next(
            (
                message
                for message, content in zip(messages, contents)
                if _contains_run(content, ngram_cf)
            ),
            " ".join(ngram),
        )
        memes.append(ngram_cf)
        meme_bait.append(bait)
    if len(meme_bait) < per_category:
        filler = [m for m in generic_pool if m not in meme_bait]
        meme_bait.extend(rng.sample(filler, min(per_category - len(meme_bait), len(filler))))

    # short/degenerate: the chat's own short messages + fixed noise literals.
    short_pool = [m for m, c in zip(messages, contents) if 0 < len(c) < 3]
    short = list(DEGENERATE_LITERALS)
    rng.shuffle(short_pool)
    short.extend(short_pool[: max(0, per_category - len(short))])
    while len(short) < per_category:  # tiny snapshots: recycle literals
        short.append(DEGENERATE_LITERALS[len(short) % len(DEGENERATE_LITERALS)])

    categories = {
        "generic": generic,
        "topical": topical[:per_category],
        "meme-bait": meme_bait[:per_category],
        "short-degenerate": short[:per_category],
    }
    return PromptSet(
        version=_version_of(categories, memes),
        snapshot_label=snapshot_label,
        seed=seed,
        categories=categories,
        memes=memes,
    )


def _contains_run(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )


def save_prompts(prompt_set: PromptSet, path: Path = PROMPTS_PATH) -> None:
    payload = {
        "version": prompt_set.version,
        "snapshot_label": prompt_set.snapshot_label,
        "seed": prompt_set.seed,
        "categories": {name: prompt_set.categories[name] for name in CATEGORIES},
        "memes": prompt_set.memes,
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
