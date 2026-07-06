from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from app.core.lexicon import BAD_ENDING_WORDS, STOPWORDS
from app.core.markov import content_tokens, tokenize

SHORT_REPLY_MAX_TOKENS = 3
NATURAL_LENGTH_MIN = 5
NATURAL_LENGTH_MAX = 14
LONG_REPLY_SOFT_LIMIT = 24

# Per-reply target length modes: the scorer's natural_length peak moves to the
# sampled mode's band, so replies stop clustering in the 5-14 token range.
LENGTH_MODES: tuple[str, ...] = ("short", "medium", "long")
LENGTH_MODE_BANDS: dict[str, tuple[int, int]] = {
    "short": (1, 4),
    "medium": (NATURAL_LENGTH_MIN, NATURAL_LENGTH_MAX),
    "long": (15, LONG_REPLY_SOFT_LIMIT),
}
# Tokens past the band's upper bound over which the score decays to 0.5
# (matches the historical medium-band decay: 14 -> 24).
NATURAL_LENGTH_DECAY_SPAN = LONG_REPLY_SOFT_LIMIT - NATURAL_LENGTH_MAX


def sample_length_mode(
    weights: tuple[float, float, float],
    rng: random.Random,
) -> str:
    """Pick a target length mode using short/medium/long weights."""
    return rng.choices(population=LENGTH_MODES, weights=weights, k=1)[0]

CLEAN_END_BONUS = 0.35
BALANCED_DELIMITERS_BONUS = 0.25
UNBALANCED_DELIMITERS_PENALTY = 0.50
BAD_ENDING_PENALTY = 0.80

LEXICAL_DIVERSITY_WEIGHT = 1.00
SHORT_LEXICAL_DIVERSITY_BASE = 0.40
SHORT_LEXICAL_DIVERSITY_WEIGHT = 0.40
NATURAL_LENGTH_WEIGHT = 1.00
CONTEXT_RELEVANCE_WEIGHT = 0.80
CONTEXT_RELEVANCE_CAP = 0.80

REPEATED_TOKEN_WEIGHT = 0.60
REPEATED_BIGRAM_WEIGHT = 1.00
REPEATED_TRIGRAM_WEIGHT = 1.30

# Verbatim-quote penalty: content n-gram size shared with the corpus index and
# the per-backoff-order coherence penalties. A candidate whose 4-grams all come
# from one training message is a quote; one that fell back to the 1-gram chain
# is word salad — both lose to recombined middle-path candidates in selection.
VERBATIM_NGRAM_SIZE = 4
ORDER1_COHERENCE_PENALTY = 0.70
ORDER2_COHERENCE_PENALTY = 0.10

@dataclass(frozen=True, slots=True)
class CandidateScore:
    completion_quality: float
    lexical_diversity: float
    natural_length: float
    context_relevance: float
    repetition_penalty: float
    recent_penalty: float = 0.0
    verbatim_penalty: float = 0.0
    coherence_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.completion_quality
            + self.lexical_diversity
            + self.natural_length
            + self.context_relevance
            - self.repetition_penalty
            - self.recent_penalty
            - self.verbatim_penalty
            - self.coherence_penalty
        )


def _normalized_content(tokens: list[str]) -> list[str]:
    return [token.lower() for token in content_tokens(tokens)]


def meaningful_tokens(tokens: list[str]) -> list[str]:
    return [
        token
        for token in _normalized_content(tokens)
        if token not in STOPWORDS
    ]


def _delimiter_balance(text: str) -> tuple[bool, bool]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    has_delimiters = False
    for character in text:
        if character in pairs:
            has_delimiters = True
            stack.append(pairs[character])
        elif character in pairs.values():
            has_delimiters = True
            if not stack or stack.pop() != character:
                return True, False
    if stack:
        return True, False
    has_quotes = any(character in text for character in ('"', "«", "»"))
    balanced_quotes = (
        text.count('"') % 2 == 0 and text.count("«") == text.count("»")
    )
    return has_delimiters or has_quotes, balanced_quotes


def completion_quality(text: str, tokens: list[str]) -> float:
    stripped = text.rstrip()
    content = _normalized_content(tokens)
    score = 0.0
    if stripped.endswith((".", "!", "?")):
        score += CLEAN_END_BONUS
    has_delimiters, balanced_delimiters = _delimiter_balance(stripped)
    if has_delimiters:
        if balanced_delimiters:
            score += BALANCED_DELIMITERS_BONUS
        else:
            score -= UNBALANCED_DELIMITERS_PENALTY
    if (
        not content
        or content[-1] in BAD_ENDING_WORDS
        or stripped.endswith(("(", "[", "{", "«"))
    ):
        score -= BAD_ENDING_PENALTY
    return score


def lexical_diversity(tokens: list[str]) -> float:
    content = _normalized_content(tokens)
    if not content:
        return 0.0
    diversity = len(set(content)) / len(content)
    if len(content) <= SHORT_REPLY_MAX_TOKENS:
        return SHORT_LEXICAL_DIVERSITY_BASE + (
            diversity * SHORT_LEXICAL_DIVERSITY_WEIGHT
        )
    return diversity * LEXICAL_DIVERSITY_WEIGHT


def natural_length(tokens: list[str], mode: str = "medium") -> float:
    lower, upper = LENGTH_MODE_BANDS[mode]
    length = len(_normalized_content(tokens))
    if length <= 0:
        return 0.0
    if length < lower:
        return NATURAL_LENGTH_WEIGHT * (0.40 + 0.50 * length / lower)
    if length <= upper:
        return NATURAL_LENGTH_WEIGHT
    excess = length - upper
    if excess <= NATURAL_LENGTH_DECAY_SPAN:
        return NATURAL_LENGTH_WEIGHT * (
            1.0 - 0.5 * excess / NATURAL_LENGTH_DECAY_SPAN
        )
    return NATURAL_LENGTH_WEIGHT * 0.5


def context_relevance(tokens: list[str], context_tokens: list[str]) -> float:
    candidate = set(meaningful_tokens(tokens))
    context = set(meaningful_tokens(context_tokens))
    if not candidate or not context:
        return 0.0
    overlap_ratio = len(candidate & context) / len(candidate)
    return min(CONTEXT_RELEVANCE_CAP, overlap_ratio * CONTEXT_RELEVANCE_WEIGHT)


def _repeated_ngram_ratio(tokens: list[str], size: int) -> float:
    if len(tokens) < size:
        return 0.0
    ngrams = [
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    ]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def repetition_penalty(tokens: list[str]) -> float:
    content = _normalized_content(tokens)
    if not content:
        return 0.0
    token_counts = Counter(content)
    repeated_tokens = sum(count - 1 for count in token_counts.values() if count > 1)
    token_ratio = repeated_tokens / len(content)
    return (
        token_ratio * REPEATED_TOKEN_WEIGHT
        + _repeated_ngram_ratio(content, 2) * REPEATED_BIGRAM_WEIGHT
        + _repeated_ngram_ratio(content, 3) * REPEATED_TRIGRAM_WEIGHT
    )


def build_recent_reply_trigrams(
    recent_texts: Iterable[str],
) -> set[tuple[str, str, str]]:
    """Collect content trigrams of recently sent replies for overlap penalties."""
    trigrams: set[tuple[str, str, str]] = set()
    for text in recent_texts:
        content = _normalized_content(tokenize(text))
        trigrams.update(
            (content[index], content[index + 1], content[index + 2])
            for index in range(len(content) - 2)
        )
    return trigrams


def recent_reply_overlap(
    tokens: list[str],
    recent_trigrams: set[tuple[str, str, str]],
) -> float:
    """Share of the candidate's content trigrams already seen in recent replies."""
    if not recent_trigrams:
        return 0.0
    content = _normalized_content(tokens)
    if len(content) < 3:
        return 0.0
    candidate_trigrams = {
        (content[index], content[index + 1], content[index + 2])
        for index in range(len(content) - 2)
    }
    return len(candidate_trigrams & recent_trigrams) / len(candidate_trigrams)


def verbatim_ngram_overlap(
    tokens: list[str],
    corpus_ngrams: frozenset[tuple[str, ...]] | set[tuple[str, ...]],
    size: int = VERBATIM_NGRAM_SIZE,
) -> float:
    """Share of the candidate's content ``size``-grams found in the corpus index.

    1.0 means every window of the candidate exists verbatim in some training
    message (a quote); recombination across messages produces novel windows and
    lowers the share. Candidates shorter than ``size`` content tokens score 0 —
    they are governed by the short-reply anti-repeat instead.
    """
    if not corpus_ngrams:
        return 0.0
    content = _normalized_content(tokens)
    if len(content) < size:
        return 0.0
    windows = [
        tuple(content[index : index + size])
        for index in range(len(content) - size + 1)
    ]
    hits = sum(1 for window in windows if window in corpus_ngrams)
    return hits / len(windows)


def coherence_penalty_for_order(markov_order_used: int) -> float:
    """Penalty for how far the walk had to back off (1-gram = word salad)."""
    if markov_order_used <= 1:
        return ORDER1_COHERENCE_PENALTY
    if markov_order_used == 2:
        return ORDER2_COHERENCE_PENALTY
    return 0.0


def score_candidate(
    text: str,
    tokens: list[str],
    context_tokens: list[str],
    length_mode: str = "medium",
) -> CandidateScore:
    return CandidateScore(
        completion_quality=completion_quality(text, tokens),
        lexical_diversity=lexical_diversity(tokens),
        natural_length=natural_length(tokens, length_mode),
        context_relevance=context_relevance(tokens, context_tokens),
        repetition_penalty=repetition_penalty(tokens),
    )
