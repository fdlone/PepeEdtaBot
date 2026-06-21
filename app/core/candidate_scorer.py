from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.core.lexicon import BAD_ENDING_WORDS, STOPWORDS
from app.core.markov import content_tokens

SHORT_REPLY_MAX_TOKENS = 3
NATURAL_LENGTH_MIN = 5
NATURAL_LENGTH_MAX = 14
LONG_REPLY_SOFT_LIMIT = 24

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

@dataclass(frozen=True, slots=True)
class CandidateScore:
    completion_quality: float
    lexical_diversity: float
    natural_length: float
    context_relevance: float
    repetition_penalty: float

    @property
    def total(self) -> float:
        return (
            self.completion_quality
            + self.lexical_diversity
            + self.natural_length
            + self.context_relevance
            - self.repetition_penalty
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


def natural_length(tokens: list[str]) -> float:
    length = len(_normalized_content(tokens))
    if length <= 0:
        return 0.0
    if length < NATURAL_LENGTH_MIN:
        return NATURAL_LENGTH_WEIGHT * (0.40 + 0.12 * length)
    if length <= NATURAL_LENGTH_MAX:
        return NATURAL_LENGTH_WEIGHT
    if length <= LONG_REPLY_SOFT_LIMIT:
        excess = length - NATURAL_LENGTH_MAX
        span = LONG_REPLY_SOFT_LIMIT - NATURAL_LENGTH_MAX
        return NATURAL_LENGTH_WEIGHT * (1.0 - 0.5 * excess / span)
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


def score_candidate(
    text: str,
    tokens: list[str],
    context_tokens: list[str],
) -> CandidateScore:
    return CandidateScore(
        completion_quality=completion_quality(text, tokens),
        lexical_diversity=lexical_diversity(tokens),
        natural_length=natural_length(tokens),
        context_relevance=context_relevance(tokens, context_tokens),
        repetition_penalty=repetition_penalty(tokens),
    )
