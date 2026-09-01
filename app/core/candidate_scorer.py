from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from app.core.lexicon import BAD_ENDING_WORDS, STOPWORDS
from app.core.markov import build_windows, content_tokens, tokenize
from app.core.morphology import stem_token

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
    base_weights: tuple[float, float, float] | None = None,
) -> str:
    """Pick a target length mode using short/medium/long weights.

    Mood modulation clamps each scaled multiplier at 0.0, so a legal config
    with a single positive base weight (e.g. ``0,0,1`` plus sleepy at strength
    >= 2.0) can zero the whole vector — ``random.choices`` raises on all-zero
    weights. Fall back to the pre-mood ``base_weights``, then to uniform.
    """
    if sum(weights) <= 0.0:
        if base_weights is not None and sum(base_weights) > 0.0:
            weights = base_weights
        else:
            weights = (1.0, 1.0, 1.0)
    return rng.choices(population=LENGTH_MODES, weights=weights, k=1)[0]


# Length mirroring: the mode was sampled from fixed weights, so the message
# being answered had no say in it -- a three-token answer to a three-token
# question landed in "long" mode 20% of the time and paid ~0.4 natural_length
# for being the right length. These are the ends of the ramp: an incoming
# message at or below SHORT gets the full tilt toward short, at or above LONG
# the full tilt toward long, and everything between interpolates.
CONTEXT_LENGTH_SHORT_TOKENS = 4
CONTEXT_LENGTH_LONG_TOKENS = 14


def context_length_weights(
    weights: tuple[float, float, float],
    incoming_tokens: int,
    strength: float,
) -> tuple[float, float, float]:
    """Tilt short/medium/long weights toward the answered message's own length.

    ``strength`` is the tilt at the ends of the ramp: the short and long
    weights are divided/multiplied by ``1 + strength``, so 0.0 returns the
    weights untouched (the pre-conditioning behaviour) and 1.0 doubles the
    long weight for a long incoming message while halving the short one.
    The medium weight is the pivot and never moves.
    """
    if strength <= 0.0 or incoming_tokens <= 0:
        return weights
    span = CONTEXT_LENGTH_LONG_TOKENS - CONTEXT_LENGTH_SHORT_TOKENS
    ramp = (incoming_tokens - CONTEXT_LENGTH_SHORT_TOKENS) / span
    # -1.0 at the short end of the ramp, +1.0 at the long end.
    position = min(1.0, max(0.0, ramp)) * 2.0 - 1.0
    tilt = (1.0 + strength) ** position
    short_weight, medium_weight, long_weight = weights
    return (short_weight / tilt, medium_weight, long_weight * tilt)

CLEAN_END_BONUS = 0.35
BALANCED_DELIMITERS_BONUS = 0.25
UNBALANCED_DELIMITERS_PENALTY = 0.50
BAD_ENDING_PENALTY = 0.80

NATURAL_LENGTH_WEIGHT = 1.00
# 1.6 (2026-07-09): at 0.80 the topic term could not outweigh the completion +
# diversity + natural_length ceiling (2.35), so fluent off-topic candidates won.
# Raised together with SELECTION_SCORE_MARGIN 0.3 -- neither alone was enough:
# IDF fixed which candidate ranks top, the margin fixed whether it survives the
# softmax roll. eval_prod: context_win_given_top 0.55 -> 0.89, verbatim flat.
# The cap is pinned to the weight: idf_context_relevance is already normalized
# to [0, 1], so a lower cap would only clip the strongest on-topic matches.
CONTEXT_RELEVANCE_WEIGHT = 1.60
CONTEXT_RELEVANCE_CAP = 1.60

# Token-repeat weights fold in the former lexical_diversity reward: for a
# reply the repeated-token ratio IS the diversity deficit (1 - unique/len), so
# the old "+1.0 diversity - 0.60 repeats" pair collapses into a single 1.60
# penalty slope (short replies had a 0.40 slope on a 0.40 base -> 1.00 slope
# plus a flat 0.20 offset preserving the old short-vs-long score gap).
REPEATED_TOKEN_WEIGHT = 1.60
SHORT_REPEATED_TOKEN_WEIGHT = 1.00
SHORT_REPLY_SCORE_OFFSET = 0.20
REPEATED_BIGRAM_WEIGHT = 1.00
REPEATED_TRIGRAM_WEIGHT = 1.30

# Verbatim-quote penalty: content n-gram size shared with the corpus index.
# A candidate whose 4-grams all come from one training message is a quote and
# loses to recombined candidates in selection.
VERBATIM_NGRAM_SIZE = 4

def verbatim_ngram_windows(tokens: list[str]) -> list[tuple[str, ...]]:
    """Casefolded content-token 4-grams of one message.

    Единственная реализация «что считается окном»: по ней и пишется
    накопительный индекс цитат при обучении, и считается доля совпадений при
    оценке кандидата. Раньше инфраструктура держала свою копию (вместе с
    копиями набора пунктуации и размера n-граммы), потому что импорт из ядра
    замкнул бы цикл — ядро зависело от инфраструктуры. Цикла больше нет.
    """
    content = [token.casefold() for token in content_tokens(tokens)]
    return build_windows(content, VERBATIM_NGRAM_SIZE)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    completion_quality: float
    natural_length: float
    context_relevance: float
    repetition_penalty: float
    recent_penalty: float = 0.0
    verbatim_penalty: float = 0.0
    # M2R-320: bonus minus penalty from the chat's active collocations. Signed
    # on purpose — the components are counted separately in telemetry, and the
    # score only needs their net effect. 0 until the phase-4 gate raises the
    # weights from their neutral defaults.
    collocation_delta: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.completion_quality
            + self.natural_length
            + self.context_relevance
            + self.collocation_delta
            - self.repetition_penalty
            - self.recent_penalty
            - self.verbatim_penalty
        )


def _normalized_content(tokens: list[str]) -> list[str]:
    return [token.lower() for token in content_tokens(tokens)]


def meaningful_tokens(tokens: list[str]) -> list[str]:
    return [
        token
        for token in _normalized_content(tokens)
        if token not in STOPWORDS
    ]


def stemmed_meaningful_tokens(tokens: list[str]) -> set[str]:
    """Stemmed content-token set: the fold key for relevance overlap."""
    return {stem_token(token) for token in meaningful_tokens(tokens)}


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


def build_token_idf(documents: Iterable[list[str]]) -> dict[str, float]:
    """Inverse document frequency of each content token across ``documents``.

    A document is one training message. Tokens appearing in most messages
    (pronouns, fillers) get an IDF near 0; rare ones (names, jargon) get a high
    one. Smoothed so a token present in every document still scores above 0.
    """
    document_count = 0
    seen: Counter[str] = Counter()
    for tokens in documents:
        document_count += 1
        seen.update(stemmed_meaningful_tokens(tokens))
    if not document_count:
        return {}
    return {
        token: math.log((document_count + 1) / (count + 1)) + 1.0
        for token, count in seen.items()
    }


def idf_context_relevance(
    tokens: list[str],
    context_tokens: list[str],
    idf: Mapping[str, float],
) -> float:
    """Share of the context's *informative* mass that the candidate echoes back.

    The plain ``context_relevance`` divides the overlap by the candidate length,
    so a one-word echo sharing a pronoun outscores a long on-topic continuation
    sharing a proper noun -- raising CONTEXT_RELEVANCE_WEIGHT only amplifies
    that. Here each shared token contributes its IDF, and the total is
    normalized by the context's own IDF mass: matching a rare name is worth much
    more than matching "кто", and candidate length no longer enters.

    Overlap is computed on approximate stems (``stem_token``): exact-form
    matching scored "гнойному пидору" at zero against "гнойный пидор", muting
    the context bonus for most inflected Russian answers. The IDF mapping is
    keyed by the same stems (``build_token_idf``).
    """
    candidate = stemmed_meaningful_tokens(tokens)
    context = stemmed_meaningful_tokens(context_tokens)
    if not candidate or not context:
        return 0.0
    # A candidate whose informative tokens are all echoed from the context adds
    # nothing new -- it is a parrot, not a reply. Without this guard a pure echo
    # scores the *maximum* relevance (it returns 100% of whatever context mass
    # it shares), and short echoes slip past both the verbatim gate (needs
    # 4-grams) and the short_context_copy reject.
    if candidate <= context:
        return 0.0
    if not idf:
        return context_relevance(tokens, context_tokens)
    default = max(idf.values())
    context_mass = sum(idf.get(token, default) for token in context)
    if context_mass <= 0.0:
        return 0.0
    shared_mass = sum(idf.get(token, default) for token in candidate & context)
    return min(
        CONTEXT_RELEVANCE_CAP,
        (shared_mass / context_mass) * CONTEXT_RELEVANCE_WEIGHT,
    )


def _repeated_ratio(items: list[tuple[str, ...]] | list[str]) -> float:
    """Share of ``items`` that are duplicate occurrences (0 for empty input)."""
    if not items:
        return 0.0
    counts = Counter(items)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(items)


def _repeated_ngram_ratio(tokens: list[str], size: int) -> float:
    return _repeated_ratio(build_windows(tokens, size))


def repetition_penalty(tokens: list[str]) -> float:
    """Internal-repetition penalty, diversity reward folded in (see weights).

    The repeated-token term carries the former lexical_diversity component:
    ranking-wise the merge only removes a constant (+1.0 long / +0.8 short)
    from every total, which softmax selection and the margin band ignore.
    """
    content = _normalized_content(tokens)
    if not content:
        return 0.0
    is_short = len(content) <= SHORT_REPLY_MAX_TOKENS
    token_weight = SHORT_REPEATED_TOKEN_WEIGHT if is_short else REPEATED_TOKEN_WEIGHT
    return (
        _repeated_ratio(content) * token_weight
        + _repeated_ngram_ratio(content, 2) * REPEATED_BIGRAM_WEIGHT
        + _repeated_ngram_ratio(content, 3) * REPEATED_TRIGRAM_WEIGHT
        + (SHORT_REPLY_SCORE_OFFSET if is_short else 0.0)
    )


def build_recent_reply_trigrams(
    recent_texts: Iterable[str],
) -> set[tuple[str, ...]]:
    """Collect content trigrams of recently sent replies for overlap penalties."""
    trigrams: set[tuple[str, ...]] = set()
    for text in recent_texts:
        trigrams.update(build_windows(_normalized_content(tokenize(text)), 3))
    return trigrams


def recent_reply_overlap(
    tokens: list[str],
    recent_trigrams: set[tuple[str, ...]],
) -> float:
    """Share of the candidate's content trigrams already seen in recent replies."""
    if not recent_trigrams:
        return 0.0
    candidate_trigrams = set(build_windows(_normalized_content(tokens), 3))
    if not candidate_trigrams:
        return 0.0
    return len(candidate_trigrams & recent_trigrams) / len(candidate_trigrams)


def verbatim_ngram_overlap(
    tokens: list[str],
    corpus_ngrams: AbstractSet[tuple[str, ...]],
    size: int = VERBATIM_NGRAM_SIZE,
    *,
    exempt_recognized_unit: bool = False,
) -> float:
    """Share of the candidate's content ``size``-grams found in the corpus index.

    1.0 means every window of the candidate exists verbatim in some training
    message (a quote); recombination across messages produces novel windows and
    lowers the share. Candidates shorter than ``size`` content tokens score 0 —
    they are governed by the short-reply anti-repeat instead.

    ``exempt_recognized_unit`` (M3R-120) drops ONE recognized borrowed unit
    from the count — see ``recognized_unit_windows`` for what counts as one.
    Off by default: the guard has its own knob so its effect stays attributable
    in the phase 5 verdict, and the neutral path is byte-identical.
    """
    if not corpus_ngrams:
        return 0.0
    windows = build_windows(_normalized_content(tokens), size)
    if not windows:
        return 0.0
    hit_flags = [window in corpus_ngrams for window in windows]
    hits = sum(hit_flags)
    if not exempt_recognized_unit:
        return hits / len(windows)
    unit = recognized_unit_windows(hit_flags, size)
    remaining = len(windows) - unit
    if unit == 0 or remaining == 0:
        # remaining == 0: кроме единицы в кандидате ничего нет, то есть это
        # цитата, а не заимствование внутри своей речи — исключение
        # отменяется (design D2, роадмап: «штраф остаётся за ответ, целиком
        # корпусный»).
        return hits / len(windows)
    return (hits - unit) / remaining


# M3R-120: предел одной признанной единицы заимствования, в контентных
# токенах. Роадмап называет единицу («одна признанная фраза/якорная цитата»),
# но не определяет её операционально, а прямые прочтения не работают: окно
# цитирования — 4 контентных токена, фраза индекса M3R-000 — 2–3, поэтому
# «исключить окна внутри фразы» было бы пустой операцией, а «исключить все
# задетые окна» — слишком щедрым.
#
# Операциональное определение: единица — самый длинный непрерывный корпусный
# отрезок кандидата, если он не длиннее предела. Предел 6 = 3 токена фразы
# плюс охват окна (4−1): столько нужно, чтобы заимствованная фраза попала в
# единицу вместе с накрывающими её окнами. Длиннее — это уже предложение
# целиком, то есть цитата, и она платит.
VERBATIM_UNIT_MAX_TOKENS = 6


def recognized_unit_windows(hit_flags: list[bool], size: int) -> int:
    """Сколько окон занимает **одна** признанная единица (0, если её нет).

    Единица — самая длинная непрерывная серия корпусных окон: серия из ``k``
    окон покрывает ``k + size - 1`` контентных токенов, и единицей она
    считается, только пока это число не превышает ``VERBATIM_UNIT_MAX_TOKENS``.
    Более длинная серия — цитата, а не единица, и не исключается вовсе.
    """
    longest = current = 0
    for flag in hit_flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    if longest == 0 or longest + size - 1 > VERBATIM_UNIT_MAX_TOKENS:
        return 0
    return longest


# Corpus-based replies are the bot's voice — a training message plus its own
# continuation is welcome. Only near-pure quotes should pay: below this share
# of corpus 4-grams the verbatim penalty is zero, above it the penalty ramps
# linearly to full strength at share 1.0. With the tolerated base at 0.6 a
# typical extended copy (quote + ~8 novel tokens, share ~0.5) pays nothing,
# while a quote with one word changed (share ~0.9) still pays 75%.
VERBATIM_TOLERATED_SHARE = 0.6


def verbatim_quote_severity(share: float) -> float:
    """Map corpus 4-gram share to penalty severity in [0, 1]."""
    if share <= VERBATIM_TOLERATED_SHARE:
        return 0.0
    return (share - VERBATIM_TOLERATED_SHARE) / (1.0 - VERBATIM_TOLERATED_SHARE)


def score_candidate(
    text: str,
    tokens: list[str],
    context_tokens: list[str],
    length_mode: str = "medium",
) -> CandidateScore:
    return CandidateScore(
        completion_quality=completion_quality(text, tokens),
        natural_length=natural_length(tokens, length_mode),
        context_relevance=context_relevance(tokens, context_tokens),
        repetition_penalty=repetition_penalty(tokens),
    )
