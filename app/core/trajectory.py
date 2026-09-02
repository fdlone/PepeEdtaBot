"""Trajectory identity of a candidate: the edges its walk used (M3R-011).

One definition for both the structural escape gate (``tools/eval``) and the
diversity bonus of the selection window (M3R-100): two readers must not drift
apart on what "the same trajectory" means. Pure functions, no I/O.
"""

from __future__ import annotations

# Below this edge overlap two candidates count as different trajectories.
# Registered as ``structural_escape.edge_overlap_similar`` in
# ``tools/eval/eval_thresholds.yaml``; a test pins the two to the same value.
EDGE_OVERLAP_SIMILAR = 0.5


def trajectory_edges(tokens: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    """Adjacent token pairs of a candidate — the edges its walk used.

    Edges rather than tokens because the question is which *path through the
    chain* the candidate took: two answers built from the same words in a
    different order are different walks, and a bag of words cannot tell them
    apart. Content tokens only, so two candidates differing by a comma are the
    same trajectory rather than two.
    """
    return frozenset(zip(tokens, tokens[1:], strict=False))


def edge_overlap(
    a: frozenset[tuple[str, str]], b: frozenset[tuple[str, str]]
) -> float:
    """Share of edges two candidates have in common, normalized by the smaller.

    Normalizing by the smaller set — not by the union, as Jaccard would — makes
    a candidate whose edges are a subset of another's (the same walk cut short,
    or extended past the same base) count as the SAME trajectory. Jaccard would
    call them two, inflating the project's headline metric exactly where there
    is no difference to report.

    Two edgeless candidates (a single token each) overlap fully only when they
    are the same edgeless set; otherwise a token has no path to share.
    """
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / min(len(a), len(b))
