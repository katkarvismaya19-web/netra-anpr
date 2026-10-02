"""
netra.metrics
=============

Metrics used to evaluate the recognition stage.

* **Plate accuracy** - percentage of plates read *exactly* right. This is the
  number that matters in practice (one wrong character = wrong vehicle).
* **Character Error Rate (CER)** - edit distance between prediction and truth
  divided by the length of the truth. Shows how close wrong readings are.
"""

from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: minimum insertions, deletions and substitutions."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def character_error_rate(predictions: list[str], targets: list[str]) -> float:
    errors = sum(edit_distance(p, t) for p, t in zip(predictions, targets))
    total = sum(len(t) for t in targets)
    return errors / max(total, 1)


def plate_accuracy(predictions: list[str], targets: list[str]) -> float:
    return sum(p == t for p, t in zip(predictions, targets)) / max(len(targets), 1)
