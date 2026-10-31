"""Deterministic parity statistics for compiled raw margins."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def ranking_analysis(reference: np.ndarray, candidate: np.ndarray) -> dict[str, int]:
    """Count all pairwise order/tie relations exactly in O(n log n)."""
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.ndim != 1 or candidate.shape != reference.shape:
        raise ValueError("ranking inputs must be same-length vectors")
    n = len(reference)
    values, candidate_rank = np.unique(candidate, return_inverse=True)
    bit = np.zeros(len(values) + 1, dtype=np.int64)

    def add(index: int) -> None:
        index += 1
        while index < len(bit):
            bit[index] += 1
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        index += 1
        while index > 0:
            total += int(bit[index])
            index -= index & -index
        return total

    order = np.argsort(reference, kind="stable")
    inversions = collapsed = 0
    native_tie_pairs = native_ties_preserved = 0
    previous = 0
    start = 0
    while start < n:
        end = start + 1
        while end < n and reference[order[end]] == reference[order[start]]:
            end += 1
        group = candidate_rank[order[start:end]]
        for rank in group:
            less_or_equal = prefix(int(rank))
            less = prefix(int(rank) - 1) if rank else 0
            inversions += previous - less_or_equal
            collapsed += less_or_equal - less
        size = end - start
        native_tie_pairs += size * (size - 1) // 2
        _, counts = np.unique(group, return_counts=True)
        native_ties_preserved += int(np.sum(counts * (counts - 1) // 2))
        for rank in group:
            add(int(rank))
        previous += size
        start = end
    total_pairs = n * (n - 1) // 2
    ordered_native_pairs = total_pairs - native_tie_pairs
    return {
        "rows": n,
        "total_pairs": total_pairs,
        "native_ordered_pairs": ordered_native_pairs,
        "concordant_ordered_pairs": ordered_native_pairs - inversions - collapsed,
        "rank_inversions": inversions,
        "native_order_collapsed_to_compiled_tie": collapsed,
        "native_tie_pairs": native_tie_pairs,
        "native_ties_preserved": native_ties_preserved,
        "native_ties_split": native_tie_pairs - native_ties_preserved,
        "native_unique_values": int(np.unique(reference).size),
        "compiled_unique_values": int(np.unique(candidate).size),
    }


def parity_statistics(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    labels = np.asarray(labels)
    error = candidate - reference
    fpr_ref, tpr_ref, _ = roc_curve(labels, reference)
    fpr_candidate, tpr_candidate, _ = roc_curve(labels, candidate)
    native_auc = float(roc_auc_score(labels, reference))
    compiled_auc = float(roc_auc_score(labels, candidate))
    native_ks = float(np.max(tpr_ref - fpr_ref))
    compiled_ks = float(np.max(tpr_candidate - fpr_candidate))
    return {
        "max_absolute_margin_error": float(np.max(np.abs(error))),
        "mean_absolute_margin_error": float(np.mean(np.abs(error))),
        "root_mean_squared_margin_error": float(np.sqrt(np.mean(error * error))),
        "minimum_signed_error": float(np.min(error)),
        "maximum_signed_error": float(np.max(error)),
        "exact_margin_matches": int(np.count_nonzero(error == 0)),
        "sign_inversions": int(np.count_nonzero((reference >= 0) != (candidate >= 0))),
        "ranking": ranking_analysis(reference, candidate),
        "native_roc_auc": native_auc,
        "compiled_roc_auc": compiled_auc,
        "roc_auc_difference": compiled_auc - native_auc,
        "native_ks": native_ks,
        "compiled_ks": compiled_ks,
        "ks_difference": compiled_ks - native_ks,
    }
