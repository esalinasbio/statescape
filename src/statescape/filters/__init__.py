"""
Filtering layer for Statescape.

This module provides pure functions that take metric values (NumPy arrays)
and ConformerSet objects, and return new ConformerSets according to
threshold, top-k, or percentile-based predicates. All operations are
side-effect free and append to the provenance log of the resulting set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

from statescape.core.conformer import ConformerSet, FilterProvenance


MetricArray = np.ndarray
PredicateFn = Callable[[MetricArray], MetricArray]


@dataclass(frozen=True)
class PredicateDescription:
    """
    Human-readable description of a predicate applied to metric values.
    """

    name: str
    details: str


def threshold(
    values: Sequence[float],
    *,
    op: str,
    value: float,
) -> MetricArray:
    """
    Build a boolean mask by thresholding metric values.

    Parameters
    ----------
    values:
        Metric values for each conformer.
    op:
        Comparison operator: one of '<', '<=', '>', '>='.
    value:
        Threshold value.
    """
    arr = np.asarray(values, dtype=float)
    if op == "<":
        return arr < value
    if op == "<=":
        return arr <= value
    if op == ">":
        return arr > value
    if op == ">=":
        return arr >= value
    raise ValueError(f"Unsupported operator '{op}'. Expected one of '<', '<=', '>', '>='.")


def top_k(
    values: Sequence[float],
    *,
    k: int,
    largest: bool = False,
) -> MetricArray:
    """
    Build a boolean mask keeping the top-k conformers according to a metric.

    Parameters
    ----------
    values:
        Metric values for each conformer.
    k:
        Number of conformers to keep.
    largest:
        If True, keep largest values; otherwise, keep smallest.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if k <= 0:
        raise ValueError("k must be > 0.")
    if k >= n:
        return np.ones_like(arr, dtype=bool)

    order = np.argsort(arr)
    if largest:
        order = order[::-1]
    keep_idx = order[:k]
    mask = np.zeros_like(arr, dtype=bool)
    mask[keep_idx] = True
    return mask


def percentile_cutoff(
    values: Sequence[float],
    *,
    percentile: float,
    keep: str = "below",
) -> MetricArray:
    """
    Build a boolean mask using a percentile cutoff on metric values.

    Parameters
    ----------
    values:
        Metric values for each conformer.
    percentile:
        Percentile in [0, 100].
    keep:
        'below' to keep values <= percentile,
        'above' to keep values >= percentile.
    """
    arr = np.asarray(values, dtype=float)
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be in [0, 100].")

    cutoff = float(np.percentile(arr, percentile))
    if keep == "below":
        return arr <= cutoff
    if keep == "above":
        return arr >= cutoff
    raise ValueError("keep must be 'below' or 'above'.")


def filter_by_mask(
    conformers: ConformerSet,
    mask: Sequence[bool],
    *,
    metric_name: str,
    predicate_desc: PredicateDescription,
) -> Tuple[ConformerSet, ConformerSet]:
    """
    Apply a boolean mask to a ConformerSet, returning (kept, dropped) sets.

    Both outputs carry an updated immutable provenance record describing
    the filtering operation.
    """
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.size != len(conformers):
        raise ValueError(
            f"Mask length ({mask_arr.size}) must match number of conformers ({len(conformers)})."
        )

    kept_set, dropped_set = conformers.split_by_mask(mask_arr)

    n_before = len(conformers)
    n_after = len(kept_set)
    n_removed = n_before - n_after

    record = FilterProvenance(
        metric=metric_name,
        predicate=f"{predicate_desc.name}: {predicate_desc.details}",
        n_before=n_before,
        n_after=n_after,
        n_removed=n_removed,
    )

    new_provenance = kept_set.provenance + (record,)
    kept_with_prov = ConformerSet(
        conformers=kept_set.conformers,
        metadata=kept_set.metadata,
        parent=kept_set.parent,
        provenance=new_provenance,
    )

    # Dropped set also gets the same provenance record for traceability.
    dropped_with_prov = ConformerSet(
        conformers=dropped_set.conformers,
        metadata=dropped_set.metadata,
        parent=dropped_set.parent,
        provenance=new_provenance,
    )

    return kept_with_prov, dropped_with_prov


def filter_by_metric(
    conformers: ConformerSet,
    values: Sequence[float],
    predicate_mask_fn: Callable[[MetricArray], MetricArray],
    *,
    metric_name: str,
    predicate_desc: PredicateDescription,
) -> Tuple[ConformerSet, ConformerSet]:
    """
    Filter a ConformerSet based on precomputed metric values and a predicate.

    This function is pure: it does not modify the input ConformerSet and
    always returns new ConformerSet instances.

    Parameters
    ----------
    conformers:
        Input ConformerSet.
    values:
        Metric values, one per conformer.
    predicate_mask_fn:
        Callable mapping metric array -> boolean mask array.
    metric_name:
        Name of the metric used (for provenance).
    predicate_desc:
        Human-readable description of the predicate (for provenance).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size != len(conformers):
        raise ValueError(
            f"Metric array length ({arr.size}) must match number of conformers ({len(conformers)})."
        )

    mask = predicate_mask_fn(arr)
    return filter_by_mask(
        conformers,
        mask,
        metric_name=metric_name,
        predicate_desc=predicate_desc,
    )


__all__ = [
    "PredicateDescription",
    "threshold",
    "top_k",
    "percentile_cutoff",
    "filter_by_mask",
    "filter_by_metric",
]

