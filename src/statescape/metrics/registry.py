"""
Simple registry for metrics functions.

Metrics are registered under stable string names so that workflows
can refer to them symbolically.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np

MetricFn = Callable[..., np.ndarray]

_REGISTRY: Dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """
    Decorator to register a metric function under a stable name.
    """

    def decorator(fn: MetricFn) -> MetricFn:
        if name in _REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered.")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_metric(name: str) -> MetricFn:
    """
    Retrieve a registered metric function by name.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - simple error branch
        raise KeyError(f"Unknown metric '{name}'.") from exc


def list_metrics() -> Iterable[str]:
    """
    Return an iterable of registered metric names.
    """
    return _REGISTRY.keys()


__all__ = ["MetricFn", "register_metric", "get_metric", "list_metrics"]

