"""
Metrics layer for Statescape.

This package provides thin adapters around external scientific libraries
such as mdtraj, RDKit, OpenMM, etc. Metric functions operate on primitive
data structures (arrays, paths, topologies) rather than ConformerSet objects.
"""

from . import trajectory  # noqa: F401

__all__ = ["trajectory"]

