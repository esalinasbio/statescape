"""
Trajectory-related helpers and metrics.

This module is part of the metrics layer and is allowed to depend on
external scientific libraries such as mdtraj.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

from statescape.core.conformer import ConformerSet


def count_trajectory_frames_mdtraj(trajectory: str | Path, topology: str | Path) -> int:
    """
    Return the number of frames in a trajectory using mdtraj.

    Parameters
    ----------
    trajectory:
        Path to the trajectory file (e.g. XTC, DCD, TRR).
    topology:
        Path to the topology file compatible with the trajectory
        (e.g. PDB, GRO, PSF).

    Notes
    -----
    This function lives in the metrics layer to keep the core data
    structures free of dependencies on external scientific libraries.
    """
    try:
        import mdtraj as md  # type: ignore
    except ImportError as e:  # pragma: no cover - import error path
        msg = (
            "mdtraj is required to read trajectories. "
            "Install it with `pip install mdtraj`."
        )
        raise ImportError(msg) from e

    traj_path = Path(trajectory)
    topo_path = Path(topology)

    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    if not topo_path.exists():
        raise FileNotFoundError(topo_path)

    n_frames = 0
    for chunk in md.iterload(str(traj_path), top=str(topo_path)):
        n_frames += chunk.n_frames
    return n_frames


def conformerset_from_xtc_mdtraj(
    trajectory: str | Path,
    topology: str | Path,
    *,
    stride: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> ConformerSet:
    """
    Convenience helper to build a ConformerSet from an XTC trajectory
    using mdtraj to determine the number of frames.

    This keeps the mdtraj dependency in the metrics layer while still
    providing a single-call API for trajectory-backed conformer sets.
    """
    n_frames = count_trajectory_frames_mdtraj(trajectory, topology)
    meta = dict(metadata or {})
    meta.setdefault("n_frames_total", n_frames)
    return ConformerSet.from_trajectory(
        trajectory=trajectory,
        topology=topology,
        n_frames=n_frames,
        stride=stride,
        metadata=meta,
    )

