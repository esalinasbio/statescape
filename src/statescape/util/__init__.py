from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import mdtraj as md
import numpy as np

if TYPE_CHECKING:
    from statescape.core.conformer import ConformerSet

def validate_selection(
    top: md.Topology,
    selection: str
) -> np.ndarray:
    """
    Select atoms from a topology and validate the result.
    Returns an array of atom indices.
    """
    try:
        idx = top.select(selection)
    except Exception as e:
        raise ValueError(f"Invalid atom selection '{selection}': {e}") from e

    if idx.size == 0:
        raise ValueError(f"Atom selection '{selection}' matched no atoms.")
    
    return idx

def get_sequence(
    traj: md.Trajectory | ConformerSet
):
    top = traj.topology
    seq = [res.code for res in top.residues]
    return "".join(seq)

def save_pdb(
    traj: md.Trajectory | ConformerSet,
    path
):
    """Save a single-frame trajectory as a PDB file. Returns the written path."""
    if type(traj).__name__ == "ConformerSet":
        traj = traj.trajectory
    if len(traj) != 1:
        raise ValueError(f"Expected a single frame, got {len(traj)}.")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traj.save_pdb(out)
    return out

def save_colvar(
    features: np.ndarray,
    labels: list[str],
    path: str | Path,
    *,
    time: np.ndarray | list | None = None
) -> Path:
    """
    Save feature array as a PLUMED COLVAR file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_frames = features.shape[0]
    if time is None:
        time = np.arange(n_frames, dtype=float)

    header = "#! FIELDS time " + " ".join(labels)
    data = np.column_stack([time, features])
    np.savetxt(out, data, header=header, comments="", fmt="%.6f")
    return out