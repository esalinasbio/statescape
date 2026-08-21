from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mdtraj as md
import numpy as np

from statescape._vendor.amino._colvar import Colvar

if TYPE_CHECKING:
    from statescape.core.conformer import ConformerSet

def as_trajectory(obj: md.Trajectory | ConformerSet) -> md.Trajectory:
    """
    Return the underlying trajecotry of a ConformerSet, or `obj` unchanged
    """ 
    return getattr(obj, "trajectory", obj)

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

def get_sequence(traj: md.Trajectory | ConformerSet) -> str:
    """
    One letter sequence of protein residues in `traj`
    Non protein residues are skipped, unknown protein residues are written as X
    """
    top = as_trajectory(traj).topology
    seq = [res.code or "X" for res in top.residues if res.is_protein]
    return "".join(seq)

def load_colvar(path: str | Path) -> np.ndarray:
    """Reads a PLUMED COLVAR file as an (n_frames, n_features) array."""
    return Colvar.from_file(str(path)).data.T # colvar stores (n_features, n_frames)

def load_feature_matrix(path: str | Path, *, format: str) -> np.ndarray:
    """Load a feature file as an (n_frames, n_features) array"""
    path = Path(path)
    if format == 'npy':
        return np.load(path)
    if format == 'colvar':
        return load_colvar(path)
    raise ValueError(f"Unknown feature format {format!r} for {path}")

def save_pdb(
    traj: md.Trajectory | ConformerSet,
    path
):
    """Save a single-frame trajectory as a PDB file. Returns the written path."""
    traj = as_trajectory(traj)
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