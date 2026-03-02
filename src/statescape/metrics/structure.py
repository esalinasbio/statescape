"""
Structure-based metrics implemented on top of mdtraj and tmtools.

All functions here operate on primitive inputs (paths, coordinate arrays)
and return NumPy arrays of metric values, one per conformer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .registry import register_metric


@register_metric("rmsd_mdtraj_pdbs")
def rmsd_mdtraj_pdbs(
    pdb_files: Sequence[str | Path],
    *,
    ref_index: int = 0,
    selection: str | None = None,
) -> np.ndarray:
    """
    Compute RMSD (Å) of a set of PDB structures to a reference using mdtraj.

    Parameters
    ----------
    pdb_files:
        Sequence of PDB file paths. They must be structurally compatible.
    ref_index:
        Index of the reference structure within `pdb_files` (default: 0).
    selection:
        Optional mdtraj atom selection string. If None, all atoms are used.
    """
    if not pdb_files:
        raise ValueError("`pdb_files` must not be empty.")

    try:
        import mdtraj as md  # type: ignore
    except ImportError as e:  # pragma: no cover - import error branch
        raise ImportError("mdtraj is required for RMSD metrics.") from e

    paths = [str(Path(p)) for p in pdb_files]
    traj = md.load(paths)

    if selection is not None:
        atom_indices = traj.topology.select(selection)
    else:
        atom_indices = None

    if ref_index < 0 or ref_index >= traj.n_frames:
        raise IndexError(f"ref_index {ref_index} out of range for {traj.n_frames} frames.")

    ref = traj[ref_index]
    rmsd = md.rmsd(traj, ref, atom_indices=atom_indices)
    return rmsd.astype(np.float64)


@register_metric("rmsd_mdtraj_xtc")
def rmsd_mdtraj_xtc(
    trajectory: str | Path,
    topology: str | Path,
    *,
    ref_frame: int = 0,
    selection: str | None = None,
) -> np.ndarray:
    """
    Compute RMSD (Å) of frames in a trajectory to a reference frame using mdtraj.

    Parameters
    ----------
    trajectory:
        Path to the trajectory file (e.g. XTC).
    topology:
        Path to the topology file compatible with the trajectory (e.g. PDB).
    ref_frame:
        Index of the reference frame within the trajectory (default: 0).
    selection:
        Optional mdtraj atom selection string. If None, all atoms are used.
    """
    try:
        import mdtraj as md  # type: ignore
    except ImportError as e:  # pragma: no cover - import error branch
        raise ImportError("mdtraj is required for RMSD metrics.") from e

    traj_path = Path(trajectory)
    topo_path = Path(topology)
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    if not topo_path.exists():
        raise FileNotFoundError(topo_path)

    traj = md.load(str(traj_path), top=str(topo_path))

    if selection is not None:
        atom_indices = traj.topology.select(selection)
    else:
        atom_indices = None

    if ref_frame < 0 or ref_frame >= traj.n_frames:
        raise IndexError(f"ref_frame {ref_frame} out of range for {traj.n_frames} frames.")

    ref = traj[ref_frame]
    rmsd = md.rmsd(traj, ref, atom_indices=atom_indices)
    return rmsd.astype(np.float64)


@register_metric("tm_score_tmtools_pdbs")
def tm_score_tmtools_pdbs(
    pdb_files: Sequence[str | Path],
    *,
    ref_index: int = 0,
) -> np.ndarray:
    """
    Compute TM-score of a set of PDB structures relative to a reference using tmtools.

    This function loads each PDB, extracts residue-level coordinates and sequences
    via tmtools.io, and runs TM-align once per model against the reference.

    Parameters
    ----------
    pdb_files:
        Sequence of PDB file paths.
    ref_index:
        Index of the reference structure within `pdb_files` (default: 0).
    """
    if not pdb_files:
        raise ValueError("`pdb_files` must not be empty.")

    try:
        from tmtools import tm_align  # type: ignore
        from tmtools.io import get_residue_data, get_structure  # type: ignore
    except ImportError as e:  # pragma: no cover - import error branch
        msg = (
            "tmtools (and its Biopython-based IO helpers) are required for TM-score metrics. "
            "Install them with `pip install tmtools biopython`."
        )
        raise ImportError(msg) from e

    paths = [Path(p) for p in pdb_files]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    # Load reference structure
    ref_path = paths[ref_index]
    ref_struct = get_structure(str(ref_path))
    ref_chain = next(ref_struct.get_chains())
    ref_coords, ref_seq = get_residue_data(ref_chain)

    tm_scores: list[float] = []
    for idx, path in enumerate(paths):
        struct = get_structure(str(path))
        chain = next(struct.get_chains())
        coords, seq = get_residue_data(chain)

        res = tm_align(ref_coords, coords, ref_seq, seq)
        # Use TM-score normalized by reference length
        tm_scores.append(float(res.tm_norm_chain1))

    return np.asarray(tm_scores, dtype=np.float64)


@register_metric("peptide_bond_length_deviation")
def peptide_bond_length_deviation(
    trajectory: str | Path,
    topology: str | Path,
    *,
    ideal_length: float = 1.33,
) -> np.ndarray:
    """
    Compute mean absolute deviation of peptide bond lengths from an ideal value.

    For each frame, this metric:
    - identifies peptide bonds as C(i) - N(i+1) pairs across residues
    - computes their distances
    - returns the mean |d - ideal_length| over all such bonds
    """
    try:
        import mdtraj as md  # type: ignore
    except ImportError as e:  # pragma: no cover - import error branch
        raise ImportError("mdtraj is required for peptide bond metrics.") from e

    traj_path = Path(trajectory)
    topo_path = Path(topology)
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    if not topo_path.exists():
        raise FileNotFoundError(topo_path)

    traj = md.load(str(traj_path), top=str(topo_path))
    top = traj.topology

    peptide_pairs = []
    for chain in top.chains:
        residues = list(chain.residues)
        for i in range(len(residues) - 1):
            res_i = residues[i]
            res_j = residues[i + 1]
            c_atom = next((a for a in res_i.atoms if a.name == "C"), None)
            n_atom = next((a for a in res_j.atoms if a.name == "N"), None)
            if c_atom is not None and n_atom is not None:
                peptide_pairs.append((c_atom.index, n_atom.index))

    if not peptide_pairs:
        raise ValueError("No peptide bonds found in topology.")

    pair_indices = np.asarray(peptide_pairs, dtype=int)
    dists = md.compute_distances(traj, pair_indices)  # shape: (n_frames, n_bonds)
    deviations = np.abs(dists - ideal_length)
    return deviations.mean(axis=1).astype(np.float64)


@register_metric("steric_clash_heavy_atoms")
def steric_clash_heavy_atoms(
    trajectory: str | Path,
    topology: str | Path,
    *,
    cutoff: float = 1.0,
) -> np.ndarray:
    """
    Count steric clashes between heavy atoms for each frame.

    A clash is counted when the distance between two heavy atoms is below `cutoff`.
    This uses mdtraj's neighbor search over all frames for performance.
    """
    try:
        import mdtraj as md  # type: ignore
    except ImportError as e:  # pragma: no cover - import error branch
        raise ImportError("mdtraj is required for steric clash metrics.") from e

    traj_path = Path(trajectory)
    topo_path = Path(topology)
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)
    if not topo_path.exists():
        raise FileNotFoundError(topo_path)

    traj = md.load(str(traj_path), top=str(topo_path))
    top = traj.topology

    heavy_indices = np.array(
        [atom.index for atom in top.atoms if atom.element.symbol != "H"],
        dtype=int,
    )
    if heavy_indices.size == 0:
        raise ValueError("No heavy atoms found in topology.")

    # Neighbor search across all frames; heavy lifting is inside mdtraj.
    neighbors = md.compute_neighbors(
        traj,
        cutoff=cutoff,
        query_indices=heavy_indices,
        haystack_indices=heavy_indices,
    )

    # neighbors is a list of arrays (one per frame), each containing indices
    # of neighbors for the query atoms. We count total neighbor relationships.
    clash_counts = np.fromiter((len(arr) for arr in neighbors), dtype=int, count=len(neighbors))
    return clash_counts.astype(np.int64)


__all__ = [
    "rmsd_mdtraj_pdbs",
    "rmsd_mdtraj_xtc",
    "tm_score_tmtools_pdbs",
    "peptide_bond_length_deviation",
    "steric_clash_heavy_atoms",
]

