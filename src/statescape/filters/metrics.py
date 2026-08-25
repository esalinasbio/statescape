# metrics.py
from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import mdtraj as md
import numpy as np
from pathlib import Path
from natsort import natsorted

from statescape.util import as_trajectory, get_sequence, validate_selection

if TYPE_CHECKING:
    from statescape.core.conformer import ConformerSet


def _unpack(
    traj:md.Trajectory | ConformerSet,
    ref: md.Trajectory | str | Path | None
) -> tuple[md.Trajectory, md.Trajectory]:
    """Resolve (trajectory, reference), defaults to its own reference if `traj` is a ConformerSet"""
    if hasattr(traj, "trajectory"):
        if ref is None:
            ref = traj.reference
        traj = traj.trajectory
    if ref is None:
        raise ValueError("Missing reference. `traj` is not a ConformerSet")
    if isinstance(ref, (str, Path)):
        ref= md.load(str(ref))
    return traj, ref

def rmsd(
    traj: md.Trajectory | ConformerSet,
    ref: md.Trajectory | str | Path | None = None,
    *, 
    selection: str = "backbone"
) -> np.ndarray:
    """
    RMSD in \u00C5 for each frame in `traj` against `ref`, over `selection` (default backbone) 
    If a ConformerSet is passed, its reference is used if `ref` is None.
    """
    traj, ref = _unpack(traj, ref)
    sel = validate_selection(traj.topology, selection)
    return md.rmsd(traj, ref, atom_indices=sel)*10

def tmscore(
    traj: md.Trajectory | ConformerSet,
    ref: md.Trajectory | str | Path | None = None
) -> np.ndarray:
    """
    TM-score for each frame in `traj` against `ref`.
    If a ConformerSet is passed, its reference is used if `ref` is None.
    """
    try:
        from tmtools import tm_align
    except ImportError as e:
        raise ImportError("tmtools is required. Install with `pip install tmtools`.") from e

    traj, ref = _unpack(traj, ref)

    # reference
    ref_ca = validate_selection(ref.topology, "protein and name CA")
    ref_coords = ref.xyz[0, ref_ca, :] # (N, 3)
    ref_seq = get_sequence(ref)

    # frame
    frame_ca = validate_selection(traj.topology, "protein and name CA")
    frame_seq = get_sequence(traj)
    if len(frame_seq) != frame_ca.size:
        raise ValueError(f"Got {frame_ca.size} CA atoms but {len(frame_seq)} residues.")

    scores = np.empty(traj.n_frames)
    for i, xyz in enumerate(traj.xyz):
        scores[i] = tm_align(ref_coords, xyz[frame_ca, :], ref_seq, frame_seq).tm_norm_chain1
    return scores


def peptide_bonds(traj: md.Trajectory | ConformerSet) -> np.ndarray:
    """
    Peptide bond lenghts of C(i)-N(i+1) in \u00C5, shape (n_frames, n_bonds)
    Only consecutive residues within the same chain are considered
    """
    traj = as_trajectory(traj)
    top = traj.topology

    # Find C_i - N_i+1 pairs
    pairs =[]
    for chain in top.chains:
        residues = [r for r in chain.residues if r.is_protein]
        for a,b in zip(residues, residues[1:]):
            if b.resSeq - a.resSeq != 1:
                continue
            C = next((i.index for i in a.atoms if i.name == 'C'), None)
            N = next((i.index for i in b.atoms if i.name == 'N'), None)
            if C is not None and N is not None:
                pairs.append((C,N))

    if not pairs:
        raise ValueError("Cannot find consecutive protein residues with C/N atoms")

    return md.compute_distances(traj, np.array(pairs)) * 10 # in Angstroms

def clashes(
        traj: md.Trajectory | ConformerSet,
        *,
        cutoff: float = 1.1,
        selection: str = "not element H"
) -> np.ndarray:
    """
    Number of non-bonded heavy atom pairs closer than `cutoff` \u00C5

    Parameters
    -----------
    cutoff: clash distance in Angstrom (default 1.1)
    selection: atom selection (default: all heavy atoms)
    """
    from scipy.spatial import cKDTree

    traj = as_trajectory(traj)
    idx = validate_selection(traj.topology, selection)
    bonded = {frozenset((i[0].index, i[1].index)) for i in traj.topology.bonds}

    counts = np.empty(traj.n_frames, dtype=int)
    for i, xyz in enumerate(traj.xyz):
        pairs = cKDTree(xyz[idx]).query_pairs(cutoff / 10.0, output_type='ndarray')
        counts[i] = sum(1 for a,b in pairs if frozenset((int(idx[a]), int(idx[b]))) not in bonded)
    return counts

def _resolve_paths(source: ConformerSet | str | Path | Sequence[str, Path]) -> list[Path]:
    """
    Resolve a ConformerSet, directory, file or sequence of files to a natsorted path list
    used to extract pLDDT from B-factor column direclty from the files
    """
    if hasattr(source, 'sources'):
        paths = source.sources
        if paths is None:
            raise ValueError("ConformerSet was built in memory and has no source files.")
        return [Path(p) for p in paths]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            files = natsorted(path.glob('*.pdb'))
            if not files:
                raise FileNotFoundError(f'No PDB files in {path}')
            return [Path(f) for f in files]
        return [path]
    return [Path(p) for p in source]

def plddt(source: ConformerSet | str | Path | Sequence[str | Path]) -> np.ndarray:
    """
    Mean pLDDT per model, read from the b-factor column of PDB files. Only works with AF models
    Accepts a ConformerSet, directory of PDBs, or an explicit file list. Only one model per PDB files

    Notes
    ------
    MDTraj discards b-factor column, so we need to read the files directly. 
    The returned order matches `ConformerSet.names` only when it was build from the same directory/file list.
    """
    paths = _resolve_paths(source)
    scores = np.zeros(len(paths), dtype=float)

    for i, path in enumerate(paths):
        current = []
        for line in Path(path).read_text().splitlines():
            # Only look for CA atoms
            if line.startswith('ATOM') and line[12:16].strip() == "CA":
                current.append(float(line[60:66]))
        if not current:
            raise ValueError(f"No data in CA B-factor columns found in {path}")
        scores[i] = np.mean(current)
    return scores