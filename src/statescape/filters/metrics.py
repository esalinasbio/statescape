# metrics.py
from __future__ import annotations
from typing import TYPE_CHECKING

import mdtraj as md
import numpy as np
from pathlib import Path

from statescape.util import get_sequence, validate_selection

if TYPE_CHECKING:
    from statescape.core.conformer import ConformerSet

def rmsd(
    traj: md.Trajectory | ConformerSet,
    ref: md.Trajectory | str | Path | None = None,
    *, 
    selection: str = "backbone"
) -> np.ndarray:
    """
    Compute RMSD (in \u00C5) for each frame in traj against reference.
    Defaults to backbone atoms. 
    If a ConformerSet is passed, its reference will be used if `ref` is not declared.
    """
    if type(traj).__name__ == "ConformerSet":
        if ref is None:
            ref = traj.reference
        traj = traj.trajectory
    if isinstance(ref, (str, Path)):
        ref = md.load(str(ref))
    sel = validate_selection(traj.topology, selection)
    return md.rmsd(traj, ref, atom_indices=sel)*10

def tmscore(
    traj: md.Trajectory | ConformerSet,
    ref: md.Trajectory | str | Path | None = None
) -> np.ndarray:
    """
    Compute TM-score for each frame in traj against reference.
    If a ConformerSet is passed, its reference will be used if `ref` is not declared.
    """
    try:
        from tmtools import tm_align
    except ImportError as e:
        raise ImportError("tmtools is required. Install with `pip install tmtools`.") from e

    if type(traj).__name__ == "ConformerSet":
        if ref is None:
            ref = traj.reference
        traj = traj.trajectory
    if isinstance(ref, (str, Path)):
        ref = md.load(str(ref))

    # reference
    ref_ca = validate_selection(ref.topology, "name CA")
    ref_coords = ref.xyz[0, ref_ca, :] # (N, 3)
    ref_seq = get_sequence(ref)

    scores =[]
    for frame in traj:
        frame_ca = validate_selection(frame.topology, "name CA")
        frame_coords = frame.xyz[0, frame_ca, :]
        frame_seq = get_sequence(frame)
        result = tm_align(ref_coords, frame_coords, ref_seq, frame_seq)
        scores.append(result.tm_norm_chain1)

    return np.array(scores)