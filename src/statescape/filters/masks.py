from __future__ import annotations
from typing import TYPE_CHECKING

import mdtraj as md
import numpy as np
from pathlib import Path

from . import metrics

if TYPE_CHECKING:
    from statescape.core.conformer import ConformerSet

__all__ = ["by_rmsd", "by_tmscore", "by_plddt", "by_peptide_bonds", "by_clashes", 'by_peptide_bond_stats']

def by_rmsd(
    traj: md.Trajectory | ConformerSet,
    ref: md.Trajectory | str | Path | None = None,
    *,
    cutoff: float,
    selection: str = "backbone"
) -> np.ndarray:
    """True when RMSD to `ref` is <= `cutoff` Angstom"""
    return metrics.rmsd(traj, ref, selection=selection) <= cutoff

def by_tmscore(
    traj: md.Trajectory | ConformerSet,
    ref: md.Trajectory | str | Path | None = None,
    *,
    cutoff: float
) -> np.ndarray:
    """True when TM-score against `ref` is  >= `cutoff`"""
    return metrics.tmscore(traj, ref) >= cutoff

def by_clashes(
    traj: md.Trajectory | ConformerSet,
    *, 
    cutoff: float = 1.1, 
    selection: str = "not element H"
) -> np.ndarray:
    """True where no non-bonded pair is closer than `cutoff` Angstrom"""
    return metrics.clashes(traj, cutoff=cutoff, selection=selection) == 0 

def by_plddt(source, * , cutoff: float = 70.0) -> np.ndarray:
    """
    True when mean pLDDT is >= `cutoff`
    
    Reads B-factor from the source PDB files, so returned mask is one
    entry per file, not per frame. Apply this before any other filter:
    a derived ConformerSet has no source files and will raise.
    """
    return metrics.plddt(source) >= cutoff

def by_peptide_bonds(
    traj: md.Trajectory | ConformerSet,
    *, 
    lo: float = 1.2, hi: float = 1.6
) -> np.ndarray:
    """True when every C-N peptide bond lenght lies in [lo, hi] Angstrom"""
    d = metrics.peptide_bonds(traj)
    return ((d >= lo) & (d <= hi)).all(axis=1)

def by_peptide_bond_stats(
    traj: md.Trajectory | ConformerSet,
    *, 
    mean_cutoff: float = 1.4, 
    sd_cutoff: float = 0.2
) -> np.ndarray:
    """
    True where the C-N peptide bond lenght distribution is well behaved.
    Same behaviour as `af2rave` peptide bond filter

    A frame is rejected (False) when its mean bond length is larger than `mean_cutoff`,
    or the standart deviation across bonds exceeds `sd_cutoff`, both in Angstrom
    """
    d = metrics.peptide_bonds(traj)
    return (d.mean(axis=1) <= mean_cutoff) & (d.std(axis=1) <= sd_cutoff)