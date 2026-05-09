import mdtraj as md
import numpy as np
from . import metrics

def by_rmsd(
    traj: md.Trajectory,
    ref: md.Trajectory,
    cutoff: float,
    *,
    selection: str = "backbone"
) -> np.ndarray:
    """Returns a boolean mask where RMSD <= cutoff"""
    return metrics.rmsd(traj, ref, selection=selection) <= cutoff

def by_tmscore(
    traj: md.Trajectory,
    ref: md.Trajectory,
    cutoff: float
) -> np.ndarray:
    """Returns a boolean mask where TM-score >= cutoff"""
    return metrics.tmscore(traj, ref) >= cutoff