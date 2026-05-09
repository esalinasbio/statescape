from __future__ import annotations

import tempfile
import os
import mdtraj as md
from pathlib import Path


def _fix_frame(
    frame: md.Trajectory,
    ph: float,
    *,
    fix_missing_residues: bool = True,
    fix_missing_atoms: bool = True,
    replace_nonstandard: bool = True,
    remove_heterogens: bool = True,
    keep_water: bool = True
) -> md.Trajectory:
    """Run PDBFixer on a single frame and return a fixed md.Trajectory."""

    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as e:
        raise ImportError("pdbfixer is required. Please install with `pip install pdbfixer`.") from e

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp_in:
        frame.save_pdb(tmp_in.name)
        tmp_in_path = tmp_in.name

    fixer = PDBFixer(filename=tmp_in_path)
    if fix_missing_residues:
        fixer.findMissingResidues()
    if replace_nonstandard:
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
    if remove_heterogens:
        fixer.removeHeterogens(keepWater=keep_water)
    if fix_missing_atoms:
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp_out:
        PDBFile.writeFile(fixer.topology, fixer.positions, tmp_out)
        tmp_out_path = tmp_out.name

    try:
        fixed = md.load_pdb(tmp_out_path)
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
        Path(tmp_out_path).unlink(missing_ok=True)

    return fixed

def add_hydrogens(
    traj: md.Trajectory,
    *,
    ph: float = 7.0,
    fix_missing_residues: bool = True,
    fix_missing_atoms: bool = True,
    replace_nonstandard: bool = True,
    remove_heterogens: bool = True,
    keep_water: bool = True,
) -> md.Trajectory:
    """
    Add hydrogens to each frame using PDBFixer.

    Parameters
    ----------
    ph : pH for protonation state assignment (default 7.0)
    fix_missing_residues : find and add missing residues from SEQRES records
    fix_missing_atoms : find and add missing heavy atoms
    replace_nonstandard : replace nonstandard residues with standard equivalents
    remove_heterogens : remove non-protein molecules
    keep_water : when remove_heterogens=True, keep water molecules
    """

    frames = [_fix_frame(
        traj[i], ph,
        fix_missing_residues=fix_missing_residues,
        fix_missing_atoms=fix_missing_atoms,
        replace_nonstandard=replace_nonstandard,
        remove_heterogens=remove_heterogens,
        keep_water=keep_water) for i in range(traj.n_frames)]

    return md.join(frames)

def add_hydrogens_missmatch(
    traj: md.Trajectory,
    *,
    ph: float = 7.0,
    fix_missing_residues: bool = True,
    fix_missing_atoms: bool = True,
    replace_nonstandard: bool = True,
    remove_heterogens: bool = True,
    keep_water: bool = True,
) -> md.Trajectory:
    """
    Add hydrogens to each frame and save to disk.
    Returns list of written PDB paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i in range(traj.n_frames):
        fixed = _fix_frame(traj[i], ph, ...)
        path = output_dir / f"frame_{i:05d}.pdb"
        fixed.save_pdb(str(path))
        paths.append(path)
    
    return paths