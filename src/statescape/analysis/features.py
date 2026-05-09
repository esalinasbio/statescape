import select
import numpy as np
import mdtraj as md

from statescape.util import validate_selection

def ca_coordinates(
    traj: md.Trajectory,
    selection: str = "name CA"
) -> np.ndarray:
    """
    Extracts C-alpha coordinates as a flat feature matrix (n_frames, n_CA * 3) in Angstrom.
    Superposition onto frame 0 is applied automatically.
    If `selection` is declared, only the C-alpha distances within `selection` will be computed.
    """
    if selection != "name CA":
        ca_indices = validate_selection(traj.topology, f"({selection}) and name CA")
    else:
        ca_indices = validate_selection(traj.topology, selection)
    traj.superpose(traj, 0, atom_indices=ca_indices)
    coords = traj.atom_slice(ca_indices).xyz * 10 # nm to Angstrom
    return coords.reshape(coords.shape[0], -1)

def ca_distances(
    traj: md.Trajectory,
    selection: str = "name CA"
) -> np.ndarray:
    """
    Pairwise C-alpha distances as a feature matrix (n_frames, n_pairs) in Angstrom.
    Computes all C-alpha distances by default. 
    If `selection` is declared, only the c-alpha distances within `selection` will be computed.
    """
    if selection != "name CA":
        ca_indices = validate_selection(traj.topology, f"({selection}) and name CA")
    else:
        ca_indices = validate_selection(traj.topology, selection)
    pairs = np.array([(i, j) for idx, i in enumerate(ca_indices) for j in ca_indices[idx + 1:]]) # (num_pairs, 2)
    distances = md.compute_distances(traj, pairs)
    return distances * 10 # nm to Angstrom

def backbone_dihedrals(
    traj: md.Trajectory,
    selection: str = 'all',
    sincos: bool = True
) -> np.ndarray:
    """
    Backbone phi/psi angles as a feature matrix (n_frames, n_angles * 2).
    Optional: Use sin/cos to handle periodicity (default = True)
    """
    if selection != 'all':
        idx = traj.topology.select(selection)
        traj = traj.atom_slice(idx)

    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    angles = np.concatenate([phi, psi], axis=1)
    if sincos:
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return angles

def sidechain_dihedrals(
    traj: md.Trajectory,
    selection: str = 'all',
    sincos: bool = True
) -> np.ndarray:
    """
    Sidechain chi1/chi2 angles as a feature matrix (n_frames, n_angles * 2).
    Optional: Use sin/cos to handle periodicity (default = True)
    """
    if selection != 'all':
        idx = traj.topology.select(selection)
        traj = traj.atom_slice(idx)

    _, chi1 = md.compute_chi1(traj)
    _, chi2 = md.compute_chi2(traj)
    angles = np.concatenate([chi1, chi2], axis=1)
    if sincos:
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return angles

def all_dihedrals(
    traj: md.Trajectory,
    selection: str = 'all',
    sincos: bool = True
) -> np.ndarray:
    """
    NOT WORKING
    Sidechain chi1/chi2 angles as a feature matrix (n_frames, n_angles * 2).
    Optional: Use sin/cos to handle periodicity (default = True)
    """
    if selection != 'all':
        idx = traj.topology.select(selection)
        traj = traj.atom_slice(idx)

    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    _, chi1 = md.compute_chi1(traj)
    _, chi2 = md.compute_chi2(traj)
    angles = np.concatenate([phi, psi, chi1, chi2], axis=1)
    if sincos:
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
    return angles

def custom(
    traj: md.Trajectory,
    selection: str
) -> np.ndarray:
    """
    Coordinates of a custom atom selection as a flat feature matrix.
    """
    indices = validate_selection(traj.topology, selection)
    traj.superpose(traj, 0, atom_indices=indices)
    coords = traj.atom_slice(indices).xyz * 10.0
    return coords.reshape(coords.shape[0], -1)