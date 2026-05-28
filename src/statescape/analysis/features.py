import numpy as np
import mdtraj as md

from statescape.util import validate_selection

def ca_coordinates(
    traj: md.Trajectory,
    selection: str = "name CA"
) -> tuple[np.ndarray, list[str]]:
    """
    Extracts C-alpha coordinates as a flat feature matrix (n_frames, n_CA * 3) in Angstrom.
    Superposition onto frame 0 is applied automatically.
    If `selection` is declared, only the C-alpha distances within `selection` will be computed.
    Returns (features, labels) where labels are like 'x_CA_5', 'y_CA_5', 'z_CA_5'
    """
    if selection != "name CA":
        ca_indices = validate_selection(traj.topology, f"({selection}) and name CA")
    else:
        ca_indices = validate_selection(traj.topology, selection)
    traj.superpose(traj, 0, atom_indices=ca_indices)
    coords = traj.atom_slice(ca_indices).xyz * 10 # nm to Angstrom

    labels = []
    for i in ca_indices:
        resid = traj.topology.atom(i).residue.resSeq
        labels.extend([f"x_CA_{resid}", f"y_CA_{resid}", f"z_CA_{resid}"])

    return coords.reshape(coords.shape[0], -1), labels

def ca_distances(
    traj: md.Trajectory,
    selection: str = "name CA"
) -> tuple[np.ndarray, list[str]]:
    """
    Pairwise C-alpha distances as a feature matrix (n_frames, n_pairs) in Angstrom.
    Computes all C-alpha distances by default. Returns (features, labels).
    If `selection` is declared, only the c-alpha distances within `selection` will be computed.
    """
    if selection != "name CA":
        ca_indices = validate_selection(traj.topology, f"({selection}) and name CA")
    else:
        ca_indices = validate_selection(traj.topology, selection)
    pairs = np.array([(i, j) for idx, i in enumerate(ca_indices) for j in ca_indices[idx + 1:]]) # (num_pairs, 2)
    distances, _ = md.compute_distances(traj, pairs)

    top = traj.topology
    labels = [f'CA_{top.atom(i).residue.resSeq}-CA_{top.atom(j).residue.resSeq}' for i,j in pairs]

    return distances * 10, labels

def backbone_dihedrals(
    traj: md.Trajectory,
    selection: str = 'all',
    sincos: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Backbone phi/psi angles. Returns (features, labels).
    Optional: Use sin/cos to handle periodicity (default = True)
    """
    if selection != 'all':
        idx = traj.topology.select(selection)
        traj = traj.atom_slice(idx)

    phi_idx, phi = md.compute_phi(traj)
    psi_idx, psi = md.compute_psi(traj)
    angles = np.concatenate([phi, psi], axis=1)

    labels = (_dih_labels("phi", phi_idx, traj.topology) + _dih_labels('psi', psi_idx, traj.topology))

    if sincos:
        angles = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
        labels = [f"sin_{l}" for l in labels] + [f"cos_{l}" for l in labels]

    return angles, labels

def sidechain_dihedrals(
    traj: md.Trajectory,
    selection: str = 'all',
    sincos: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Sidechain chi1/chi2 angles. Returns (features, lables).
    Optional: Use sin/cos to handle periodicity (default = True)
    """
    if selection != 'all':
        idx = traj.topology.select(selection)
        traj = traj.atom_slice(idx)

    chi1_idx, chi1 = md.compute_chi1(traj)
    chi2_idx, chi2 = md.compute_chi2(traj)
    angles = np.concatenate([chi1, chi2], axis=1)

    labels = (_dih_labels("chi1", chi1_idx, traj.topology) + _dih_labels('chi2', chi2_idx, traj.topology))

    if sincos:
        angles = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
        labels = [f"sin_{l}" for l in labels] + [f"cos_{l}" for l in labels]

    return angles, labels

def all_dihedrals(
    traj: md.Trajectory,
    selection: str = 'all',
    sincos: bool = True
) -> np.ndarray:
    """
    All backbone (phi/psi) and sidechain (chi1/chi2) dihedrals. Returns (features, labels).
    Optional: Use sin/cos to handle periodicity (default = True)
    """
    if selection != 'all':
        idx = traj.topology.select(selection)
        traj = traj.atom_slice(idx)

    dih = []
    labels = []
    for type, func in [("phi", md.compute_phi), ("psi", md.compute_psi), ("chi1", md.compute_chi1), ("chi2", md.compute_chi2)]:
        atom_idx, value = func(traj)
        dih.append(value)
        labels.extend(_dih_labels(type, atom_idx, traj.topology))

    angles = np.concatenate(dih, axis=1)

    if sincos:
        angles = np.concatenate([np.sin(angles), np.cos(angles)], axis=1)
        labels = [f"sin_{l}" for l in labels] + [f"cos_{l}" for l in labels]

    return angles, labels

def custom(
    traj: md.Trajectory,
    selection: str,
) -> tuple[np.ndarray, list[str]]:
    """Coordinates of a custom atom selection. Returns (features, labels)."""
    indices = validate_selection(traj.topology, selection)
    traj.superpose(traj, 0, atom_indices=indices)
    coords = traj.atom_slice(indices).xyz * 10
    
    labels = []
    for i in indices:
        atom = traj.topology.atom(i)
        resid = atom.residue.resSeq
        labels.extend([f"x_{atom.name}_{resid}", f"y_{atom.name}_{resid}", f"z_{atom.name}_{resid}"])
    
    return coords.reshape(coords.shape[0], -1), labels

def featurize(
    traj: md.Trajectory,
    method: str = "all_dihedrals",
    **kwargs
) -> tuple[np.ndarray, list[str]]:
    """
    Extract a feature matrix from an `md.Trajectory`.
    Returns (features, labels) where features has shape (n_frames, n_features).

    Methods
    -------
    'ca_coordinates'      : flat C-alpha xyz coordinates (default)
    'ca_distances'        : pairwise C-alpha distances
    'backbone_dihedrals'  : backbone phi/psi angles (sin/cos transformed by default)
    'sidechain_dihedrals' : side chain dihedrals chi1/chi2 (sin/cos transformed by default)
    'all_dihedrals'       : all psi/phi/chi1/chi2 dihedrals (sin/cos transformed by default)
    'custom'              : Pairwise distanes in `selection`. Requires `selection` kwarg
    """
    feature_map = {
        "ca_coordinates"      : ca_coordinates,
        "ca_distances"        : ca_distances,
        "backbone_dihedrals"  : backbone_dihedrals,
        "sidechain_dihedrals" : sidechain_dihedrals,
        "all_dihedrals"       : all_dihedrals,
        "custom"              : custom,
    }
    if method == "custom" and "selection" not in kwargs:
        raise ValueError("'custom' method requires a 'selection' argument.")
    if method not in feature_map:
        raise ValueError(f"Unknown feature method: {method!r}. Available: {list(feature_map.keys())}")
    return feature_map[method](traj, **kwargs)

def _dih_labels(name: str, atom_idx: np.ndarray, topology) -> list[str]:
    """Build labels for dihedral angles like `phi_5` (phi angle of residue 5)"""
    return [f"{name}_{topology.atom(i[1]).residue.resSeq}" for i in atom_idx]