from __future__ import annotations

import mdtraj as md
import numpy as np

from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Iterable, Sequence

from .preparation import add_hydrogens, _fix_frame
from statescape.util import save_pdb, save_colvar
from statescape.filters import masks
from statescape.analysis import features, dimensionality, clustering
from statescape.analysis.dimensionality import ReductionResult

class ConformerSet:
    """
    collection of multiple protein conformations
    """

    def __init__(
        self, 
        input: str | Path | Sequence[str | Path] | md.Trajectory | None = None,
        *,
        trajectory: str | md.Trajectory | None = None,
        topology: str | Path | None = None,
        reference: str | Path | md.Trajectory | None = None,
    ) -> None:

        # Manage input 

        if input is None and trajectory is None:
            raise (ValueError("Missing input. Provide either a PDB file or list of files, a folder containing PDB files, or a trajectory (.xtc, .nc, etc.) + topology files."))
        if input is not None and trajectory is not None:
            raise (ValueError("Provide only one input or trajectory."))

        #self._input = input if input is not None else trajectory
        self._input = input
        self._source = _verify_source(input)

        if trajectory is not None:
            if isinstance(trajectory, md.Trajectory):
                self._traj = trajectory
            elif topology is not None:
                self._traj = md.load(trajectory, top=topology)
            else:
                raise ValueError("A topology file is required when loading a trajectory file.")
            self._names = [f"frame_{i}" for i in range(self._traj.n_frames)]
        else:
            if topology:
                self._traj = md.load(self._source, top=topology)
            else:
                self._traj = md.load(self._source)
            
            if isinstance(self._source, list):
                self._names = [Path(s).stem for s in self._source]
            else:
                self._names = [f"{Path(self._source).stem}_{i}" for i in range(self._traj.n_frames)]
        self._name_index = {name: i for i, name in enumerate(self._names)}

        # reference and topology
        if reference is None:
            self._ref = self._traj[0]
            self._ref_source = self._names[0]
        elif isinstance(reference, md.Trajectory):
            if reference.n_frames == 0:
                raise ValueError("Reference trajectory is empty.")
            self._ref = reference[0]
            self._ref_source = reference
        else: 
            self._ref = md.load(reference)[0]
            self._ref_source = str(reference)
        
        if topology:
            self._top = self._traj.topology 
            self._top_file = topology
        else:
            self._top = self._ref.topology
            self._top_file = None

    # Class methods

    @classmethod
    def _from_traj(
        cls,
        traj: md.Trajectory,
        *,
        ref:  str | Path | md.Trajectory | None = None,
        names: Sequence[str] | None = None
    ) -> ConformerSet:
        obj = cls(trajectory=traj, reference=ref)
        if names is not None:
            if len(names) != traj.n_frames:
                raise ValueError(f"Got {len(names)} names for {traj.n_frames} frames")
            obj._names = list(names)
            obj._name_index = {n: i for i, n in enumerate(obj._names)}
        return obj

    @classmethod
    def merge(
        cls, 
        sets: Sequence[ConformerSet], 
        *, 
        reference: str | Path | md.Trajectory | None = None
    ) -> ConformerSet:
        """Concatenate multiple ConformerSets into one."""
        if not sets:
            raise ValueError("Provide at least 2 ConformerSets.")
        merged = md.join([s.trajectory for s in sets])
        names = [f"{i}_{n}" for i, s in enumerate(sets) for n in s.names]
        ref = reference if reference is not None else sets[0].reference
        return cls._from_traj(merged, ref=ref, names=names)

    # filtering and manipulation

    def subset(self, idx: Sequence[int]) -> ConformerSet:
        """Return a new set containing the requested frame indices."""
        idx = np.asarray(idx, dtype=int)
        return self._from_traj(self._traj[idx], ref=self._ref, names=[self._names[i] for i in idx])
    
    def filter(self, mask: Sequence[bool]) -> ConformerSet:
        """Return a new set containing frames where the mask is True."""
        keep = np.asarray(mask, dtype=bool)
        if keep.ndim != 1 or keep.shape[0] != len(self):
            raise ValueError(f"Mask has shape {keep.shape}, expected 1D of lenght {len(self)}")
        if not keep.any():
            raise ValueError(f"Mask removed all {len(self)} conformations, nothing left to filter")
        return self.subset(np.flatnonzero(keep))

    def split_by_mask(self, mask: Sequence[bool]) -> tuple[ConformerSet, ConformerSet]:
        """Split into `(kept, dropped)` using a boolean mask."""
        keep = np.asarray(mask, dtype=bool)
        return self.filter(keep), self.filter(~keep)

    def filter_rmsd(
        self,
        cutoff: float,
        *,
        selection: str = "backbone"
    ) -> ConformerSet:
        """Filter conformations with RMSD < `cutoff` to the reference. Defaults to backbone atoms."""
        mask = masks.by_rmsd(self._traj, self._ref, cutoff, selection=selection)
        return self.filter(mask)

    def filter_tmscore(
        self,
        cutoff: float
    ) -> ConformerSet:
        """Filter conformations with TM-score > `cutoff` to the reference."""
        mask = masks.by_tmscore(self._traj, self._ref, cutoff)
        return self.filter(mask)


    def compute_features(
        self,
        method: str = 'ca_coordinates',
        **kwargs,
    ) -> np.ndarray:
        """
        Extract a feature matrix from this ConformerSet.
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
        return features.featurize(self._traj, method, **kwargs)


    def dimensionality(
        self,
        method: str = 'pca',
        *,
        n_components: int = 10,
        features: np.ndarray | None = None,
        feature_method: str = "ca_coordinates",
        feature_kwargs: dict | None = None,
        **kwargs
    ) -> ReductionResult:
        """
        Dimensionality reduction over a feature matrix.
        Returns a ReductionResult with .coords (n_frames, n_components).

        Parameters
        ----------
        method : 'pca' | 'umap' # more will be coming
        n_components : number of output dimensions
        features : precomputed (n_frames, n_features) array. If None, compute_features(feature_method) is called automatically.
        feature_method : which feature extractor to use when features=None
        feature_kwargs : kwargs forwarded to the featurizer; **kwargs go to the reducer
        """

        dim_map = {
            "pca" : lambda: dimensionality.pca(features, n_components=n_components, **kwargs),
            "umap": lambda: dimensionality.umap(features, n_components=n_components, **kwargs)
        }

        if features is None:
            features, _ = self.compute_features(feature_method, **(feature_kwargs or {}))

        if method not in dim_map:
            raise ValueError(f"Unknown dimensionality reduction: {method!r}. Available: {list(dim_map.keys())}")
        return dim_map[method]()

    def cluster(
        self, 
        method: str = 'kmeans',
        *,
        n_clusters: int = 50,
        radius: float = 1.5,
        n_components: int = 2,
        latent_coords: np.ndarray | None = None,
        **kwargs
    ) -> tuple[np.ndarray, dict[int, int]]:
        """
        Cluster conformations in a latent space. Defaults to clustering in PCA space over C-alpha space.
        Returns (labels, {cluster_id: frame_index}).

        Parameters
        ----------
        method : 'kmeans' | 'gmm' | 'regular_space'
        n_clusters : number of clusters (kmeans and gmm only)
        radius : radius threshold (regular_space only)
        n_components : how many dimensions of latent_coords to use
        latent_coords : any (n_frames, n_features) array. If None, PCA over C-alpha coordinates is computed automatically.
        """
        cluster_map = {
            "kmeans": lambda: clustering.kmeans(latent_coords, n_clusters=n_clusters, n_components=n_components, **kwargs),
            "gmm": lambda: clustering.gmm(latent_coords, n_clusters=n_clusters, n_components=n_components),
            "regular_space": lambda: clustering.regular_space(latent_coords, radius, n_components=n_components)
        }

        if latent_coords is None:
            latent_coords = self.dimensionality().coords
        if len(latent_coords) != len(self._traj):
            raise ValueError(f'latent_coords has {len(latent_coords)} rows but ConformerSet has {len(self._traj)} frames.')
        if method not in cluster_map:
            raise ValueError(f"Unknown feature method: {method!r}. Available: {list(cluster_map.keys())}")

        return cluster_map[method]()


    def reduce(
        self,
        *,
        ph: float = 7.0,
        fix_missing_residues: bool = True,
        fix_missing_atoms: bool = True,
        replace_nonstandard: bool = True,
        remove_heterogens: bool = True,
        keep_water: bool = True,
    ) -> ConformerSet:
        """
        Add hydrogens to all conformations using PDBFixer.
        Returns a new ConformerSet with protonated structures.

        Call this before `save_cluster_representatives()` if protonated structures are needed as seeds.

        Parameters
        ----------
        ph : pH for protonation state assignment (default 7.0)
        fix_missing_residues : find and add missing residues from SEQRES records
        fix_missing_atoms : find and add missing heavy atoms
        replace_nonstandard : replace nonstandard residues with standard equivalents
        remove_heterogens : remove non-protein molecules
        keep_water : when remove_heterogens=True, keep water molecules
        """
        fixed = add_hydrogens(
            self._traj, ph=ph,
            fix_missing_residues=fix_missing_residues,
            fix_missing_atoms=fix_missing_atoms,
            replace_nonstandard=replace_nonstandard,
            remove_heterogens=remove_heterogens,
            keep_water=keep_water,
        )
        return self._from_traj(fixed, ref=self._ref, names=self._names)


    def save_cluster_representatives(
        self, 
        representatives: dict[int, int],
        output_dir: str | Path,
        *,
        prefix: str = "cluster",
        subfolders: bool = True,
        protonate: bool = False,
        ph: float = 7.0,
        fix_missing_residues: bool = False,
        fix_missing_atoms: bool = False,
        replace_nonstandard: bool = False,
        remove_heterogens: bool = False,
        keep_water: bool = True
    ) -> list[Path]:
        """
        Save one PDB per cluster representative into per-cluster subfolders. 
        Set `subfolders = False` to have all the clusters saved in the same directory.

        If `protonate = True`, hydrogen atoms are added at ph = `ph` (default = 7.0) before saving.

        `output_dir` structure if `subfolders = True`:

        output_dir/
        ├── cluster_00/
        │   └── <name>.pdb
        ├── cluster_01/
        │   └── <name>.pdb
        ...

        Returns the list of written paths.
        """
        output_dir = Path(output_dir)
        written = []
        for cid, frame_idx in representatives.items():
            subfolder = output_dir / f"{prefix}_{cid:02d}" if subfolders else output_dir
            subfolder.mkdir(parents=True, exist_ok=True)
            name = self._names[frame_idx]
            frame = self._traj[frame_idx]

            if protonate:
                frame = _fix_frame(
                    frame, ph,
                    fix_missing_residues=fix_missing_residues,
                    fix_missing_atoms=fix_missing_atoms,
                    replace_nonstandard=replace_nonstandard,
                    remove_heterogens=remove_heterogens,
                    keep_water=keep_water,
                )

            path = save_pdb(frame, subfolder / f"{name}.pdb")
            written.append(path)
        return written

    def save_features_colvar(
        self,
        path: str | Path,
        method: str = 'all_dihedrals',
        **kwargs
    ):
        """
        Compute features and save them as a PLUMED COLVAR file.
        By default, computes backbone and side chain dihedrals (phi, psi, chi1, chi2).
        """
        features, labels = self.compute_features(method, **kwargs)
        return save_colvar(features, labels, path)
        
    # properties

    @property
    def trajectory(self) -> md.Trajectory:
        """Return the loaded trajectory."""
        return self._traj

    @property
    def topology(self):
        """Return the resolved topology object."""
        return self._top

    @property
    def reference(self) -> md.Trajectory:
        """Return the single-frame reference trajectory."""
        return self._ref

    @property
    def names(self) -> list:
        """Return the list of conformer names, based on source type.""" 
        return self._names

    @property
    def reference_file(self):
        """The reference file."""
        return self._ref_source
    
    @property
    def topology_file(self):
        """the topology file"""
        return self._top_file

    def __len__(self) -> int:
        """Return the number of conformations inside ConformerSet."""
        return self._traj.n_frames

    def __iter__(self) -> Iterable[md.Trajectory]:
        for i in range(len(self)):
            yield self._traj[i]

    def __getitem__(self, key):
        try:
            return self._traj[self._name_index[key]]
        except KeyError:
            raise KeyError(f"Structure '{key}' not found.")

    def __repr__(self) -> str:
        return (
            f"ConformerSet("
            f"n_frames={self._traj.n_frames}, "
            f"n_atoms={self._traj.n_atoms}, "
            f"source={_source_repr(self._input)}"
            f")"
        )


def _verify_source(source) -> str | list[str] | None:
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            files = natsorted(glob(str(path / "*.pdb")))
            if not files:
                raise FileNotFoundError(f"No PDB files found in folder: {path}")
            return files
        return str(path)
    return [str(Path(s)) for s in source]

def _source_repr(source: str | Path | Sequence[str | Path] | None) -> str:
    """Build a concise source string for `ConformerSet.__repr__`."""
    if source is None:
        return "in-memory"
    if isinstance(source, (str, Path)):
        return str(source)
    return "[" + ", ".join(str(s) for s in source) + "]"