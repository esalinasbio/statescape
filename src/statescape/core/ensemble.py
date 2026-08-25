from __future__ import annotations

import warnings
from typing import Iterable, Tuple
from pathlib import Path

import mdtraj as md
import numpy as np
import yaml
from natsort import natsorted
from tqdm.auto import tqdm

from statescape.analysis import clustering, dimensionality, features, feature_selection
from statescape.analysis.dimensionality import ReductionResult
from statescape.util import load_feature_matrix, save_colvar, save_pdb

class Ensemble:
    """
    Collection of Molecular Dynamics simulations ensemble.
    This works as a featurization pipeline: each trajectory is loaded into memory one at a time, featurized, and saved to diks. 
    `Ensemble` itself never holds raw traejctory data into memory.
    """

    def __init__(
        self,
        pairs: list[Tuple[Path,Path]],
        *,
        names: list[str] | None = None
    ):
        """
        Build from explicit (trajectory, topology) pairs.
        For filesystem search from a folder structure, please use `Ensemble.from_root()`.
        """
        if not pairs: 
            raise ValueError('Ensemble cannot be empty.')

        self._pairs = [(Path(traj), Path(top)) for traj, top in pairs]
        self._names = list(names) if names is not None else [f'traj_{i:03d}' for i in range(len(pairs))]

        if len(self._names) != len(self._pairs):
            raise ValueError(f"Number of names ({len(self._names)}) must match number of Topology + Trajectory pairs ({len(self._pairs)}).")

    @classmethod
    def find(
        cls, 
        root: str | Path,
        *,
        subfolder: str | None = "*",
        traj_pattern: str,
        top_pattern: str | None = None,
        shared_topology: str | Path | None = None
    ) -> Ensemble:
        """
        Walk through `root/subfolder_pattern/` and pair trajectory + topology found inside each subfolder.
        Pairing is done by location: each subfolder must contain exactly one matching trajectory and topology (if not using `shared_topology`).

        Parameters
        ----------
        subfolder: which subfoldes inside `root` to consider (default '*', all subfolders)
        traj_pattern: relative trajectory pattern inside each subfolder (eg. 'prod/*.nc')
        top_pattern: relative topology pattern inside each subfolder (e.g. '*.prmtop', '*.pdb')
        shared_topology: single topology file used by all trajectories (mutually exclusive with `top_pattern`)
        """

        if (top_pattern is None) == (shared_topology is None):
            raise ValueError("Provide just one of `top_pattern` or `shared_topology`")

        root = Path(root)
        
        if subfolder is None:
            traj_paths = natsorted(Path(i) for i in root.glob(traj_pattern))
            if len(traj_paths) == 0:
                raise FileNotFoundError(f"No trajectories matching '{traj_pattern}' in {root}.")

            if shared_topology is not None:
                top_paths = [Path(shared_topology)] * len(traj_paths)
            else:
                top_matches = natsorted(Path(i) for i in root.glob(top_pattern))
                if len(top_matches) != len(traj_paths):
                    raise ValueError(
                        f"Either `shared_topology` or one topology per trajectory is required. "
                        f"Found {len(traj_paths)} trajectories and {len(top_matches)} topologies."
                    )
                top_paths = top_matches

            pairs = list(zip(traj_paths, top_paths))
            names = [i.stem for i in traj_paths]
            return cls(pairs, names=names)

        subfolders = natsorted(p for p in root.glob(subfolder) if p.is_dir())
        if len(subfolders) == 0:
            raise FileNotFoundError(f"No subfolders matching '{subfolder}' inside {root}.")

        pairs, names = [], []
        for sub in subfolders:
            traj_matches = natsorted(sub.glob(traj_pattern))
            if len(traj_matches) == 0:
                print(f"No trajectory matching '{traj_pattern}' inside {sub}.")
                continue
            if len(traj_matches) > 1:
                raise ValueError(f"Multiple trajectories matching '{traj_pattern}' in {sub}: {traj_matches}")
            traj_path = traj_matches[0]

            if shared_topology is not None:
                top_path = Path(shared_topology)
            else:
                top_matches = natsorted(Path(i) for i in sub.glob(top_pattern))
                if len(top_matches) == 0:
                    raise FileNotFoundError(f"No topology matching '{top_pattern}' inside {sub}.")
                if len(top_matches) > 1:
                    raise ValueError(f"Multiple topologies matching '{top_pattern}' in {sub}: {top_matches}")
                top_path = top_matches[0]

            pairs.append((traj_path, top_path))
            names.append(sub.name)

        return cls(pairs, names=names)

    # Featurization

    def compute_features(
        self, 
        output_dir: str | Path,
        method: str,
        *,
        format: str = "colvar",
        overwrite: bool = False,
        **kwargs
    ) -> list[Path]:
        """
        Featurize each trajectory and save features to disk. 
        One file per trajectory.

        Writes:
            output_dir/
            ├── <name_0>_<method>.<ext>
            ├── <name_1>_<method>.<ext>
            ├── ...
            ├── labels.yaml
            └── features.yaml

        Parameters
        ----------
        output_dir: location to write features' file
        method: feature extraction method, see `analysis.feature.featurize` (default: 'all_dihedrals')
        fromat: 'colvar' (default, PLUMED-compatible) or 'npy' (numpy, binary, faster loading)
        overwrite: if False, skip trajectories whose files already exist

        Returns the list of feature file paths.
        """
        if format not in ['colvar', 'npy']:
            raise ValueError(f"`format` must be 'colvar' or npy', got {format!r}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = "npy" if format == "npy" else "dat"
        feat_paths, n_frames = [],[]
        labels_path = output_dir/'labels.yaml'
        labels = None
        if labels_path.exists() and not overwrite:
            labels = yaml.safe_load(labels_path.read_text())

        pbar = tqdm(
            enumerate(self._pairs),
            total = len(self),
            desc=f'Computing {method} features'
        )

        for i, (traj_path, top_path) in pbar:
            pbar.set_postfix_str(self._names[i])
            out_path = output_dir / f"{self._names[i]}_{method}.{ext}"
            if out_path.exists() and not overwrite:
                X = load_feature_matrix(out_path, format=format)
                n_frames.append(int(X.shape[0]))
                feat_paths.append(out_path)
                continue

            traj = md.load(str(traj_path), top=str(top_path))
            feat, label = features.featurize(traj, method, **kwargs)
            del traj

            if labels is None:
                labels = label
            elif label != labels:
                first = next((k for k, (a,b) in enumerate(zip(label, labels)) if a != b), None)
                detail = (
                    f"First difference at index {first}: got {label[first]!r}, expected {labels[first]!r}"
                    if first is not None else f"got {len(label)} labels, expected {len(labels)}"
                )
                raise ValueError(
                    f"Feature labels differ for trajectory {i} "
                    f"({self._names[i]}, {traj_path}): {detail}"
                )

            if format == "npy":
                np.save(out_path, feat)
            else:
                save_colvar(feat, label, out_path)

            n_frames.append(feat.shape[0])
            feat_paths.append(out_path)
        
        #write shared label file
        if labels is not None:
            labels_path.write_text(yaml.safe_dump(labels, sort_keys=False))
        #write manifest
        manifest = {
            "method": method,
            "format": format,
            "n_trajectories": len(self),
            "n_features": len(labels) if labels else None,
            "kwargs": {k: str(v) for k, v in kwargs.items()},
            "trajectories": [
                {
                    'name': self._names[i],
                    "trajectory_path": str(self._pairs[i][0]),
                    "topology_path": str(self._pairs[i][1]),
                    "feature_file": str(feat_paths[i]),
                    "n_frames": n_frames[i]
                }
                for i in range(len(self))
            ]
        }
        (output_dir / "features.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

        return feat_paths

    @staticmethod
    def load_features(features_dir: str | Path) -> tuple[list[Path], list[str], dict]:
        """
        Load a previously computed feature set.
        Returns (feature_paths, labels, manifest)
        """
        features_dir = Path(features_dir)
        manifest = yaml.safe_load((features_dir / 'features.yaml').read_text())
        labels = yaml.safe_load((features_dir / 'labels.yaml').read_text())
        feature_paths = [Path(t["feature_file"]) for t in manifest["trajectories"]]
        return feature_paths, labels, manifest

    def _load_feature_blocks(
        self,
        features_dir: str | Path,
        *,
        stride: int = 1
    ) -> tuple[list[np.ndarray], list[str]]:
        """
        Loads one feature array per replica. Trajectory doundaries are preserved.
        Methods that depend on frame order (like tICA) recieve the blocks as they are.

        Parameters
        ---------
        features_dir: directory written by `compute_features`
        stride: load every `stride`-th frame of each file

        Returns
        -------
        blocks: list[np.ndarray]
                `blocks[i]` has shape (n_frames_i, n_features) for simulation i
        labels: list[str]
                Feature names, shared by every block
        """
        feature_paths, labels, manifest = self.load_features(features_dir)
        format = manifest['format']

        blocks = []
        for path in feature_paths:
            X = load_feature_matrix(path, format=format)
            # check if feature_matrix has the same number of features as labels
            if X.shape[1] != len(labels):
                raise ValueError(f"{path} has {X.shape[1]} columns, but {features_dir / 'labels.yaml'} has {len(labels)} labels")
            blocks.append(X[::stride] if stride > 1 else X)

        size_gb = sum(b.nbytes for b in blocks) / 1024 ** 3
        if size_gb > 5.0:
            warnings.warn(f"Loaded {size_gb:.1f} GB from {features_dir}, increase `stride` to reduce size.", stacklevel=3)
        return blocks, labels

    def select_features(
        self, 
        features_dir: str | Path,
        *,
        method: str = 'amino',
        stride: int = 1,
        output_dir: str | Path | None = None,
        **kwargs
    ) -> list [str]:
        """
        Run Automatic feature selection over the whole ensemble

        Loads the feature file, concatenates trajectories (optionally strided), and runs
        the feature selection method. If `output_dir` is given, writes a reduced feature 
        set there with the same layout as `features_dir`.

        Unlike `compute_features`, this holds the concatenated feature matrix in memory,
        so use `stride` on long trajectories.

        Returns the selected feature labels.
        """
        features_dir = Path(features_dir)
        _, _, manifest = self.load_features(features_dir)
        format = manifest["format"]

        blocks, labels = self._load_feature_blocks(features_dir, stride=stride)
        feature_matrix = np.vstack(blocks)
        del blocks

        selector = {
            "amino": feature_selection.amino.select,
        }
        if method not in selector:
            raise ValueError(f"Unknown selection method: {method!r}. Available: {list(selector)}")
        selected_labels = selector[method](feature_matrix,labels, **kwargs)

        if output_dir is not None:
            self._write_selected(output_dir, labels, selected_labels, manifest, format)
        return selected_labels

    def _write_selected(
        self, 
        output_dir: str | Path,
        labels: list[str], 
        selected_labels: list[str], 
        manifest: dict, 
        format: str
    ) -> None:
        """
        Slice selected columns (`selected_labels`) from each feature file  and save.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = "npy" if format == "npy" else "dat"
        col_idx = [labels.index(i) for i in selected_labels]
        method = manifest["method"]

        trajs = []
        for i in manifest['trajectories']:
            X = load_feature_matrix(Path(i['feature_file']), format=format)
            X_sel = X[:, col_idx]
            out_path = output_dir / f'{i["name"]}_{method}_selected.{ext}'
            if format == "npy":
                np.save(out_path, X_sel)
            else:
                save_colvar(X_sel, selected_labels, out_path)
            trajs.append({**i, "feature_file": str(out_path), "n_frames": X_sel.shape[0]})

        (output_dir / "labels.yaml").write_text(yaml.safe_dump(selected_labels, sort_keys=False))
        new_manifest = {
            **manifest,
            "method": f"{method}_selected",
            "n_features": len(selected_labels),
            "trajectories": trajs
        }
        (output_dir / "features.yaml").write_text(yaml.safe_dump(new_manifest, sort_keys=False))

    ## Dimensionality reduction and clustering

    @staticmethod
    def _row_index(blocks: list[np.ndarray]) -> np.ndarray:
        """
        Map each row of `np.vstack(blocks)` to its origin index

        Returns
        --------
        np.ndarray: Shape (n_rows,2). First column is the simulation number, the second
        column is the frame index within that simulation, counting rows of the block.
        If strided, it corresponds to strided frames, not frames of the original trajectory
        """
        # Maps every item in blocks to a 2D array: [[block_idx, frame_idx], ...]
        # example: block 0 with 2 frames, and block 1 with 3 frames -> [[0,0], [0,1], [1,0], [1,1], [1,2]]
        return np.vstack([np.column_stack([np.full(len(b), i), np.arange(len(b))]) for i, b in enumerate(blocks)])

    def dimensionality(
        self,
        features_dir: str | Path,
        *,
        method: str = 'pca',
        n_components: int = 10,
        stride: int = 1,
        **kwargs
    ) -> tuple[ReductionResult, np.ndarray]:
        """
        Dimensionality reduction over the feature amtrix of the whole MD ensemble.
        Every simulation is reduced to one shared space, so the resulting coordinates
        are directly comparable between simulations

        Parameters
        ----------
        features_dir: directory written by `compute_features`
        method: 'pca' or 'umap' (default: pca)
        n_components: number of output dimensions
        stride: load every `stride`-th frame of each file
        **kwargs: passed to the reducer

        Returns
        -------
        result: ReductionResult with `.coords` with shape (n_rows, n_components)
        index: (replica, frame) pair for every row of `result.coords`
        """
        blocks, _ = self._load_feature_blocks(features_dir, stride=stride)
        index = self._row_index(blocks)

        dim_map = {
            "pca": lambda: dimensionality.pca(np.vstack(blocks), n_components=n_components, **kwargs),
            "umap": lambda: dimensionality.umap(np.vstack(blocks), n_components=n_components, **kwargs)
        }

        return dim_map[method](), index

    def cluster(
        self,
        coords: np.ndarray,
        index: np.ndarray,
        *,
        method: str = 'kmeans',
        n_clusters: int = 50,
        radius: float = 1.5, # only for regular space clustering
        n_components: int = 2,
        **kwargs
    ) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
        """
        Cluster frames in latent coordinates and get one representative per cluster

        Parameters
        ----------
        cords: (n_rows,n_features) latent coordinates from dimensionality
        index: (n_rows, 2) simulation/frame index from dimensionality
        method: 'kmeans', 'gmm' or 'regular_space' (default: kmeans)
        n_clusters: number of clusters (kmeans and gmm only)
        radius: radius threshold (regular_space only)
        n_components: how many leading columns of `coords` to use.
            Meaningful only for ordered spaces like PCA. For UMAP, set the 
            `n_components` at dimensionality reduction time and leave this at the full
            embedding size
        **kwargs: passed to the clusterer

        Returns
        --------
        labels: Cluster id array of every row of `coords`
        representatives: Cluster id to the (simulation, frame) pair nearest its centroid
        """
        coords = np.asarray(coords)
        index = np.asarray(index)

        if len(coords) != len(index):
            raise ValueError(f"coords has {len(coords)} rows but index has {len(index)}.")
        if coords.ndim != 2:
            raise ValueError(f'coords must be 2D, got shape {coords.shape}')
        if coords.shape[1] < n_components:
            raise ValueError(f"coords has {coords.shape[1]} columns, n_components={n_components} requested")

        data = coords[:, :n_components]

        cluster_map = {
            "kmeans": lambda: clustering.kmeans(data, n_clusters=n_clusters, **kwargs),
            "gmm": lambda: clustering.gmm(data, n_clusters=n_clusters, **kwargs),
            "regular_space": lambda: clustering.regular_space(data, radius)
        }
        if method not in cluster_map:
            raise ValueError(f"Unknown cluster method: {method!r}. Available: {list(cluster_map.keys())}")

        labels, reps = cluster_map[method]()
        # dictionary {cluster_id: (simulation_idx, frame_idx)}
        representatives = {int(cid): (int(index[row, 0]), int(index[row, 1])) for cid, row in reps.items()}
        return labels, representatives

    def save_cluster_representatives(
            self,
            representatives: dict[int, tuple[int, int]],
            output_dir: str | Path,
            *,
            stride: int = 1,
            prefix: str = 'cluster'
    ) -> list:
        """
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        by_sim= {}
        for cid, (sim, frame) in representatives.items():
            if not 0 <= sim < len(self):
                raise ValueError(f'Cluster {cid} mention simulation {sim}, but the ensemble has {len(self)}')
            by_sim.setdefault(sim, []).append((int(cid), int(frame)))

        write = {}
        for sim in sorted(by_sim):
            traj_path, top_path = self._pairs[sim]
            traj = md.load(str(traj_path), top=str(top_path))
            for cid, frame in by_sim[sim]:
                frame_idx = f * stride
                if frame_idx >= traj.n_frames:
                    raise IndexError(f"Cluster {cid:}: frame {frame_idx} is out of bonds for {traj_path} ({traj.n_frames} total frames).")
                out = output_dir / f'{prefix}_{cid:02d}_{self._names[sim]}_f{frame_idx}.pdb'
                write[cid] = save_pdb(traj[frame_idx], out)
            del traj

        return [write[cid] for cid in sorted(write)]


    #properties

    @property
    def n_trajectories(self) -> int:
        return len(self._pairs)

    @property
    def names (self) -> list[str]:
        return list(self._names)

    @property
    def trajectories(self) -> list[Path]:
        return [traj for traj, _ in self._pairs]

    @property
    def topologies(self) -> list[Path]:
        return [top for _, top in self._pairs]
    

    def __len__(self) -> int:
        return len(self._pairs)

    def __iter__(self) -> Iterable[tuple[str, Path, Path]]:
        for name, (traj, top) in zip(self._names, self._pairs):
            yield name, traj, top

    def __getitem__(self, idx: int) -> md.Trajectory:
        traj, top = self._pairs[idx]
        return md.load(str(traj), str(top))

    def __repr__(self) -> str:
        return (f"Ensemble(n_trajectories={len(self)}, names = [{self._names[0]!r}, ..., {self._names[-1]!r}])")