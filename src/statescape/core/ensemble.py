from __future__ import annotations

import mdtraj as md
import numpy as np
import yaml

from glob import glob
from pathlib import Path
from natsort import natsorted
from tqdm.auto import tqdm
from typing import Iterable, Sequence, Tuple

from statescape.analysis import features, feature_selection, dimensionality, clustering
from statescape._vendor.amino._colvar import Colvar
from statescape.util import save_colvar

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
        names: list[str]
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
                    raise FileNotFoundError(f"No trajectory matching '{top_pattern}' inside {sub}.")
                if len(top_matches) > 1:
                    raise ValueError(f"Multiple trajectories matching '{top_pattern}' in {sub}: {top_matches}")
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
                X = self._load_feature_matrix(out_path, format)
                n_frames.append(int(X.shape[0]))
                feat_paths.append(out_path)
                continue

            traj = md.load(str(traj_path), top=str(top_path))
            feat, label = features.featurize(traj, method, **kwargs)
            del traj

            if labels is None:
                labels = label
            elif label != labels:
                raise ValueError(
                    f'Feature labels differ for trajectory {i} ({self._names[i]}, {traj_path}):'
                    f'got {len(label)} labels, expected {len(labels)}'
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

    @staticmethod
    def _load_feature_matrix(path: Path, format: str) -> np.ndarray:
        """
        Load feature files as an (n_frames, n_features) feature matrix
        """
        if format== 'npy':
            return np.load(path)
        return Colvar.from_file(str(path)).data.T # colvar stores (n_features, n_frames)

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
        Run Automatic feature selection
        Loads the feature directory, concatenates trajectories (optionally strided), and runs
        the feature selection method.
        If `output_dir` is given, writes a reduced feature set to disk.

        Returns the selected feature labels.
        """
        feature_paths, labels, manifest = self.load_features(features_dir)
        format = manifest["format"]

        trajs = []
        for i in feature_paths:
            X = self._load_feature_matrix(Path(i), format)
            if stride > 1:
                X = X[::stride]
            trajs.append(X)
        feature_matrix = np.vstack(trajs)

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
            X = self._load_feature_matrix(Path(i['feature_file']), format)
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