from __future__ import annotations

import glob
from natsort import natsorted
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Any, Tuple, Sequence


@dataclass(frozen=True)
class ConformerRef:
    '''
    A reference to a single protein conformation.

    This object does not contain the conformational data itself, but rather a reference to it.
    It only describes where and how to retrieve the conformational data.
    '''
    path: Path
    frame: int | None = None
    topology: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConformerSet:
    '''
    Immutable collection of protein conformations.

    This object represents a logical dataset of conformers that may
    originate from different sources (e.g. PDB files, trajectory frames, etc.).
    '''
    conformers: Tuple[ConformerRef, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parent: ConformerSet | None = None

    @classmethod
    def from_pdb_files(
        cls,
        pdb_files: Sequence[str | Path],
        *,
        metadata: Mapping[str, Any] | None = None,
        parent: ConformerSet | None = None,
        strict: bool = True,
    ) -> ConformerSet:
        """
        Build a ConformerSet from a list of PDB file paths.

        If `strict=True`, missing paths raise `FileNotFoundError`.
        """
        conformers: list[ConformerRef] = []
        for p in pdb_files:
            path = Path(p)
            if strict and not path.exists():
                raise FileNotFoundError(path)
            conformers.append(ConformerRef(path=path))

        return cls(
            conformers=tuple(conformers),
            metadata=dict(metadata or {}),
            parent=parent,
        )

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        recursive: bool = False,
        metadata: Mapping[str, Any] | None = None,
        parent: ConformerSet | None = None,
        strict: bool = True,
    ) -> ConformerSet:
        """
        Build a ConformerSet from a folder containing PDB files.

        If `recursive=True`, scans subfolders. Files are sorted by path.
        If `strict=True`, an empty result raises `FileNotFoundError`.
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(folder_path)
        if not folder_path.is_dir():
            raise NotADirectoryError(folder_path)

        pattern = "**/*.pdb" if recursive else "*.pdb"
        pdbs = natsorted(folder_path.glob(pattern), key=lambda p: str(p))

        if strict and not pdbs:
            raise FileNotFoundError(f"No .pdb files found in {folder_path}")

        return cls.from_pdb_files(pdbs, metadata=metadata, parent=parent, strict=strict)

    @classmethod
    def from_trajectory(
        cls,
        trajectory: str | Path,
        *,
        topology: str | Path | None = None,
        n_frames: int | None = None,
        frames: Sequence[int] | None = None,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        metadata: Mapping[str, Any] | None = None,
        parent: ConformerSet | None = None,
    ) -> ConformerSet:
        """
        Build a ConformerSet from a trajectory (e.g. XTC) by creating one ConformerRef per frame.

        This constructor is purely index-based and does not inspect the trajectory file.
        Callers are responsible for determining the number of frames (e.g. via the metrics layer).

        You must provide either:
        - `frames`: explicit sequence of frame indices, or
        - `n_frames`: total number of frames, optionally with `start`/`stop`/`stride` to define a range.
        """
        traj_path = Path(trajectory)
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)

        topo_path: Path | None = Path(topology) if topology is not None else None
        if topo_path is not None and not topo_path.exists():
            raise FileNotFoundError(topo_path)

        if frames is not None and n_frames is not None:
            raise ValueError("Provide either `frames` or `n_frames`, not both.")

        if stride <= 0:
            raise ValueError("stride must be >= 1")
        if start < 0:
            raise ValueError("start must be >= 0")
        if stop is not None and stop < 0:
            raise ValueError("stop must be >= 0 or None")

        # Mode 1: explicit frame indices
        if frames is not None:
            if not frames:
                raise ValueError("`frames` must not be empty.")
            if any(f < 0 for f in frames):
                raise ValueError("Frame indices in `frames` must be >= 0.")
            frame_indices = tuple(frames)
        else:
            # Mode 2: range over known number of frames
            if n_frames is None:
                raise ValueError("You must provide either `frames` or `n_frames`.")
            if n_frames < 0:
                raise ValueError("`n_frames` must be >= 0.")

            effective_stop = n_frames if stop is None else min(stop, n_frames)
            if start > effective_stop:
                raise ValueError(f"start ({start}) must be <= stop ({effective_stop})")
            frame_indices = tuple(range(start, effective_stop, stride))

        conformers = tuple(
            ConformerRef(path=traj_path, frame=i, topology=topo_path)
            for i in frame_indices
        )
        return cls(
            conformers=conformers,
            metadata=dict(metadata or {}),
            parent=parent,
        )

    @classmethod
    def from_source(
        cls,
        *,
        pdb_files: Sequence[str | Path] | None = None,
        folder: str | Path | None = None,
        trajectory: str | Path | None = None,
        topology: str | Path | None = None,
        recursive: bool = False,
        stride: int = 1,
        metadata: Mapping[str, Any] | None = None,
        parent: ConformerSet | None = None,
        strict: bool = True,
    ) -> ConformerSet:
        """
        Unified constructor for common conformer sources.

        This method supports:
        - Multiple PDB files:    pass `pdb_files=[...]`
        - A folder of PDB files: pass `folder="path/to/dir"`
        - A trajectory + topology (e.g. XTC + PDB): pass `trajectory=...`, `topology=...`

        Notes
        -----
        - For the trajectory case, this method uses mdtraj to count frames
          (without loading the full trajectory into memory) and then builds
          a ConformerSet via `from_trajectory`. This intentionally introduces
          an optional dependency on mdtraj for a better UX, as requested.
        """
        sources = [
            pdb_files is not None,
            folder is not None,
            trajectory is not None,
        ]
        if sum(int(s) for s in sources) != 1:
            raise ValueError(
                "Specify exactly one of `pdb_files`, `folder`, or `trajectory`."
            )

        if pdb_files is not None:
            return cls.from_pdb_files(
                pdb_files=pdb_files,
                metadata=metadata,
                parent=parent,
                strict=strict,
            )

        if folder is not None:
            return cls.from_folder(
                folder=folder,
                recursive=recursive,
                metadata=metadata,
                parent=parent,
                strict=strict,
            )

        # Trajectory + topology
        traj_path = Path(trajectory)  # type: ignore[arg-type]
        topo_path = Path(topology) if topology is not None else None
        if topo_path is None:
            raise ValueError("`topology` is required when `trajectory` is provided.")

        if stride <= 0:
            raise ValueError("stride must be >= 1")

        n_frames_total = _count_trajectory_frames_mdtraj(traj_path, topo_path)
        merged_meta = dict(metadata or {})
        merged_meta.setdefault("n_frames_total", n_frames_total)

        return cls.from_trajectory(
            trajectory=traj_path,
            topology=topo_path,
            n_frames=n_frames_total,
            stride=stride,
            metadata=merged_meta,
            parent=parent,
        )

    def __len__(self) -> int:
        return len(self.conformers)

    def __iter__(self):
        return iter(self.conformers)

    def __getitem__(self, index: int) -> ConformerRef:
        return self.conformers[index]

    def __contains__(self, item: ConformerRef) -> bool:
        return item in self.conformers

    def subset(self, indices: Sequence[int]) -> ConformerSet:
        """
        Return a new ConformerSet with conformers at the given indices.

        This is a cheap view: it reuses existing ConformerRef instances
        and sets this ConformerSet as the parent of the result.
        """
        if not indices:
            return ConformerSet(
                conformers=tuple(),
                metadata=dict(self.metadata),
                parent=self,
                provenance=self.provenance,
            )

        n = len(self.conformers)
        idx_list = list(indices)
        for i in idx_list:
            if i < 0 or i >= n:
                raise IndexError(f"Conformer index out of range: {i}")

        new_conformers = tuple(self.conformers[i] for i in idx_list)
        return ConformerSet(
            conformers=new_conformers,
            metadata=dict(self.metadata),
            parent=self,
            provenance=self.provenance,
        )

    def split_by_indices(self, indices: Sequence[int]) -> Tuple[ConformerSet, ConformerSet]:
        """
        Split this ConformerSet into kept and dropped subsets based on indices.

        Returns (kept, dropped), where `kept` contains conformers at the given
        indices and `dropped` contains the remaining conformers.
        """
        n = len(self.conformers)
        idx_set = set(indices)
        for i in idx_set:
            if i < 0 or i >= n:
                raise IndexError(f"Conformer index out of range: {i}")

        kept = []
        dropped = []
        for pos, ref in enumerate(self.conformers):
            if pos in idx_set:
                kept.append(ref)
            else:
                dropped.append(ref)

        kept_set = ConformerSet(
            conformers=tuple(kept),
            metadata=dict(self.metadata),
            parent=self,
            provenance=self.provenance,
        )
        dropped_set = ConformerSet(
            conformers=tuple(dropped),
            metadata=dict(self.metadata),
            parent=self,
            provenance=self.provenance,
        )
        return kept_set, dropped_set

    def split_by_mask(self, mask: Sequence[bool]) -> Tuple[ConformerSet, ConformerSet]:
        """
        Split this ConformerSet into kept and dropped subsets based on a boolean mask.

        The mask must have the same length as this ConformerSet. `True` values mark
        conformers to keep; `False` values mark conformers to drop.
        """
        if len(mask) != len(self.conformers):
            raise ValueError(
                f"Mask length ({len(mask)}) must match number of conformers ({len(self.conformers)})."
            )

        kept = []
        dropped = []
        for ref, flag in zip(self.conformers, mask):
            if flag:
                kept.append(ref)
            else:
                dropped.append(ref)

        kept_set = ConformerSet(
            conformers=tuple(kept),
            metadata=dict(self.metadata),
            parent=self,
            provenance=self.provenance,
        )
        dropped_set = ConformerSet(
            conformers=tuple(dropped),
            metadata=dict(self.metadata),
            parent=self,
            provenance=self.provenance,
        )
        return kept_set, dropped_set

    @classmethod
    def merge(
        cls,
        sets: Sequence[ConformerSet],
        *,
        metadata: Mapping[str, Any] | None = None,
        parent: ConformerSet | None = None,
        allow_duplicates: bool = True,
    ) -> ConformerSet:
        """
        Merge multiple ConformerSets into a single one.

        By default this concatenates conformers from all input sets.
        If `allow_duplicates` is False, duplicate ConformerRef objects
        (by value) are removed while preserving order.
        """
        all_conformers: list[ConformerRef] = []
        for s in sets:
            all_conformers.extend(s.conformers)

        if not allow_duplicates:
            seen: set[ConformerRef] = set()
            unique: list[ConformerRef] = []
            for ref in all_conformers:
                if ref not in seen:
                    seen.add(ref)
                    unique.append(ref)
            all_conformers = unique

        merged_metadata = dict(metadata or {})
        return cls(
            conformers=tuple(all_conformers),
            metadata=merged_metadata,
            parent=parent,
        )


def _count_trajectory_frames_mdtraj(trajectory: Path, topology: Path) -> int:
    """
    Return number of frames in a trajectory using mdtraj.

    This helper is used by `ConformerSet.from_source` for the trajectory case.
    """
    try:
        import mdtraj as md  # type: ignore
    except ImportError as e:  # pragma: no cover - import error path
        msg = (
            "mdtraj is required to use `ConformerSet.from_source` with a trajectory. "
            "Install it with `pip install mdtraj`."
        )
        raise ImportError(msg) from e

    n_frames = 0
    for chunk in md.iterload(str(trajectory), top=str(topology)):
        n_frames += chunk.n_frames
    return n_frames
