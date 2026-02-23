from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Any, Tuple, Iterable, Sequence


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
    def from_pdb_folder(
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

        globber = folder_path.rglob if recursive else folder_path.glob
        pdbs = sorted(
            {p for p in globber("*") if p.is_file() and p.suffix.lower() == ".pdb"},
            key=lambda p: str(p),
        )
        if strict and not pdbs:
            raise FileNotFoundError(f"No .pdb files found in {folder_path}")

        return cls.from_pdb_files(pdbs, metadata=metadata, parent=parent, strict=strict)

    @classmethod
    def from_trajectory(
        cls,
        trajectory: str | Path,
        *,
        topology: str | Path | None = None,
        start: int = 0,
        stop: int | None = None,
        stride: int = 1,
        metadata: Mapping[str, Any] | None = None,
        parent: ConformerSet | None = None,
    ) -> ConformerSet:
        """
        Build a ConformerSet from a trajectory (e.g. XTC) by creating one ConformerRef per frame.

        Notes:
        - Many trajectory formats (XTC/DCD/TRR/...) require an external `topology` file (e.g. PDB/GRO/PSF).
        - This constructor *counts* frames using `MDAnalysis` or `mdtraj` if available.
          If neither is installed, it raises an ImportError explaining what to install.
        """
        traj_path = Path(trajectory)
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)

        topo_path: Path | None = Path(topology) if topology is not None else None
        if topo_path is not None and not topo_path.exists():
            raise FileNotFoundError(topo_path)

        if stride <= 0:
            raise ValueError("stride must be >= 1")
        if start < 0:
            raise ValueError("start must be >= 0")
        if stop is not None and stop < 0:
            raise ValueError("stop must be >= 0 or None")

        n_frames = _count_trajectory_frames(traj_path, topo_path)
        effective_stop = n_frames if stop is None else min(stop, n_frames)
        if start > effective_stop:
            raise ValueError(f"start ({start}) must be <= stop ({effective_stop})")

        conformers = tuple(
            ConformerRef(path=traj_path, frame=i, topology=topo_path)
            for i in range(start, effective_stop, stride)
        )
        return cls(
            conformers=conformers,
            metadata=dict(metadata or {}),
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


def _count_trajectory_frames(trajectory: Path, topology: Path | None) -> int:
    """
    Return number of frames in a trajectory.
    """
    try:
        import mdtraj as md  # type: ignore
    except ImportError as e:
        msg = (
            "mdtraj is required to read trajectories. "
            "Install it with `pip install mdtraj`."
        )
        raise ImportError(msg) from e

    if topology is None:
        raise ValueError(
            "topology is required to read this trajectory "
            "(e.g. an XTC needs a PDB/GRO/PSF topology)"
        )

    n_frames = 0
    for chunk in md.iterload(str(trajectory), top=str(topology)):
        n_frames += chunk.n_frames
    return n_frames
