from __future__ import annotations

import numpy as np
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple, Sequence, Any

from natsort import natsorted


@dataclass(frozen=True)
class ConformerRef:
    """
    A reference to a single protein conformation.

    This object does not contain the conformational data itself, but rather a reference to it.
    It only describes where and how to retrieve the conformational data.
    """

    path: Path
    frame: int | None = None
    topology: Path | None = None

    def __repr__(self) -> str:
        if self.frame is not None:
            return f"ConformerRef({self.path.name}, frame={self.frame})"
        else:
            return f"ConformerRef({self.path.name})"


@dataclass(frozen=True)
class ConformerSet:
    """
    Immutable collection of protein conformations.

    This object represents a logical dataset of conformers that may
    originate from different sources (e.g. PDB files, trajectory frames, etc.).
    """

    conformers: Tuple[ConformerRef, ...]
    reference: Path | None = None
    _is_folder: bool | None = None
    load_in_memory: bool = True
    _in_memory_conformations: tuple[Any, ...] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _trajectory_cache: dict[tuple[Path, Path], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.load_in_memory and self._in_memory_conformations is None:
            object.__setattr__(
                self,
                "_in_memory_conformations",
                tuple(self._load_conformation(ref) for ref in self.conformers),
            )

    @classmethod
    def from_pdb_files(
        cls,
        pdb_files: Sequence[str | Path],
        *,
        reference: str | Path | None = None,
        strict: bool = True,
        _is_folder: bool = False,
        load_in_memory: bool = True,
    ) -> ConformerSet:
        """
        Build a ConformerSet from a list of PDB file paths.

        If `strict=True`, missing paths raise `FileNotFoundError`.
        Optionally provide a `reference` PDB path for alignment or comparison.
        """
        conformers: list[ConformerRef] = []
        for p in pdb_files:
            path = Path(p)
            if strict and not path.exists():
                raise FileNotFoundError(path)
            conformers.append(ConformerRef(path=path))

        ref_path: Path | None = Path(reference) if reference is not None else None
        if ref_path is not None and strict and not ref_path.exists():
            raise FileNotFoundError(ref_path)

        return cls(
            conformers=tuple(conformers),
            reference=ref_path,
            _is_folder=_is_folder,
            load_in_memory=load_in_memory,
        )

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        recursive: bool = False,
        reference: str | Path | None = None,
        strict: bool = True,
        _is_folder: bool = True,
        load_in_memory: bool = True,
    ) -> ConformerSet:
        """
        Build a ConformerSet from a folder containing PDB files.

        If `recursive=True`, scans subfolders. Files are sorted by path.
        If `strict=True`, an empty result raises `FileNotFoundError`.
        Optionally provide a `reference` PDB path for alignment or comparison.
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

        return cls.from_pdb_files(
            pdbs,
            reference=reference,
            strict=strict,
            _is_folder=True,
            load_in_memory=load_in_memory,
        )

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
        reference: str | Path | None = None,
        _is_folder: bool = False,
        load_in_memory: bool = True,
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
        ref_path: Path | None = Path(reference) if reference is not None else None
        if ref_path is not None and not ref_path.exists():
            raise FileNotFoundError(ref_path)
        return cls(
            conformers=conformers,
            reference=ref_path,
            load_in_memory=load_in_memory,
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
        reference: str | Path | None = None,
        strict: bool = True,
        _is_folder: bool = False,
        load_in_memory: bool = True,
    ) -> ConformerSet:
        """
        Unified constructor for common conformer sources.

        This method supports:
        - Multiple PDB files:    pass `pdb_files=[...]`
        - A folder of PDB files: pass `folder="path/to/dir"`
        - A trajectory + topology (e.g. XTC + PDB): pass `trajectory=...`, `topology=...`

        Optionally provide a `reference` PDB path for alignment or comparison (all sources).

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

        # check that exactly one source is provided
        if sum(int(s) for s in sources) != 1:
            raise ValueError(
                "Specify exactly one of `pdb_files`, `folder`, or `trajectory`."
            )

        if pdb_files is not None:
            return cls.from_pdb_files(
                pdb_files=pdb_files,
                reference=reference,
                strict=strict,
                load_in_memory=load_in_memory,
            )

        if folder is not None:
            return cls.from_folder(
                folder=folder,
                recursive=recursive,
                reference=reference,
                strict=strict,
                _is_folder=True,
                load_in_memory=load_in_memory,
            )

        # Trajectory + topology
        traj_path = Path(trajectory)  # type: ignore[arg-type]
        topo_path = Path(topology) if topology is not None else None
        if topo_path is None:
            raise ValueError("`topology` is required when `trajectory` is provided.")

        if stride <= 0:
            raise ValueError("stride must be >= 1")

        n_frames_total = _count_trajectory_frames_mdtraj(traj_path, topo_path)

        return cls.from_trajectory(
            trajectory=traj_path,
            topology=topo_path,
            n_frames=n_frames_total,
            stride=stride,
            reference=reference,
            load_in_memory=load_in_memory,
        )

    def subset(self, indices: Sequence[int]) -> ConformerSet:
        """
        Return a new ConformerSet with conformers at the given indices.

        This is a cheap view: it reuses existing ConformerRef instances.
        """
        if not indices:
            return self._new_like(conformers=tuple(), loaded=tuple())

        n = len(self.conformers)
        idx_list = list(indices)
        for i in idx_list:
            if i < 0 or i >= n:
                raise IndexError(f"Conformer index out of range: {i}")

        new_conformers = tuple(self.conformers[i] for i in idx_list)
        loaded: tuple[Any, ...] | None = None
        if self._in_memory_conformations is not None:
            loaded = tuple(self._in_memory_conformations[i] for i in idx_list)
        return self._new_like(conformers=new_conformers, loaded=loaded)

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

        kept_loaded: tuple[Any, ...] | None = None
        dropped_loaded: tuple[Any, ...] | None = None
        if self._in_memory_conformations is not None:
            kept_loaded_list: list[Any] = []
            dropped_loaded_list: list[Any] = []
            for pos, conf in enumerate(self._in_memory_conformations):
                if pos in idx_set:
                    kept_loaded_list.append(conf)
                else:
                    dropped_loaded_list.append(conf)
            kept_loaded = tuple(kept_loaded_list)
            dropped_loaded = tuple(dropped_loaded_list)

        kept_set = self._new_like(conformers=tuple(kept), loaded=kept_loaded)
        dropped_set = self._new_like(conformers=tuple(dropped), loaded=dropped_loaded)
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

        kept_loaded: tuple[Any, ...] | None = None
        dropped_loaded: tuple[Any, ...] | None = None
        if self._in_memory_conformations is not None:
            kept_loaded_list: list[Any] = []
            dropped_loaded_list: list[Any] = []
            for conf, flag in zip(self._in_memory_conformations, mask):
                if flag:
                    kept_loaded_list.append(conf)
                else:
                    dropped_loaded_list.append(conf)
            kept_loaded = tuple(kept_loaded_list)
            dropped_loaded = tuple(dropped_loaded_list)

        kept_set = self._new_like(conformers=tuple(kept), loaded=kept_loaded)
        dropped_set = self._new_like(conformers=tuple(dropped), loaded=dropped_loaded)
        return kept_set, dropped_set

    def filter(self, fn: Callable[[int], bool]) -> ConformerSet:
        """
        Return a subset selected by an index predicate.

        The predicate receives each conformer index and must return a boolean.
        """
        indices = [i for i in range(len(self.conformers)) if fn(i)]
        return self.subset(indices)

    def filter_mask(self, mask: Sequence[bool]) -> ConformerSet:
        """
        Return a subset selected by a boolean mask.

        The mask must have the same length as this ConformerSet.
        """
        if len(mask) != len(self.conformers):
            raise ValueError(
                f"Mask length ({len(mask)}) must match number of conformers ({len(self.conformers)})."
            )
        indices = [i for i, keep in enumerate(mask) if keep]
        return self.subset(indices)

    def groupby_source(self) -> dict[tuple[Path, Path | None], list[ConformerRef]]:
        """
        Group conformers by source (path, topology), preserving insertion order.
        """
        groups: dict[tuple[Path, Path | None], list[ConformerRef]] = {}
        for ref in self.conformers:
            key = (ref.path, ref.topology)
            if key not in groups:
                groups[key] = []
            groups[key].append(ref)
        return groups

    @classmethod
    def merge(
        cls,
        sets: Sequence[ConformerSet],
        *,
        reference: str | Path | None = None,
        allow_duplicates: bool = True,
    ) -> ConformerSet:
        """
        Merge multiple ConformerSets into a single one.

        By default this concatenates conformers from all input sets.
        If `allow_duplicates` is False, duplicate ConformerRef objects
        (by value) are removed while preserving order.
        If `reference` is not provided, the reference from the first set is used (if any).
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

        ref_path: Path | None = Path(reference) if reference is not None else None
        if ref_path is None and sets:
            ref_path = sets[0].reference
        return cls(
            conformers=tuple(all_conformers),
            reference=ref_path,
            load_in_memory=all(s.load_in_memory for s in sets) if sets else True,
        )

    def to_pdb(self, output_dir: Path) -> None:
        """
        Export each conformer as a PDB file in `output_dir`.

        - Static conformers (`frame is None`) are copied with their original filename.
        - Trajectory conformers (`frame is not None`) are saved as `<basename>_<frame>.pdb`
          where <basename> comes from the trajectory file name (no extension).
        - Output files will not be overwritten: if a filename would be repeated, append a numeric suffix.
        - Output order matches the original conformer order in `self.conformers`.
        - Efficient: trajectory sources are loaded at most once each.
        - Raises ValueError if any trajectory conformer lacks a topology.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename_counts: dict[str, int] = {}
        traj_loaded: dict[tuple[Path, Path], Any] = {}

        for ref in self.conformers:
            if ref.frame is None:
                _export_static(ref=ref, output_dir=output_dir, counts=filename_counts)
            else:
                _export_frame(
                    ref=ref,
                    output_dir=output_dir,
                    counts=filename_counts,
                    traj_cache=traj_loaded,
                )
        return

    def __repr__(self) -> str:
        n = len(self.conformers)
        if n == 0:
            return "ConformerSet(n=0)"
        # Count unique sources
        sources = len({(r.path, r.topology) for r in self.conformers})
        # Detect types
        has_traj = any(r.frame is not None for r in self.conformers)
        has_pdb = any(r.frame is None for r in self.conformers)
        types = []
        if has_traj:
            types.append("trajectory")
        if has_pdb:
            types.append("pdb")
        path = {str(self.conformers[0].path.parent)}
        ref = self.reference.name if self.reference is not None else None
        return (
            f"ConformerSet("
            f"n={n}, "
            f"sources={sources}, "
            f"type={types}, "
            f"path={path}, "
            f"reference={ref}"
            f")"
        )

    def __len__(self) -> int:
        return len(self.conformers)

    def __iter__(self):
        return iter(self.conformers)

    def __getitem__(self, index: int) -> ConformerRef:
        return self.conformers[index]

    def __contains__(self, item: ConformerRef) -> bool:
        return item in self.conformers

    def get_conformation(self, index: int) -> Any:
        """Return one conformation by index."""
        if index < 0 or index >= len(self.conformers):
            raise IndexError(f"Conformer index out of range: {index}")
        if self._in_memory_conformations is not None:
            return self._in_memory_conformations[index]
        return self._load_conformation(self.conformers[index])

    def get_conformations(self) -> tuple[Any, ...]:
        """Return all conformations for this set."""
        if self._in_memory_conformations is not None:
            return self._in_memory_conformations
        return tuple(self._load_conformation(ref) for ref in self.conformers)

    def _new_like(
        self,
        *,
        conformers: tuple[ConformerRef, ...],
        loaded: tuple[Any, ...] | None = None,
    ) -> ConformerSet:
        should_load = self.load_in_memory and loaded is None
        new_set = ConformerSet(
            conformers=conformers,
            reference=self.reference,
            _is_folder=self._is_folder,
            load_in_memory=should_load,
        )
        if self.load_in_memory and loaded is not None:
            object.__setattr__(new_set, "_in_memory_conformations", loaded)
            object.__setattr__(new_set, "load_in_memory", True)
        return new_set

    def _load_conformation(self, ref: ConformerRef) -> Any:
        try:
            import mdtraj as md  # type: ignore
        except ImportError as e:  # pragma: no cover - import error path
            msg = (
                "mdtraj is required to load conformations in memory. "
                "Install it with `pip install mdtraj`."
            )
            raise ImportError(msg) from e

        if ref.frame is None:
            return md.load(str(ref.path))

        if ref.topology is None:
            raise ValueError(
                f"ConformerRef with trajectory {ref.path} (frame {ref.frame}) is missing a topology."
            )
        traj = _get_trajectory(
            traj_path=ref.path,
            topo_path=ref.topology,
            cache=self._trajectory_cache,
        )
        return traj[ref.frame]


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


def _resolve_filename(name: str, counts: dict[str, int]) -> str:
    """Return a non-overwriting filename using per-name counts."""
    count = counts.get(name, 0)
    if count == 0:
        resolved = name
    else:
        if "." in name:
            base, ext = name.rsplit(".", 1)
            resolved = f"{base}_{count}.{ext}"
        else:
            resolved = f"{name}_{count}"
    counts[name] = count + 1
    return resolved


def _export_static(ref: ConformerRef, output_dir: Path, counts: dict[str, int]) -> None:
    """Copy a static conformer path to output with collision-safe naming."""
    out_name = ref.path.name
    out_final = _resolve_filename(out_name, counts)
    out_path = output_dir / out_final
    shutil.copy2(ref.path, out_path)


def _get_trajectory(
    traj_path: Path,
    topo_path: Path,
    cache: dict[tuple[Path, Path], Any],
) -> Any:
    """Load and cache a trajectory for a trajectory/topology pair."""
    traj_key = (traj_path, topo_path)
    if traj_key not in cache:
        try:
            import mdtraj as md  # type: ignore
        except ImportError as e:  # pragma: no cover - import error path
            msg = (
                "mdtraj is required to export trajectory frames to PDB. "
                "Install it with `pip install mdtraj`."
            )
            raise ImportError(msg) from e
        cache[traj_key] = md.load(str(traj_path), top=str(topo_path))
    return cache[traj_key]


def _export_frame(
    ref: ConformerRef,
    output_dir: Path,
    counts: dict[str, int],
    traj_cache: dict[tuple[Path, Path], Any],
) -> None:
    """Export one trajectory frame as a single-frame PDB file."""
    if ref.topology is None:
        raise ValueError(
            f"ConformerRef with trajectory {ref.path} (frame {ref.frame}) is missing a topology."
        )
    traj_path = ref.path
    topo_path = ref.topology
    traj = _get_trajectory(traj_path=traj_path, topo_path=topo_path, cache=traj_cache)

    frame_idx = ref.frame
    traj_basename = traj_path.stem
    out_name = f"{traj_basename}_{frame_idx}.pdb"
    out_final = _resolve_filename(out_name, counts)
    out_path = output_dir / out_final
    traj[frame_idx].save_pdb(str(out_path))
