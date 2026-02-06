# data contracts for conformers

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any, Mapping, Sequence, Tuple, Optional


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

    def __len__(self) -> int:
        return len(self.conformers)

    def __iter__(self):
        return iter(self.conformers)

    def __getitem__(self, index: int) -> ConformerRef:
        return self.conformers[index]

    def __contains__(self, item: ConformerRef) -> bool:
        return item in self.conformers
