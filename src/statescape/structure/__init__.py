"""
StateScape Structure Module — ensemble generation backends.

Not yet implemented. See `structure/bioemu.py` and `structure/colabfold.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bioemu import BioEmu
    from .colabfold import ColabFold

__all__ = [
    "BioEmu", "ColabFold"
]

def __getattr__(name: str):
    if name == "BioEmu":
        from .bioemu import BioEmu
        return BioEmu
    if name == "ColabFold":
        from .colabfold import ColabFold
        return ColabFold
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")