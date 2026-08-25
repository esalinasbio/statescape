"""
StateScape Structure Module — ensemble generation backends.

Not yet implemented. See `structure/bioemu.py` and `structure/colabfold.py`.
"""

from __future__ import annotations

__all__ = ["BioEmu", "ColabFold"]

_BACKENDS = {
    "BioEmu": "statescape.structure.bioemu",
    "ColabFold": "statescape.structure.colabfold",
}


def __getattr__(name: str):
    if name in _BACKENDS:
        raise NotImplementedError(
            f"{name} is not implemented yet ({_BACKENDS[name]} is empty). "
            "Generate the ensemble externally and load it with "
            "ConformerSet('path/to/pdbs/')."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")