'''
StateScape Structure Module
'''

from __future__ import annotations

__all__ = [
    "Structure",
]

try:
    import bioemu
except ImportError:
    bioemu = None

try:
    import colabfold
except ImportError:
    colabfold = None

if bioemu is not None:
    from .bioemu import BioEmu
elif colabfold is not None:
    from .colabfold import ColabFold
else:
    raise ImportError("No structure module found. Please install bioemu or colabfold.")