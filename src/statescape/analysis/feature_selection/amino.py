"""
StateScape wrapper to run automatic feature selection with AMINO

IF you use this module, please cite:

Pavan Ravindra, Zachary Smith and Pratyush Tiwary, Automatic mutual 
information noise omission (AMINO): generating order parameters for 
molecular systems, Mol. Syst. Des. Eng., 2020, 5, 339-348, 
https://doi.org/10.1039/C9ME00115H

Copyright 2024 Tiwary Lab. Licensed under the MIT License
"""


from __future__ import annotations

import numpy as np

from statescape._vendor.amino import AMINO

def select(
    X: np.ndarray,
    labels: list[str],
    *,
    max_op: int = 20,
    bins: int = 50,
    verbose: bool = False
) ->list[str]:
    """
    Run AMINO to select a non-redundant subset of features
    Returns the selected feature labels as a list (a subset of `labels`).

    Parameters
    ----------
    X: feature matrix (n_frames, n_features)
    labels: feature labels
    max_op: maximum number of order parameters (features) to return (default: 20)
    bins: histogram bins for MI estimation (default: 50)
    verbose: print progress (default: False)
    """
    if X.shape[1] != len(labels):
        raise ValueError(f"Feature matrix has {X.shape[1]} features, but {len(labels)} were given.")
    model = AMINO(n=max_op, bins=bins, verbose=verbose)
    model.run(labels, X.T)
    return model.result