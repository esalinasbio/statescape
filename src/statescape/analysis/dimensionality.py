import numpy as np

from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ReductionResult:
    coords: np.ndarray                    # (n_frames, n_components) <- the latent space
    model: object                         # fitted model (PCA, UMAP, etc.)
    scaler: StandardScaler | None         # fitted scaler if used
    explained_variance: np.ndarray | None # Only meaningfull for PCA

def pca(
    X: np.ndarray,
    n_components: int = 2,
    *,
    scale: bool = False  # mean-center only by default
) -> ReductionResult:
    """Performs PCA over any feature matrix X"""
    scaler = StandardScaler(with_mean=True, with_std=scale)
    X_scaled = scaler.fit_transform(X)
    model = PCA(n_components=n_components)
    coords = model.fit_transform(X_scaled)
    return ReductionResult(
        coords=coords,
        explained_variance=model.explained_variance_ratio_ * 100,
        model = model,
        scaler = scaler
    )

def umap(
    X: np.ndarray,
    n_components: int = 2,
    *,
    scale: bool = True,
    **umap_kwargs,
) -> ReductionResult:
    """UMAP over any feature matrix X."""
    try:
        import umap as umap_learn
    except ImportError as e:
        raise ImportError("umap-learn is required. Install with `pip install umap-learn`.") from e

    scaler = None
    X_scaled = X
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
    model = umap_learn.UMAP(n_components=n_components, **umap_kwargs)
    coords = model.fit_transform(X_scaled)
    return ReductionResult(coords=coords, explained_variance=None, model=model, scaler=scaler)

#### TO DO #####
# Implement tICA
