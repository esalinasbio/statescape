import numpy as np

from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ReductionResult:
    coords: np.ndarray                    # (n_frames, n_components) <- the latent space
    model: object                         # fitted model (PCA, UMAP, etc.)
    scaler: StandardScaler | None         # fitted scaler if used
    explained_variance: np.ndarray | None # only meaningfull for PCA

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

def tica(
    blocks: list[np.ndarray],
    n_components: int = 2,
    *,
    lag: int,
    scale: bool = False,
    **tica_kwargs
) -> ReductionResult:
    """
    Time-lagged independent component analysis over a list of trajectories' features.
    Each block is treated as an independent trajectory.

    Parameters
    ----------
    blocks: one (n_frames, n_features) array per trajectory
    n_components: number of independent components to keep
    lag: lag time in frames
    scale: standarize features to unit variance before fitting. Off by default,
        tICA is already scale invariant through its own covariance normalization,
        and scaling can amplify near constant features
    **tica_kwargs: forwarded to `deeptime.decomposition.TICA`

    Returns
    -------
    ReductionResult: `.coords` is the projection of every block, concatenated.
        `.model` is the fitted deeptime model. `.model.timescales(lagtime=lag)` gives
        the implied timescales in frames.
    """
    try:
        from deeptime.decomposition import TICA
    except ImportError as e:
        raise ImportError(f'deeptime is requieres. Install with `pip install deeptime`.')

    if not  blocks:
        raise ValueError('No features given')
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}.")

    short = [(i, len(b)) for i, b in enumerate(blocks) if len(b) <= lag]
    if short:
        raise ValueError(f"lag={lag} is greater than {len(short)} of {len(blocks)} trajectories.")

    scaler = None
    data = blocks
    if scale:
        scaler = StandardScaler()
        scaler.fit(np.vstack(blocks))
        data = [scaler.transform(b) for b in blocks]

    model = TICA(lagtime=lag, dim=n_components, **tica_kwargs).fit(data).fetch_model()
    coords = np.vstack([model.transform(b) for b in data])

    return ReductionResult(coords=coords, explained_variance=None, model=model, scaler=scaler)
