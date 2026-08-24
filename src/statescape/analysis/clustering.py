import numpy as np

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def _representatives(
    data: np.ndarray,
    labels: np.ndarray
) -> dict[int, int]:
    """for each cluster, return the frame index closest to the centroid."""
    reps = {}
    for cid in np.unique(labels):
        mask = labels == cid
        cluster_pts = data[mask]
        centroid = cluster_pts.mean(axis=0) # geometric centre
        local_idx = np.argmin(np.linalg.norm(cluster_pts - centroid, axis=1)) # which point is closest to centre
        reps[int(cid)] = int(np.where(mask)[0][local_idx])
    return reps

def kmeans(
    coords: np.ndarray,
    n_clusters: int,
    *,
    random_state: int | None = None
) -> tuple[np.ndarray, dict[int, int]]:
    """K-means++ clustering over all columns of `coords`"""
    #data = coords[:, :n_components] # shape (n_frames, n_components)
    labels = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, random_state=random_state).fit_predict(coords)
    return labels, _representatives(coords, labels)

def gmm(
    coords: np.ndarray,
    n_clusters: int,
    *,
    random_state: int | None = None
) -> tuple[np.ndarray, dict[int, int]]:
    """Gaussian Mixture Model clustering over all columns of `coords`"""
    from sklearn.mixture import GaussianMixture
    #data = coords[:, :n_components]
    labels = GaussianMixture(n_components=n_clusters, n_init=10, random_state=random_state).fit_predict(coords)
    return labels, _representatives(coords, labels)

def regular_space(
    coords: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Greedy regular-space clustering over all columns of `coords`.
    A new center is created whenever no existing center lies within `radius`

    Notes
    ------
    The result depends on the frame order: centres grown sequentially from frame 0,
    so a reordered or strided `coords` gives different (altough equally valid) set of
    centers.
    """
    data = coords
    center_idx = [0]
    centers = [data[0]]

    for i in range(1, len(data)):
        dist = np.linalg.norm(data[i] - np.array(centers), axis=1)
        if dist.min() > radius:
            center_idx.append(i)
            centers.append(data[i])
    
    labels = np.argmin(cdist(data, np.array(centers)), axis=1)
    return labels, _representatives(data, labels)