"""Internal clustering validation metrics (no ground truth required)."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ruff: noqa: N803, N806


def compute_inertia(X: ArrayLike, labels: ArrayLike, cluster_centers: ArrayLike) -> float:
    """Compute the inertia of samples to their assigned cluster centers.

    This follows scikit-learn's convention for inertia (sum of squared distances):
    ``inertia_ = sum(||x_i - c_{label_i}||^2 for i in samples)``

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    labels : array-like of shape (n_samples,)
        Cluster labels for each sample. Each entry should be an integer index into
        `cluster_centers`.
    cluster_centers : array-like of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    Returns
    -------
    inertia : float
        The sum of squared distances of each sample to its assigned cluster center.
    """
    X = np.asarray(X)
    labels = np.asarray(labels, dtype=int)
    cluster_centers = np.asarray(cluster_centers)

    diffs: NDArray[np.float64] = X - cluster_centers[labels]
    return float(np.sum(np.sum(diffs**2, axis=1)))
