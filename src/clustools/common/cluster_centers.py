"""Functions to compute cluster centers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster._kmeans import _labels_inertia
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.parallel import _threadpool_controller_decorator

from clustools.common.labels import filter_noise_labels, get_unique_labels

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

# ruff: noqa: N803, N806


def compute_cluster_centers(
    X: ArrayLike,
    labels: NDArray[np.int_],
    method: Literal["mean", "median", "medoid"] = "mean",
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
    unique_labels: NDArray[np.int_] | None = None,
) -> NDArray[np.floating]:
    """Compute cluster centers from data and labels.

    Parameters
    ----------
    X : ArrayLike
        Input data array of shape (n_samples, n_features).
    labels : NDArray[np.int_]
        Cluster labels for each sample.
    method : {"mean", "median", "medoid"}, default="mean"
        Method for computing cluster centers:
        - "mean": Arithmetic mean
        - "median": Component-wise median
        - "medoid": Actual data point closest to center
    noise_labels : list[int] or NDArray[np.int_], optional
        Labels to treat as noise (ignored). Default is [-1].
    unique_labels : NDArray[np.int_], optional
        Precomputed unique labels. If None, will compute them from `labels`.

    Returns
    -------
    centers : NDArray[np.floating]
        Array of cluster centers of shape (n_clusters, n_features).
    """
    if method not in ["mean", "median", "medoid"]:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)
    X = np.asarray(X)

    if unique_labels is None:
        unique_labels = get_unique_labels(labels, noise_labels)
    else:
        unique_labels = filter_noise_labels(unique_labels, noise_labels)

    # Fast path for "mean"
    if method == "mean":
        centers = np.array(
            [X[labels == label].mean(axis=0) for label in unique_labels], dtype=float
        )

        return centers

    # General loop for median / medoid
    n_features = X.shape[1]
    centers = np.zeros((len(unique_labels), n_features), dtype=float)

    for i, label in enumerate(unique_labels):
        cluster_mask = labels == label
        cluster_X = X[cluster_mask]

        if cluster_X.size == 0:
            continue

        if method == "median":
            centers[i] = np.median(cluster_X, axis=0)
        elif method == "medoid":
            # Find actual data point closest to mean (medoid)
            mean_center = cluster_X.mean(axis=0)
            distances = np.linalg.norm(cluster_X - mean_center, axis=1)
            centers[i] = cluster_X[np.argmin(distances)]

    return centers


def compute_batch_cluster_centers(
    X: ArrayLike,
    cluster_results: list[NDArray[np.int_]],
    method: Literal["mean", "median", "medoid"] = "mean",
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
) -> list[NDArray[np.floating]]:
    """Compute cluster centers for multiple clustering results.

    Applies cluster center computation to multiple clustering results in batch, using the same data
    and center computation method.

    Parameters
    ----------
    X : ArrayLike
        Input data array with shape (n_samples, n_features)
    cluster_results : list[NDArray[np.int_]]
        List of cluster label arrays, one per clustering run
    method : {"mean", "median", "medoid"}, default="mean"
        Method for computing cluster centers:
        - "mean": Arithmetic mean (fast, sensitive to outliers)
        - "median": Component-wise median (robust to outliers)
        - "medoid": Actual data point closest to center (most representative)
    noise_labels : list[int] or NDArray[np.int_], optional
        Labels to treat as noise (ignored). Default is [-1].

    Returns
    -------
    cluster_centers : list of NDArray[np.floating]
        List of cluster centers for each clustering result

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(23)
    >>> data = rng.random((100, 3))
    >>> labels1 = rng.integers(0, 5, 100)
    >>> labels2 = rng.integers(0, 4, 100)
    >>> centers = compute_batch_cluster_centers(data, [labels1, labels2], "medoid")
    >>> len(centers)
    2
    """
    X_array = np.asarray(X)
    cluster_centers: list[NDArray[np.floating]] = []

    for labels in cluster_results:
        centers = compute_cluster_centers(
            X=X_array, labels=labels, method=method, noise_labels=noise_labels
        )
        cluster_centers.append(centers)
    return cluster_centers


def assign_closer_cluster(
    X: NDArray[np.floating[Any]],
    centers: NDArray[np.floating[Any]],
    alg_threshold: int | None = 2000,
    n_threads: int | None = None,
) -> NDArray[np.int_]:
    """Assign points to the closer cluster."""
    if alg_threshold is not None and X.shape[0] > alg_threshold:
        # Use sklearn's faster implementation for large datasets
        return assign_closer_cluster_sklearn(X, centers, n_threads)

    D = cdist(X, centers)
    return np.argmin(D, axis=1)  # type: ignore[no-any-return]


_labels_inertia_threadpool_limit = _threadpool_controller_decorator(limits=1, user_api="blas")(
    _labels_inertia
)


def assign_closer_cluster_sklearn(
    X: NDArray[np.floating[Any]], centers: NDArray[np.floating[Any]], n_threads: int | None = None
) -> NDArray[np.int_]:
    """Assign points to the closer cluster using sklearn's implementation from KMeans."""
    # sample weights are not used by predict but cython helpers expect an array
    sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # Convert dtype of cluster_centers_
    cluster_centers = centers.astype(np.double)

    if n_threads is None:
        n_threads = _openmp_effective_n_threads()

    labels: NDArray[np.int_] = _labels_inertia_threadpool_limit(
        X,
        sample_weight,
        cluster_centers,
        n_threads=n_threads,
        return_inertia=False,
    )

    return labels
