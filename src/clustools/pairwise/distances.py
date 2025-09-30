"""Distance-based comparison matrices between clustering results.

This module provides functions to compute pairwise distance or cost matrices between clusterings,
such as overlap-based costs derived from the Jaccard index.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import contingency_matrix

from clustools.utils.labels import filter_noisy_aligned
from clustools.utils.matrices import normalize_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence


MetricCallable = Callable[
    [NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
]


def overlap_distance_matrix(
    labels_x: NDArray[np.int_],
    labels_y: NDArray[np.int_],
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
    metric: Literal["jaccard", "dice", "overlap"] | MetricCallable = "jaccard",
) -> NDArray[np.floating]:
    """Compute an overlap-based distance matrix between two clusterings.

    Each entry (i, j) is 1 - similarity(cluster_x_i, cluster_y_j).

    Parameters
    ----------
    labels_x : ndarray of int, shape (n_samples,)
        Cluster labels for the first clustering.
    labels_y : ndarray of int, shape (n_samples,)
        Cluster labels for the second clustering.
    noise_labels : sequence of int, optional
        Labels considered as noise and ignored. Default is None.
    metric : {"jaccard", "dice", "overlap"} or callable, default="jaccard"
        Similarity metric to use. If callable, must take (intersection, row_sum, col_sum) and return
        a similarity matrix of shape (n_x_clusters, n_y_clusters).

    Returns
    -------
    ndarray of float, shape (n_base_clusters, n_current_clusters)
        Distance matrix where lower values indicate stronger overlap.
    """
    # Filter noisy points
    if noise_labels is not None:
        labels_x, labels_y = filter_noisy_aligned(labels_x, labels_y, noise_labels=noise_labels)

    # intersection[i, j] = number of samples in both base_label_i and current_label_j
    intersection = contingency_matrix(labels_x, labels_y, sparse=False)

    row_sum = intersection.sum(axis=1, keepdims=True)  # size of each cluster in x
    col_sum = intersection.sum(axis=0, keepdims=True)  # size of each cluster in y

    # Define metric handlers
    if isinstance(metric, str):
        if metric == "jaccard":
            union = row_sum + col_sum - intersection
            sim = np.divide(intersection, union, where=union > 0)
        elif metric == "dice":
            sim = np.divide(2 * intersection, row_sum + col_sum, where=(row_sum + col_sum) > 0)
        elif metric == "overlap":
            sim = np.divide(
                intersection, np.minimum(row_sum, col_sum), where=(np.minimum(row_sum, col_sum) > 0)
            )
    else:
        # Custom callable: (intersection, row_sum, col_sum) -> similarity
        sim = metric(intersection, row_sum, col_sum)

    # Convert similarity → distance (lower is better)
    cost_matrix = 1.0 - sim

    return cost_matrix  # type: ignore[no-any-return]


def cluster_similarity_matrix(  # noqa: PLR0913
    centers_x: NDArray[np.floating] | None = None,
    centers_y: NDArray[np.floating] | None = None,
    labels_x: NDArray[np.int_] | None = None,
    labels_y: NDArray[np.int_] | None = None,
    similarity_method: Literal["center_distance", "overlap", "combined"] = "center_distance",
    distance_metric: str = "euclidean",
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
    overlap_metric: Literal["jaccard", "dice", "overlap"] | MetricCallable = "jaccard",
    center_weight: float | None = None,
    overlap_weight: float | None = None,
) -> NDArray[np.floating]:
    """Compute similarity matrix between cluster sets using various methods.

    Parameters
    ----------
    centers_x, centers_y : ndarray, shape (n_x_clusters, n_features)
        Centers of clusters. Required if similarity_method uses "center_distance".
    labels_x, labels_y : ndarray of int, shape (n_samples,)
        Cluster labels. Required if similarity_method uses "overlap".
    similarity_method : {"center_distance", "overlap", "combined"}, default="center_distance"
        Method for computing similarity:
        - "center_distance": Distance between cluster centers
        - "overlap": Overlap of cluster assignments
        - "combined": Weighted combination of distance and overlap
    distance_metric : str, default="euclidean"
        Distance metric for center distances.
    noise_labels : sequence of int, optional
        Labels to treat as noise/outliers and exclude from similarity computation.
    overlap_metric : {"jaccard", "dice", "overlap"} or callable, default="jaccard"
        Metric for overlap similarity (used if similarity_method="overlap" or "combined").
    center_weight : float, optional
        Weight for center distance when similarity_method="combined". Must be in [0, 1].
    overlap_weight : float, optional
        Weight for overlap when similarity_method="combined". Must be in [0, 1].

    Returns
    -------
    ndarray, shape (n_x_clusters, n_y_clusters)
        Cost matrix where lower values indicate better matches between clusters.
    """
    if similarity_method == "center_distance":
        if centers_x is None or centers_y is None:
            msg = "centers_x and centers_y are required for 'center_distance'"
            raise ValueError(msg)
        return pairwise_distances(centers_x, centers_y, metric=distance_metric)  # type: ignore[no-any-return]

    if similarity_method == "overlap":
        if labels_x is None or labels_y is None:
            msg = "labels_x and labels_y are required for 'overlap'"
            raise ValueError(msg)
        return overlap_distance_matrix(
            labels_x, labels_y, noise_labels=noise_labels, metric=overlap_metric
        )

    if similarity_method == "combined":
        if centers_x is None or centers_y is None or labels_x is None or labels_y is None:
            msg = "centers_x, centers_y, labels_x, and labels_y are all required for 'combined'"
            raise ValueError(msg)
        # Defaults
        center_weight = 0.5 if center_weight is None else center_weight
        overlap_weight = 0.5 if overlap_weight is None else overlap_weight

        # Compute both components
        center_costs = pairwise_distances(centers_x, centers_y, metric=distance_metric)
        overlap_costs = overlap_distance_matrix(
            labels_x, labels_y, noise_labels=noise_labels, metric=overlap_metric
        )

        # Normalize center_costs to [0, 1] (overlap_costs is already normalized)
        center_costs_norm = normalize_matrix(center_costs, method="minmax")

        # Weighted combination
        return center_weight * center_costs_norm + overlap_weight * overlap_costs

    msg = f"Unsupported similarity_method: {similarity_method}"  # type: ignore[unreachable]
    raise ValueError(msg)
