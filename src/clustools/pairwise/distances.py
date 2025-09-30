"""Distance-based comparison matrices between clustering results.

This module provides functions to compute pairwise distance or cost matrices between clusterings,
such as overlap-based costs derived from the Jaccard index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.cluster import contingency_matrix

from clustools.utils.labels import filter_noisy_aligned

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def overlap_distance_matrix(
    base_labels: NDArray[np.int_],
    current_labels: NDArray[np.int_],
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
) -> NDArray[np.floating]:
    """Compute an overlap-based distance matrix between two clusterings.

    Each entry (i, j) is defined as 1 - Jaccard(base_cluster_i, current_cluster_j).

    Parameters
    ----------
    base_labels : ndarray of int, shape (n_samples,)
        Cluster labels from the base clustering.
    current_labels : ndarray of int, shape (n_samples,)
        Cluster labels from the current clustering.
    noise_labels : sequence of int, optional
        Labels considered as noise and ignored. Default is [-1].

    Returns
    -------
    ndarray of float, shape (n_base_clusters, n_current_clusters)
        Distance matrix where lower values indicate stronger overlap.
    """
    # Remove samples where either label is noise
    base_labels, current_labels = filter_noisy_aligned(
        base_labels, current_labels, noise_labels=noise_labels
    )

    # intersection[i, j] = number of samples in both base_label_i and current_label_j
    intersection = contingency_matrix(base_labels, current_labels, sparse=False)

    row_sum = intersection.sum(axis=1, keepdims=True)
    col_sum = intersection.sum(axis=0, keepdims=True)
    union = row_sum + col_sum - intersection
    overlap = np.divide(intersection, union, where=union > 0)
    # Convert to cost (lower is better)
    cost_matrix = 1.0 - overlap

    return cost_matrix  # type: ignore[no-any-return]
