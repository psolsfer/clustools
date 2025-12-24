"""Pairwise agreement and similarity matrices between clustering results."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score, pairwise_distances

from clustools.utils.labels import is_valid_label
from clustools.utils.matrices import normalize_matrix

# ruff: noqa: ERA001


def _input_conversion(
    cluster_results: NDArray[np.integer] | Sequence[NDArray[np.integer]],
) -> NDArray[np.integer]:
    # Validate and convert input
    if isinstance(cluster_results, np.ndarray):
        if cluster_results.ndim != 2:  # noqa: PLR2004
            msg = "Array must have shape (n_labelings, n_samples)."
            raise ValueError(msg)
        return cluster_results

    cluster_results = list(cluster_results)
    # Check consistency
    n_samples = len(cluster_results[0])
    if not all(len(arr) == n_samples for arr in cluster_results):
        msg = "All labelings must have the same length."
        raise ValueError(msg)
    return np.vstack(cluster_results)  # shape (n_labelings, n_samples)


def coassociation_matrix(
    labelings: NDArray[np.integer] | Sequence[NDArray[np.integer]],
    noise_labels: Sequence[int] | None = None,
) -> NDArray[np.floating]:
    """Compute a co-association (pairwise agreement) matrix from multiple labelings.

    Parameters
    ----------
    labelings : NDArray[np.integer] | Sequence[NDArray[np.integer]]
        Multiple cluster labelings of the same set of samples.
            Accepted formats are:
            - 2D array of shape (n_labelings, n_samples), where each row is a labeling.
            - Sequence of 1D arrays, each of length n_samples.
            In both cases, each labeling contains integer cluster assignments.
    ignore_noise : bool, default=True
        Whether to exclude noise points from contributing to co-associations.
        If True, samples with labels in ``noise_labels`` are ignored in each labeling. If False,
        noise is treated as just another label.
    noise_labels : sequence of int, default=(-1,)
        Label(s) to be excluded from contributing to the co-association matrix.
        Typically used to ignore noise or outlier labels (e.g., ``-1`` from DBSCAN).
        If ``None``, no labels are excluded and all samples are included in the computation.

    Returns
    -------
    co-association matrix : ndarray of shape (n_results, n_results)
        Co-association matrix of shape (n_samples, n_samples), where entry (i, j) is the fraction of
        labelings in which samples i and j share the same label.

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array(
    ...     [
    ...         [0, 1, -1, -1],
    ...         [0, 1, 2, -1],
    ...         [0, 1, 1, -1],
    ...     ]
    ... )
    >>> C = coassociation_matrix(labels, noise_labels=[-1])

    Notes
    -----
    This function is commonly used in clustering ensemble methods, but can be applied to any
    categorical labelings to measure pairwise agreement.
    """
    labelings = _input_conversion(labelings)
    n_samples = labelings.shape[1]
    coassoc_matrix = np.zeros((n_samples, n_samples), dtype=float)
    n_valid = np.zeros((n_samples, n_samples), dtype=int)

    for clusterer_labels in labelings:
        if noise_labels is not None:
            # Mark valid points (not noise)
            valid_mask = is_valid_label(clusterer_labels, noise_labels)
            valid_labels = clusterer_labels[valid_mask]

            # Pairwise equality for valid samples only
            binary = valid_labels[:, None] == valid_labels[None, :]

            # Expand back to full matrix
            valid_pairs = np.outer(valid_mask, valid_mask)
            coassoc_matrix[valid_pairs] += binary.ravel()
            n_valid[valid_pairs] += 1

        else:  # treat noise as a cluster
            binary = clusterer_labels[:, None] == clusterer_labels[None, :]
            coassoc_matrix += binary
            n_valid += 1

    # Normalize by number of valid comparisons
    coassoc_matrix = normalize_matrix(coassoc_matrix, method="elementwise", weights=n_valid)

    return coassoc_matrix


def pairwise_ari_matrix(
    labelings: NDArray[np.integer] | Sequence[NDArray[np.integer]],
) -> NDArray[np.floating]:
    """Compute pairwise Adjusted Rand Index (ARI) scores between clustering results.

    This function computes a symmetric similarity matrix where each entry (i, j) corresponds to the
    ARI score between the i-th and j-th clustering in the input list.

    Parameters
    ----------
    labelings : NDArray[np.integer] | Sequence[NDArray[np.integer]]
        Multiple cluster labelings of the same set of samples.
        Accepted formats are:
        - 2D array of shape (n_labelings, n_samples), where each row is a labeling.
        - Sequence of 1D arrays, each of length n_samples.
        In both cases, each labeling contains integer cluster assignments.

    Returns
    -------
    ari_matrix : ndarray of shape (n_results, n_results)
        Symmetric matrix of pairwise ARI scores between clustering results.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import adjusted_rand_score
    >>> labelings = np.array(
    ...     [
    ...         [0, 0, 1, 1],
    ...         [1, 1, 0, 0],
    ...         [0, 1, 0, 1],
    ...     ]
    ... )
    >>> pairwise_ari_matrix(labelings)
    array([[ 1. ,  1. , -0.5],
           [ 1. ,  1. , -0.5],
           [-0.5, -0.5,  1. ]])
    """
    labelings = _input_conversion(labelings)

    return pairwise_distances(labelings, metric=adjusted_rand_score)  # type: ignore[no-any-return]
    # Non-sklearn calculation
    # n_labelings = labelings.shape[0]
    # ari_matrix = np.zeros((n_labelings, n_labelings), dtype=float)

    # for i in range(n_labelings):
    #     for j in range(i + 1, n_labelings):
    #         score = adjusted_rand_score(labelings[i], labelings[j])
    #         ari_matrix[i, j] = score
    #         ari_matrix[j, i] = score

    # return ari_matrix
