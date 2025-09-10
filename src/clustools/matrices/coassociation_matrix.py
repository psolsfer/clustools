"""Compute a co-association matrix."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def coassociation_matrix(
    labels: NDArray[np.integer],
    ignore_noise: bool = True,
    noise_labels: Sequence[int] = (-1,),
) -> NDArray[np.floating]:
    """Compute a co-association (pairwise agreement) matrix from multiple labelings.

    Parameters
    ----------
    labels : NDArray[np.integer]
        Array of shape (n_labelings, n_samples), where each row contains a categorical labeling
        (e.g., from a clustering algorithm).
    ignore_noise : bool, default=True
        Whether to exclude noise points from contributing to co-associations.
        If True, samples with labels in ``noise_labels`` are ignored in each labeling. If False,
        noise is treated as just another label.
    noise_labels : sequence of int, default=(-1,)
        Label(s) to be treated as noise when ``ignore_noise=True``.

    Returns
    -------
    NDArray[np.floating]
        Co-association matrix of shape (n_samples, n_samples), where entry (i, j) is the fraction of
        labelings in which samples i and j share the same label.

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array([
    ... [0, 1, -1, -1],
    ... [0, 1, 2, -1],
    ... [0, 1, 1, -1],
    ... ])
    >>> C = coassociation_matrix(labels, ignore_noise=True, noise_labels=[-1])

    Notes
    -----
    This function is commonly used in clustering ensemble methods, but can be
    applied to any categorical labelings to measure pairwise agreement.
    """
    n_samples = labels.shape[1]
    coassoc = np.zeros((n_samples, n_samples), dtype=float)
    n_valid = np.zeros((n_samples, n_samples), dtype=int)

    for clusterer_labels in labels:
        if ignore_noise:
            # Mark valid points (not noise)
            valid_mask = ~np.isin(clusterer_labels, noise_labels)
            valid_labels = clusterer_labels[valid_mask]

            # Pairwise equality for valid samples only
            binary = valid_labels[:, None] == valid_labels[None, :]

            # Expand back to full matrix
            valid_pairs = np.outer(valid_mask, valid_mask)
            coassoc[valid_pairs] += binary.ravel()
            n_valid[valid_pairs] += 1

        else:  # treat noise as a cluster
            binary = clusterer_labels[:, None] == clusterer_labels[None, :]
            coassoc += binary
            n_valid += 1

    # Normalize by number of valid comparisons
    with np.errstate(divide="ignore", invalid="ignore"):
        coassoc = np.divide(
            coassoc, n_valid, out=np.zeros_like(coassoc, dtype=float), where=n_valid > 0
        )

    return coassoc  # type: ignore[no-any-return]
