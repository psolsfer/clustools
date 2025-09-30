"""Functions to handle clustering labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from clustools.utils.constants import NOISY_LABELS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def is_valid_label(
    labels: NDArray[np.int_],
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
) -> NDArray[np.bool_]:
    """Return boolean mask indicating which labels are valid (non-noise).

    Parameters
    ----------
    labels : ndarray of int
        Cluster labels.
    noise_labels : sequence of int, optional
        Labels to treat as noise. Default is [-1].

    Returns
    -------
    ndarray of bool, shape (n_samples,)
        True for valid labels, False for noise.
    """
    if noise_labels is None:
        noise_labels = NOISY_LABELS
    return ~np.isin(labels, noise_labels)


def get_unique_labels(
    labels: NDArray[np.int_], noise_labels: NDArray[np.int_] | Sequence[int] | None = None
) -> NDArray[np.int_]:
    """Return unique non-noise cluster labels.

    Parameters
    ----------
    labels : ndarray of int
        Cluster labels.
    noise_labels : sequence of int, optional
        Labels to treat as noise (excluded from the result).
        Default is [-1].

    Returns
    -------
    ndarray of int, shape (n_unique,)
        Sorted array of unique labels with noise removed.
    """
    if noise_labels is None:
        noise_labels = NOISY_LABELS
    return np.setdiff1d(labels, noise_labels, assume_unique=False)


def filter_noise_labels(
    labels: NDArray[np.int_],
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
) -> NDArray[np.int_]:
    """Filter out noise labels from an array (preserves duplicates).

    Parameters
    ----------
    labels : ndarray of int
        Cluster labels.
    noise_labels : sequence of int, optional
        Labels to treat as noise (removed from the result).
        Default is [-1].

    Returns
    -------
    ndarray of int, shape (n_samples_non_noise,)
        Array of labels with noise removed, original order preserved.
    """
    mask = is_valid_label(labels, noise_labels)
    return labels[mask]


def filter_noisy_aligned(
    *labels: NDArray[np.int_],
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
) -> tuple[NDArray[np.int_], ...]:
    """Remove samples where any of several aligned label arrays contains noise.

    Parameters
    ----------
    *labels : sequence of ndarray of int
        One or more aligned label arrays (same length).
    noise_labels : sequence of int, optional
        Labels considered as noise. Default is [-1].

    Returns
    -------
    tuple of ndarrays
        The input arrays with noisy entries removed, preserving alignment.
    """
    # Start with "all valid"
    mask = np.ones_like(labels[0], dtype=bool)

    for arr in labels:
        mask &= is_valid_label(arr, noise_labels)

    return tuple(arr[mask] for arr in labels)
