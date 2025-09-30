"""Functions to handle clustering labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
    noise_labels : sequence of int or array-like, optional (default=None)
        Label(s) to treat as noise.
        If ``None``, no labels are excluded (all labels are returned)

    Returns
    -------
    ndarray of bool, shape (n_samples,)
        True for valid labels, False for noise.
    """
    if noise_labels is None:
        return np.ones_like(labels, dtype=bool)
    return ~np.isin(labels, noise_labels)


def get_unique_labels(
    labels: NDArray[np.int_], noise_labels: NDArray[np.int_] | Sequence[int] | None = None
) -> NDArray[np.int_]:
    """Return unique non-noise cluster labels.

    Parameters
    ----------
    labels : ndarray of int
        Cluster labels.
    noise_labels : sequence of int or array-like, optional (default=None)
        Label(s) to treat as noise and exclude from the result.
        If ``None``, no labels are excluded (all labels are returned).

    Returns
    -------
    ndarray of int, shape (n_unique,)
        Sorted array of unique labels with noise removed.
    """
    if noise_labels is None:
        noise_labels = np.array([])
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
    noise_labels : sequence of int or array-like, optional (default=None)
        Label(s) to treat as noise and filter from the result.
        If ``None``, no labels are filtered (all labels are returned)

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
    noise_labels : sequence of int or array-like, optional (default=None)
        Label(s) to treat as noise and filter from the result.
        If ``None``, no labels are filtered (all labels are returned)

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
