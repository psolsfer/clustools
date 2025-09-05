"""Functions to handle clustering labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def get_unique_labels(
    labels: NDArray[np.int_], noise_labels: NDArray[np.int_] | Sequence[int] | None
) -> NDArray[np.int_]:
    """Calculate unique non-noise labels (noise == -1)."""
    if noise_labels is None:
        noise_labels = [-1]
    return np.setdiff1d(labels, noise_labels, assume_unique=False)


def filter_noise_labels(
    labels: NDArray[np.int_],
    noise_labels: NDArray[np.int_] | Sequence[int] | None = None,
) -> NDArray[np.int_]:
    """Remove noise labels from an array of labels (does not uniquify)."""
    if noise_labels is None:
        noise_labels = [-1]
    return labels[~np.isin(labels, noise_labels)]
