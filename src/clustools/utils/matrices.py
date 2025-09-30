"""Utility functions for matrices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def normalize_matrix(
    matrix: NDArray[np.floating],
    method: Literal["minmax", "z-score", "max", "l2", "elementwise"] = "minmax",
    weights: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Normalize matrix using various normalization methods.

    Parameters
    ----------
    matrix : NDArray
        Input matrix
    method : {"minmax", "z-score", "max", "l2"}, default="min-max"
        Normalization method:
        - "min-max": Scale to [0, 1] range
        - "z-score": Standardize to mean=0, std=1
        - "max": Divide by maximum value (assumes min=0)
        - "l2": L2 normalization (scale so ||matrix||_2 = 1)
    weights : NDArray, optional
        Weight matrix for elementwise normalization (must match matrix shape)

    Returns
    -------
    NDArray
        Normalized matrix
    """
    if method == "minmax":
        min_val = matrix.min()
        max_val = matrix.max()

        if max_val == min_val:
            # All values are the same; return zeros or handle as needed
            normalized = np.zeros_like(matrix)
        else:
            normalized = (matrix - min_val) / (max_val - min_val)

    elif method == "z-score":
        mean = matrix.mean()
        std = matrix.std()
        normalized = np.zeros_like(matrix) if std == 0 else (matrix - mean) / std

    elif method == "max":
        max_val = matrix.max()
        normalized = matrix if max_val == 0 else matrix / max_val

    elif method == "l2":
        norm = np.linalg.norm(matrix)
        normalized = matrix if norm == 0 else matrix / norm

    elif method == "elementwise":
        if weights is None:
            msg = "weights parameter required for elementwise normalization"
            raise ValueError(msg)
        if weights.shape != matrix.shape:
            msg = f"weights shape {weights.shape} must match matrix shape {matrix.shape}"
            raise ValueError(msg)

        normalized = np.zeros_like(matrix, dtype=float)
        mask = weights > 0
        normalized[mask] = matrix[mask] / weights[mask]

    return normalized
