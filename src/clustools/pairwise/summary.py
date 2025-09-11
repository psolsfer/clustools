"""Summary statistics and analysis of pairwise clustering comparisons."""

import numpy as np
from numpy.typing import NDArray


def ari_summary(ari_matrix: NDArray[np.floating]) -> dict[str, float]:
    """Summarize pairwise Adjusted Rand Index (ARI) scores.

    Computes descriptive statistics of off-diagonal entries in a pairwise ARI matrix, including
    mean, minimum, maximum, and standard deviation.

    Parameters
    ----------
    ari_matrix : ndarray of shape (n_results, n_results)
        Symmetric matrix of pairwise ARI scores between clustering results.

    Returns
    -------
    summary : dict of str to float
        Dictionary with the following keys:
        - "mean": Mean ARI score
        - "min": Minimum ARI score
        - "max": Maximum ARI score
        - "std": Standard deviation of ARI scores
    """
    off_diag = ari_matrix[~np.eye(ari_matrix.shape[0], dtype=bool)]

    return (
        {
            "mean": float(np.mean(off_diag)),
            "min": float(np.min(off_diag)),
            "max": float(np.max(off_diag)),
            "std": float(np.std(off_diag)),
        }
        if len(off_diag) > 0
        else {"mean": 1.0, "min": 1.0, "max": 1.0, "std": 0.0}
    )
