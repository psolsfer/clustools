"""Base clase for the custom clusterers."""

from numbers import Integral
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

# ruff: noqa: ERA001


# ISSUE #49
class CustomClusterMixin(ClusterMixin, BaseEstimator):
    """Mixin class providing common functionality for custom clustering algorithm wrappers.

    Inherits from ClusterMixin and BaseEstimator for compatibility with scikit-learn.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    unique_counts_ : tuple of two ndarrays
        Unique labels and their counts.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    clusterer_instance_ : Any
        Instance of the clusterer after calling fit. Not all clusterers will have this.
    """

    name = ""

    # Constraints checked by _validate_params
    _parameter_constraints: ClassVar[dict[str, Any]] = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "random_state": [Interval(Integral, 0, None, closed="left"), None],  # ["random_state"],
    }

    @property
    def clusters(self) -> NDArray[np.integer] | None:
        """Return the labels of each point."""
        check_is_fitted(self, "labels_")
        return self.labels_  # type: ignore[no-any-return]

    @property
    def unique(self) -> NDArray[np.integer] | None:
        """Return the unique labels."""
        check_is_fitted(self, "unique_counts_")
        return self.unique_counts_[0] if self.unique_counts_ is not None else None

    @property
    def counts(self) -> NDArray[np.integer] | None:
        """Return the count of instances for each unique label."""
        check_is_fitted(self, "unique_counts_")
        return self.unique_counts_[1] if self.unique_counts_ is not None else None

    # def fit(self, X, y=None) -> Self:
    #     """Placeholder fit method to be overridden by subclasses."""
    #     msg = "Subclasses must implement fit()."
    #     raise NotImplementedError(msg)
