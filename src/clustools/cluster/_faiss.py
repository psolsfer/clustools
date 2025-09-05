from numbers import Integral
from typing import Any, ClassVar, Self

import faiss
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyauxlib.decorators.import_errors import require_class
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform
from sklearn.base import _fit_context
from sklearn.cluster import KMeans
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from clustools.cluster.custom_cluster_mixin import CustomClusterMixin
from clustools.common.cluster_centers import assign_closer_cluster, compute_cluster_centers
from clustools.common.inertia import compute_inertia

# ruff: noqa: N803, N806, ERA001


class FaissClusterMixin(CustomClusterMixin):
    """Base class for FAISS-based clustering algorithms.

    Provides common serialization handling for FAISS clusterers.
    """

    def __getstate__(self) -> dict[str, Any]:
        """Serialize to handle FAISS objects."""
        state = self.__dict__.copy()
        # Remove the FAISS instance which can't be pickled
        if "clusterer_instance_" in state:
            del state["clusterer_instance_"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)


@require_class("faiss")
class FaissKMeans(FaissClusterMixin, CustomClusterMixin):
    """K-means clustering using the FAISS library.

    This class is compatible with scikit-learn's API and with the
    :py:method:`~pydataset.analysis.cluster.ClusterEngine.fit` from
    :py:class:`~pydataset.analysis.cluster.ClusterEngine`.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms of
        inertia (sum of squared distances to the closest cluster center).
    verbose : bool, default=False
        Whether to print progress messages during training.
    random_state: int, default=0
        Determines random number generation for centroid initialization. Pass an integer for
        reproducible results across multiple runs.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully converging (see tol and
        max_iter), these may not align perfectly with `labels_`.
    labels_ : ndarray of shape (n_samples,)
        Cluster index (label) assigned to each data point
    unique_labels_ : ndarray of shape (n_unique_clusters,)
        Unique labels
    counts_ : ndarray of shape (n_unique_clusters,)
        Number of samples assigned to each unique cluster label.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    clusterer_instance_ : faiss.Kmeans
        The internal FAISS KMeans instance used for clustering. This won't be serialized.
    name : str
        Name of the clustering algorithm (`"FaissKMeans"`).
    """

    name = "FaissKMeans"

    # Constraints checked by _validate_params
    _parameter_constraints: ClassVar[dict[str, Any]] = {
        **CustomClusterMixin._parameter_constraints,  # noqa: SLF001
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "verbose": ["boolean"],
    }

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        n_init: int = 10,
        verbose: bool = False,
        random_state: int | None = 0,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Compute the clustering.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Training instances to cluster.
            Noted that the data will be converted to C ordering, which will cause a memory copy if
            the given data is not C-contiguous.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X_val: NDArray[np.floating[Any]] = validate_data(self, X, dtype="numeric")

        d = X_val.shape[1]
        kmeans = faiss.Kmeans(
            d,
            self.n_clusters,
            niter=self.n_init,
            verbose=self.verbose,
            seed=self.random_state,
        )
        kmeans.train(X_val)
        self.cluster_centers_ = kmeans.centroids
        D, IDX = kmeans.index.search(X_val, 1)
        self.labels_ = IDX.reshape(-1)
        self.inertia_ = np.sum(D)
        self.unique_labels_, self.counts_ = np.unique(self.labels_, return_counts=True)
        self.clusterer_instance_ = kmeans
        self._n_threads = _openmp_effective_n_threads()
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int_]:
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        X_val: NDArray[np.floating[Any]] = validate_data(self, X, dtype="numeric", reset=False)

        return assign_closer_cluster(
            X=X_val, centers=self.cluster_centers_, n_threads=self._n_threads
        )

    # def __sklearn_tags__(self):
    #     tags = super().__sklearn_tags__()
    #     tags.input_tags.sparse = False
    #     return tags


@require_class("faiss")
class FaissLSH(FaissClusterMixin, CustomClusterMixin):
    """LSH-based clustering using the FAISS library.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    nbits : int, default=8
        Number of bits per code (2^nbits = number of centroids).
    random_state: int, default=0
        Random seed

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    unique_labels_ : ndarray of shape (n_unique_clusters,)
        Unique labels
    counts_ : ndarray of shape (n_unique_clusters,)
        Number of samples assigned to each unique cluster label.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    clusterer_instance_ : faiss.Kmeans
        The internal faiss.IndexLSH instance. This won't be serialized.
    name : str
        Name of the clustering algorithm.
    """

    name = "FaissLSH"

    # Constraints checked by _validate_params
    _parameter_constraints: ClassVar[dict[str, Any]] = {
        **CustomClusterMixin._parameter_constraints,  # noqa: SLF001
        "nbits": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, n_clusters: int = 8, *, nbits: int = 8, random_state: int = 0) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.nbits = nbits
        self.random_state = random_state  # ??? Not used

    def fit(self, X: ArrayLike, y: Any = None) -> Self:
        """Compute the clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.
            Noted that the data will be converted to C ordering, which will cause a
            memory copy if the given data is not C-contiguous.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X_val: NDArray[np.floating[Any]] = validate_data(self, X, dtype="numeric")

        d = X_val.shape[1]
        index = faiss.IndexLSH(d, self.nbits)
        index.train(X_val)
        index.add(X_val)

        # Assign data points to clusters based on their nearest LSH bucket
        _, IDX = index.search(X_val, 1)
        lsh_labels = IDX.reshape(-1)

        # Compute initial cluster centers
        initial_centers = compute_cluster_centers(X_val, lsh_labels)
        num_found_clusters = initial_centers.shape[0]

        if num_found_clusters > self.n_clusters:
            # Merge clusters if we have too many
            distances = cdist(initial_centers, initial_centers)
            # Convert to condensed form before hierarchical clustering
            condensed_distances = squareform(distances)
            Z = linkage(condensed_distances, method="ward")  # Hierarchical clustering
            new_labels = fcluster(Z, self.n_clusters, criterion="maxclust")
            centers = []
            for i in range(1, self.n_clusters + 1):
                cluster_mask = new_labels == i
                if np.any(cluster_mask):  # Only take non-empty clusters
                    centers.append(initial_centers[cluster_mask].mean(axis=0))

            cluster_centers = np.array(centers)

        elif num_found_clusters < self.n_clusters:
            # NOTE The commented code is an alternate version w/o using KMeans
            # Handle case if some clusters are empty by initializing with zeros
            # cluster_centers = np.zeros((self.n_clusters, d))
            # cluster_centers[: len(initial_centers)] = initial_centers

            # If too few clusters, split largest ones
            centers = list(initial_centers)
            while len(centers) < self.n_clusters:
                largest_cluster_idx = np.argmax(np.bincount(lsh_labels))
                mask = lsh_labels == largest_cluster_idx
                if np.sum(mask) < 2:  # Stop if splitting is impossible  # noqa: PLR2004
                    break
                new_sub_clusters = KMeans(n_clusters=2, n_init=5).fit(X_val[mask]).cluster_centers_
                centers = [c for i, c in enumerate(centers) if i != largest_cluster_idx]
                centers.extend(new_sub_clusters)
            cluster_centers = np.array(centers[: self.n_clusters])  # Ensure exact count

        else:
            cluster_centers = initial_centers

        self._n_threads = _openmp_effective_n_threads()

        # Assign data points to clusters based on their nearest cluster center
        self.labels_ = assign_closer_cluster(
            X=X_val,
            centers=cluster_centers,
            n_threads=self._n_threads,
        )

        # Recompute final cluster centers
        self.cluster_centers_ = compute_cluster_centers(X_val, self.labels_)

        # Ensure labels match final cluster centers
        self.labels_ = assign_closer_cluster(
            X=X_val,
            centers=self.cluster_centers_,
            n_threads=self._n_threads,
        )

        # Compute inertia as the sum of squared distances to cluster centers
        self.inertia_ = compute_inertia(X_val, self.labels_, self.cluster_centers_)

        self.unique_labels_, self.counts_ = np.unique(self.labels_, return_counts=True)

        self.clusterer_instance_ = index
        return self

    # def __sklearn_tags__(self):
    #     tags = super().__sklearn_tags__()
    #     tags.input_tags.positive_only = True
    #     return tags
