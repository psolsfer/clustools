"""SKATER (Spatial K-luster Analysis by Tree Edge Removal) clustering algorithm."""

from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.base import _fit_context
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import validate_data

from clustools.cluster.custom_cluster_mixin import CustomClusterMixin
from clustools.metrics.internal import compute_inertia

# ruff: noqa: N803, N806


def _scale_data(
    X: NDArray[np.floating[Any]],
    method: Literal["raw", "standardize", "demean", "mad", "range_standardize", "range_adjust"],
) -> NDArray[np.floating[Any]]:
    """Scale data according to the specified method.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data to scale.
    method : str
        Scaling method to apply.

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
        Scaled data.
    """
    if method == "raw":
        return X

    if method == "standardize":
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    if method == "demean":
        return X - X.mean(axis=0)

    if method == "mad":
        # Median Absolute Deviation scaling
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        # Avoid division by zero
        mad = np.where(mad == 0, 1, mad)
        return (X - median) / mad

    if method == "range_standardize":
        # Scale to [0, 1]
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        x_range = x_max - x_min
        x_range = np.where(x_range == 0, 1, x_range)
        return (X - x_min) / x_range

    if method == "range_adjust":
        # Scale to [-1, 1]
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        x_range = x_max - x_min
        x_range = np.where(x_range == 0, 1, x_range)
        return 2 * (X - x_min) / x_range - 1

    msg = f"Unknown scaling method: {method}"
    raise ValueError(msg)


def _build_mst(
    connectivity: NDArray[np.floating[Any]] | csr_matrix,
    X: NDArray[np.floating[Any]],
    distance_metric: str = "euclidean",
) -> csr_matrix:
    """Build a Minimum Spanning Tree from spatial connectivity and feature distances.

    Parameters
    ----------
    connectivity : ndarray or sparse matrix of shape (n_samples, n_samples)
        Spatial connectivity matrix (adjacency matrix).
    X : ndarray of shape (n_samples, n_features)
        Feature data.
    distance_metric : str, default="euclidean"
        Distance metric to use.

    Returns
    -------
    mst : sparse matrix of shape (n_samples, n_samples)
        Minimum spanning tree as sparse matrix.
    """
    if not issparse(connectivity):
        connectivity = csr_matrix(connectivity)

    connectivity = (connectivity + connectivity.T) / 2

    # Compute and CACHE distances
    distances = cdist(X, X, metric=distance_metric)

    rows, cols = connectivity.nonzero()
    weighted_graph = connectivity.copy()
    weighted_graph.data = distances[rows, cols]

    mst = minimum_spanning_tree(weighted_graph)

    return mst, distances


class ClusterStats:
    """Incremental cluster statistics tracker.

    Maintains O(1) access to:
    - Cluster size
    - Feature sum
    - Feature sum of squares
    - Variance (computed from above)
    """

    def __init__(self, X: NDArray[np.floating[Any]], labels: NDArray[np.integer[Any]]):
        """Initialize statistics for all initial clusters."""
        self.n_features = X.shape[1]
        unique_labels = np.unique(labels)

        # Initialize per-cluster aggregates
        self.size: dict[int, int] = {}
        self.sum: dict[int, NDArray[np.floating[Any]]] = {}
        self.sum_sq: dict[int, NDArray[np.floating[Any]]] = {}

        for label in unique_labels:
            mask = labels == label
            cluster_data = X[mask]

            self.size[label] = len(cluster_data)
            self.sum[label] = cluster_data.sum(axis=0)
            self.sum_sq[label] = (cluster_data**2).sum(axis=0)

    def get_variance(self, label: int) -> float:
        """Get total variance for a cluster in O(1) time."""
        if label not in self.size or self.size[label] == 0:
            return 0.0

        n = self.size[label]
        mean = self.sum[label] / n
        mean_sq = self.sum_sq[label] / n

        variance = np.sum(mean_sq - mean**2)
        return float(variance)

    def split_cluster(
        self,
        old_label: int,
        new_label: int,
        old_mask: NDArray[np.bool_],
        new_mask: NDArray[np.bool_],
        X: NDArray[np.floating[Any]],
    ) -> None:
        """Update statistics after splitting a cluster."""
        # Data in new cluster
        new_data = X[new_mask]

        # Initialize new cluster stats
        self.size[new_label] = len(new_data)
        self.sum[new_label] = new_data.sum(axis=0)
        self.sum_sq[new_label] = (new_data**2).sum(axis=0)

        # Update old cluster stats (subtract new cluster)
        if old_label in self.size:
            self.size[old_label] -= self.size[new_label]
            self.sum[old_label] -= self.sum[new_label]
            self.sum_sq[old_label] -= self.sum_sq[new_label]

            # Remove if empty
            if self.size[old_label] <= 0:
                del self.size[old_label]
                del self.sum[old_label]
                del self.sum_sq[old_label]


def _compute_edge_dissimilarity(
    mst: csr_matrix,
    distances: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    stats: ClusterStats,
    cluster_edges: dict[int, list[tuple[int, int]]] | None = None,
) -> tuple[int, int, float]:
    """Find the edge in the MST with maximum dissimilarity between cluster regions.

    Uses:
    - Cached distances from MST (no cdist calls)
    - Incremental cluster statistics (no recomputation)
    - Vectorized operations (no Python loops)

    Returns
    -------
    max_edge : tuple of (int, int, float)
        Edge (i, j) with maximum dissimilarity and the dissimilarity value.
    """
    # Get all MST edges (or subset if cluster_edges provided)
    if cluster_edges is None:
        rows, cols = mst.nonzero()
    else:
        # Only consider edges in active clusters
        edge_list = []
        for edges in cluster_edges.values():
            edge_list.extend(edges)
        if not edge_list:
            return (-1, -1, 0.0)
        rows, cols = zip(*edge_list, strict=True)
        rows = np.array(rows)
        cols = np.array(cols)

    if len(rows) == 0:
        return (-1, -1, 0.0)

    # Filter: only edges within same cluster
    same_cluster = labels[rows] == labels[cols]
    rows = rows[same_cluster]
    cols = cols[same_cluster]

    if len(rows) == 0:
        return (-1, -1, 0.0)

    # Vectorized: Get edge weights from cached distances
    edge_distances = distances[rows, cols]

    # Vectorized: Get cluster variances
    cluster_labels = labels[rows]
    variances = np.array([stats.get_variance(lbl) for lbl in cluster_labels])

    # Vectorized: Compute dissimilarities
    dissimilarities = edge_distances * variances

    # Find maximum
    max_idx = np.argmax(dissimilarities)
    max_edge = (int(rows[max_idx]), int(cols[max_idx]), float(dissimilarities[max_idx]))

    return max_edge


def _split_cluster_by_edge(
    mst: csr_matrix,
    labels: NDArray[np.integer[Any]],
    edge_i: int,
    edge_j: int,
    next_label: int,
) -> NDArray[np.integer[Any]]:
    """Split a cluster by removing an edge from the MST.

    Parameters
    ----------
    mst : sparse matrix of shape (n_samples, n_samples)
        Minimum spanning tree.
    labels : ndarray of shape (n_samples,)
        Current cluster labels.
    edge_i, edge_j : int
        Nodes defining the edge to remove.
    next_label : int
        Label to assign to the new cluster.

    Returns
    -------
    new_labels : ndarray of shape (n_samples,)
        Updated cluster labels.
    """
    # Create MST copy without the edge
    mst_copy = mst.tolil()
    mst_copy[edge_i, edge_j] = 0
    mst_copy[edge_j, edge_i] = 0
    mst_copy = mst_copy.tocsr()

    # Get current cluster
    current_label = labels[edge_i]
    cluster_mask = labels == current_label
    cluster_indices = np.where(cluster_mask)[0]

    # Build subgraph for this cluster efficiently
    row_indices = []
    col_indices = []

    for idx in cluster_indices:
        neighbors = mst_copy[idx].nonzero()[1]
        for neighbor in neighbors:
            if cluster_mask[neighbor]:
                row_indices.append(idx)
                col_indices.append(neighbor)

    # Create subgraph
    n_samples = len(labels)
    data = np.ones(len(row_indices), dtype=np.float64)
    subgraph = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_samples, n_samples),
    )

    # Use scipy's connected_components (C-optimized)
    n_components, component_labels = connected_components(
        subgraph, directed=False, return_labels=True
    )

    if n_components <= 1:
        # No split occurred (shouldn't happen)
        return labels

    # Find which component contains edge_i
    component_i = component_labels[edge_i]

    # Assign new label to the other component
    new_labels = labels.copy()
    for idx in cluster_indices:
        if component_labels[idx] != component_i:
            new_labels[idx] = next_label

    return new_labels


def _build_cluster_edge_map(
    mst: csr_matrix, labels: NDArray[np.integer[Any]]
) -> dict[int, list[tuple[int, int]]]:
    """Build map of cluster_label -> list of edges within that cluster.

    Allows scanning only relevant edges instead of all MST edges.
    """
    rows, cols = mst.nonzero()
    cluster_edges: dict[int, list[tuple[int, int]]] = {}

    for i, j in zip(rows, cols, strict=True):
        if labels[i] == labels[j]:
            label = labels[i]
            if label not in cluster_edges:
                cluster_edges[label] = []
            cluster_edges[label].append((i, j))

    return cluster_edges


class SKATER(CustomClusterMixin):
    """Spatial K-luster Analysis by Tree Edge Removal.

    SKATER forms clusters by spatially partitioning data using a connectivity graph.
    It builds a Minimum Spanning Tree (MST) from the spatial connectivity and iteratively removes
    edges to create spatially contiguous clusters with similar feature values.

    This algorithm is particularly useful for segmentation of hyperspectral mappings (e.g., Raman
    mappings) where each spatial location has associated spectral data.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    connectivity : array-like or sparse matrix of shape (n_samples, n_samples), default=None
        Spatial connectivity/adjacency matrix. If None, a grid-based connectivity is assumed.
        Non-zero entries indicate spatial neighbors.
    scale_method : {'raw', 'standardize', 'demean', 'mad', 'range_standardize', 'range_adjust'}, \
            default='standardize'
        Method to scale the input data:
        - 'raw': no scaling
        - 'standardize': z-score normalization (mean=0, std=1)
        - 'demean': subtract mean
        - 'mad': median absolute deviation scaling
        - 'range_standardize': scale to [0, 1]
        - 'range_adjust': scale to [-1, 1]
    distance_metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric for computing feature dissimilarity.
    random_state : int, default=None
        Random seed for reproducibility (currently unused, reserved for future stochastic variants).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers (computed as mean of samples in each cluster).
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    unique_labels_ : ndarray of shape (n_clusters,)
        Unique cluster labels.
    counts_ : ndarray of shape (n_clusters,)
        Number of samples in each cluster.
    mst_ : sparse matrix of shape (n_samples, n_samples)
        The minimum spanning tree used for clustering.
    name : str
        Name of the clustering algorithm.

    Methods
    -------
    fit(X, y=None)
        Compute SKATER clustering.

    Notes
    -----
    The algorithm works as follows:
    1. Scale the input data according to `scale_method`
    2. Build a Minimum Spanning Tree (MST) from spatial connectivity weighted by feature distances
    3. Start with all samples in one cluster
    4. Iteratively split clusters by removing MST edges with maximum dissimilarity
    5. Continue until `n_clusters` are formed

    All resulting clusters are guaranteed to be spatially contiguous according to the
    input connectivity matrix.

    References
    ----------
    Assunção, R.M., Neves, M.C., Câmara, G. and da Costa Freitas, C. (2006).
    "Efficient regionalization techniques for socio-economic geographical units using
    minimum spanning trees." *International Journal of Geographical Information Science*,
    20(7), 797-811.

    Examples
    --------
    >>> from clustools.cluster import SKATER
    >>> import numpy as np
    >>> from sklearn.datasets import make_blobs
    >>> from scipy.sparse import csr_matrix
    >>>
    >>> # Create sample data
    >>> X, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
    >>>
    >>> # Create a simple grid connectivity (assuming 10x10 grid)
    >>> n = 10
    >>> connectivity = np.zeros((n*n, n*n))
    >>> for i in range(n):
    ...     for j in range(n):
    ...         idx = i * n + j
    ...         # Connect to neighbors
    ...         if i > 0:
    ...             connectivity[idx, (i-1)*n + j] = 1
    ...         if i < n-1:
    ...             connectivity[idx, (i+1)*n + j] = 1
    ...         if j > 0:
    ...             connectivity[idx, i*n + j-1] = 1
    ...         if j < n-1:
    ...             connectivity[idx, i*n + j+1] = 1
    >>>
    >>> # Fit SKATER
    >>> skater = SKATER(n_clusters=3, connectivity=connectivity)
    >>> skater.fit(X)
    >>> print(skater.labels_)
    """

    name = "SKATER"

    _parameter_constraints: ClassVar[dict[str, Any]] = {
        **CustomClusterMixin._parameter_constraints,  # noqa: SLF001
        "connectivity": ["array-like", "sparse matrix", None],
        "scale_method": [
            StrOptions(
                {
                    "raw",
                    "standardize",
                    "demean",
                    "mad",
                    "range_standardize",
                    "range_adjust",
                }
            )
        ],
        "distance_metric": [StrOptions({"euclidean", "manhattan"})],
    }

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        connectivity: ArrayLike | csr_matrix | None = None,
        scale_method: Literal[
            "raw",
            "standardize",
            "demean",
            "mad",
            "range_standardize",
            "range_adjust",
        ] = "standardize",
        distance_metric: Literal["euclidean", "manhattan"] = "euclidean",
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.connectivity = connectivity
        self.scale_method = scale_method
        self.distance_metric = distance_metric
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: Any = None) -> "SKATER":  # noqa: C901
        """Compute SKATER clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster. Will be converted to C-ordered float64.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input data
        X_val = validate_data(self, X, dtype=np.float64)
        n_samples = X_val.shape[0]

        # Validate n_clusters
        if self.n_clusters < 1 or self.n_clusters > n_samples:
            msg = f"n_clusters should be between 1 and n_samples={n_samples}, got {self.n_clusters}"
            raise ValueError(msg)

        # Handle connectivity matrix
        if self.connectivity is None:
            # Assume grid connectivity (simplified - user should provide connectivity)
            msg = (
                "Connectivity matrix must be provided. "
                "SKATER requires a spatial connectivity/adjacency matrix."
            )
            raise ValueError(msg)

        # Convert connectivity to sparse if needed
        connectivity_matrix = self.connectivity
        if not issparse(connectivity_matrix):
            connectivity_matrix = csr_matrix(np.asarray(connectivity_matrix))

        # Validate connectivity dimensions
        if connectivity_matrix.shape != (n_samples, n_samples):
            msg = (
                f"Connectivity matrix shape {connectivity_matrix.shape} does not match "
                f"n_samples={n_samples}"
            )
            raise ValueError(msg)

        # Scale data
        X_scaled = _scale_data(X_val, self.scale_method)

        # Build MST and cache distances
        mst, distances = _build_mst(connectivity_matrix, X_scaled, self.distance_metric)

        # Initialize all samples to cluster 0
        labels = np.zeros(n_samples, dtype=np.int32)

        # Initialize incremental statistics
        stats = ClusterStats(X_scaled, labels)

        # Build per-cluster edge map (OPTIMIZATION 3)
        cluster_edges = _build_cluster_edge_map(mst, labels)

        next_label = 1

        # Main loop with optimizations
        # FIXME Slow loop...
        while next_label < self.n_clusters:
            # Vectorized edge scoring
            edge_i, edge_j, _ = _compute_edge_dissimilarity(
                mst, distances, labels, stats, cluster_edges
            )

            if edge_i == -1:
                break

            # Fast split using scipy
            old_labels = labels.copy()
            labels = _split_cluster_by_edge(mst, labels, edge_i, edge_j, next_label)

            # Update incremental statistics
            old_label = old_labels[edge_i]
            old_mask = (old_labels == old_label) & (labels == old_label)
            new_mask = (old_labels == old_label) & (labels == next_label)
            stats.split_cluster(old_label, next_label, old_mask, new_mask, X_scaled)

            # Update cluster edge map
            if old_label in cluster_edges:
                # Redistribute edges between old and new clusters
                old_edges = cluster_edges[old_label]
                cluster_edges[old_label] = []
                cluster_edges[next_label] = []

                for i, j in old_edges:
                    if labels[i] == labels[j]:
                        cluster_label = labels[i]
                        if cluster_label not in cluster_edges:
                            cluster_edges[cluster_label] = []
                        cluster_edges[cluster_label].append((i, j))

            next_label += 1

        # Store results
        self.labels_ = labels
        self.unique_labels_, self.counts_ = np.unique(labels, return_counts=True)

        # Compute cluster centers
        self.cluster_centers_ = np.array(
            [X_val[labels == i].mean(axis=0) for i in self.unique_labels_]
        )

        # Compute inertia
        self.inertia_ = compute_inertia(X_val, labels, self.cluster_centers_)

        self.mst_ = mst

        return self
