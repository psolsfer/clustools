from numbers import Integral, Real
from typing import Any, ClassVar, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyauxlib.decorators.import_errors import require_class
from skfuzzy.cluster import cmeans, cmeans_predict
from sklearn.base import _fit_context
from sklearn.cluster._kmeans import kmeans_plusplus
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from clustools.cluster.custom_cluster_mixin import CustomClusterMixin
from clustools.metrics.internal import compute_inertia

# ruff: noqa: N803, N806


@require_class("skfuzzy")
class FuzzyCMeans(CustomClusterMixin):
    """Fuzzy C-Means clustering.

    A soft clustering algorithm where each data point belongs to multiple clusters with varying
    degrees of membership.

    This implementation is a scikit-learn compatible wrapper around the ``scikit-fuzzy`` functions
    (:func:`skfuzzy.cluster.cmeans` and :func:`skfuzzy.cluster.cmeans_predict`).

    The underlying algorithm is the classic fuzzy c-means method described by Bezdek et al. [1]_
    and popularized in Ross et al. [2]_.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form.
    fuzz_coeff : float, default=2.0
        The fuzzification coefficient that controls the degree of fuzziness of the solution.
        The value must be different than 1.0. It's usually larger, although lower values are also
        accepted.
        Larger values will "blur" the classes, with all elements tending to belong to all clusters.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    tol : float, default=0.005
        The maximum tolerance for the stopping criterion.
    max_iter : int, default=1000
        The maximum number of iterations for the algorithm to run.
    random_state : int | None, default=None
        Determines random number generation for centroid initialization. Use an int to make the
        randomness deterministic.
    init : {'random', 'k-means++'}, default='random'
        Initialization method. If "random", multiple runs are performed (`n_init` times). If
        "k-means++", a single smart initialization is used (like sklearn.KMeans).

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    unique_labels_ : ndarray of shape (n_unique_clusters,)
        Unique labels
    counts_ : ndarray of shape (n_unique_clusters,)
        Number of samples assigned to each unique cluster label.
    fpc_ : float
        Fuzzy partition coefficient.
    u_orig_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy c-partitioned matrix.
    dist_mat_ : ndarray of shape (n_samples, n_clusters)
        Euclidian distance matrix.
    name : str
        Name of the clustering algorithm.

    Methods
    -------
    fit(X, y=None)
        Compute Fuzzy C-Means clustering using skfuzzy.

    predict(X)
        Predict hard cluster labels for new samples.

    predict_proba(X=None)
        Compute membership probabilities for samples.

    Notes
    -----
    This estimator extends the ``scikit-fuzzy`` implementation with additional scikit-learn-style
    features:
        - ``n_init``: multiple random initializations (like in :class:`~sklearn.cluster.KMeans`).
        - ``init="k-means++"``: centroid initialization inspired by the k-means++ strategy [3]_,
        adapted for fuzzy membership matrices.

    References
    ----------
    .. [1] Bezdek, J.C. (1981). *Pattern Recognition with Fuzzy Objective Function Algorithms*.
           Springer Science & Business Media.
    .. [2] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.
    .. [3] Arthur, D. and Vassilvitskii, S. (2007). "k-means++: The Advantages of
           Careful Seeding." *Proceedings of the 18th Annual ACM-SIAM Symposium on
           Discrete Algorithms (SODA)*.
    """

    name = "FuzzyCMeans"

    # Constraints checked by _validate_params
    _parameter_constraints: ClassVar[dict[str, Any]] = {
        **CustomClusterMixin._parameter_constraints,  # noqa: SLF001
        "fuzz_coeff": [Interval(Real, 1, None, closed="neither")],
        "n_init": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
    }

    def __init__(  # noqa: PLR0913
        self,
        n_clusters: int = 8,
        *,
        fuzz_coeff: float = 2.0,
        n_init: int = 10,
        tol: float = 0.005,
        max_iter: int = 1000,
        random_state: int | None = None,
        init: Literal["random", "k-means++"] = "random",
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.fuzz_coeff = fuzz_coeff
        self.n_init = n_init
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init

    @_fit_context(prefer_skip_nested_validation=True)
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
        X_val = validate_data(self, X, dtype="numeric")

        best_inertia, best_run = None, None

        n_init = 1 if self.init == "k-means++" else self.n_init

        if self.init == "k-means++":
            u0 = _init_membership_kmeanspp(
                X_val, self.n_clusters, self.fuzz_coeff, random_state=self.random_state
            )
            init = u0
        else:
            init = None  # skfuzzy will do random initialization

        for i in range(n_init):
            cntr, u_orig, _, dist_mat_, _, _, fpc_ = cmeans(
                X_val.T,
                self.n_clusters,
                self.fuzz_coeff,
                error=self.tol,
                maxiter=self.max_iter,
                init=init,
                seed=None if self.random_state is None else self.random_state + i,
            )

            labels = np.argmax(u_orig, axis=0)
            inertia = compute_inertia(X_val, labels, cntr)

            if best_inertia is None or (inertia < best_inertia):
                best_inertia = inertia
                best_run = (cntr, u_orig, dist_mat_, fpc_, labels)

        # Unpack best run
        self.cluster_centers_, u_orig, dist_mat_, self.fpc_, self.labels_ = best_run  # type: ignore[misc]
        self.u_orig_: NDArray[np.floating[Any]] = u_orig.T
        self.dist_mat_ = dist_mat_.T
        self.inertia_ = best_inertia
        self.unique_labels_, self.counts_ = np.unique(self.labels_, return_counts=True)

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.floating[Any]]:
        """Predict cluster labels for new samples."""
        check_is_fitted(self)
        X_val: NDArray[np.floating[Any]] = validate_data(self, X, reset=False, dtype=np.float64)
        u, _, _, _, _, _ = cmeans_predict(
            X_val.T,
            self.cluster_centers_,
            self.fuzz_coeff,
            error=self.tol,
            maxiter=self.max_iter,
        )
        return np.argmax(u, axis=0)  # type: ignore[no-any-return]

    def predict_proba(self, X: ArrayLike | None = None) -> NDArray[np.floating[Any]]:
        """Calculate the probability density distribution for each cluster.

        Returns
        -------
        probabilities : array-like, shape=(n_samples, n_clusters)
            Probability density distribution for each cluster.
            For example, the probability for the 3rd cluster is obtained as:
            `probabilities[:, 2]`
        """
        check_is_fitted(self)

        if X is None:
            return self.u_orig_

        X_val: NDArray[np.floating[Any]] = validate_data(self, X, reset=False, dtype=np.float64)
        u, _, _, _, _, _ = cmeans_predict(
            X_val.T,
            self.cluster_centers_,
            self.fuzz_coeff,
            error=self.tol,
            maxiter=self.max_iter,
        )
        return u.T


def _init_membership_kmeanspp(
    X: ArrayLike,
    n_clusters: int,
    fuzz_coeff: float,
    random_state: int | None = None,
) -> NDArray[np.floating[Any]]:
    r"""Initialize fuzzy membership matrix using k-means++ centroids.

    This function generates initial fuzzy memberships for fuzzy c-means clustering
    by first selecting centroids using the k-means++ initialization algorithm,
    then computing fuzzy memberships based on distances to these centroids.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data points to cluster. Each row represents a sample and each
        column represents a feature.
    n_clusters : int
        The number of clusters to form. Must be greater than 1.
    fuzz_coeff : float
        The fuzzification coefficient (often denoted as 'm' in literature).
        Controls the degree of fuzziness in cluster assignments. Must be > 1.
        Values closer to 1 produce crisp (hard) assignments, while larger
        values produce softer (more fuzzy) assignments. Typical range: [1.1, 3.0].
    random_state : int | None, default=None
        Random seed for reproducible centroid initialization. If None, the
        random state is not controlled.

    Returns
    -------
    u0 : ndarray of shape (n_clusters, n_samples)
        Initial fuzzy membership matrix where u0[i, j] represents the membership degree of sample j
        to cluster i. Each column sums to 1.0, and all values are in the range (0, 1].

    Notes
    -----
    The fuzzy membership is calculated using the standard fuzzy c-means formula:

    .. math::
        u_{ij} = \\frac{1}{\\sum_{k=1}^{c} \\left(\\frac{d_{ij}}{d_{kj}}\\right)^{\\frac{2}{m-1}}}

    where:
    - :math:`u_{ij}` is the membership of sample j to cluster i
    - :math:`d_{ij}` is the distance from sample j to centroid i
    - :math:`c` is the number of clusters
    - :math:`m` is the fuzzification coefficient (fuzz_coeff)

    The k-means++ initialization ensures good initial centroid placement by
    selecting centroids with probability proportional to their squared distance
    from existing centroids.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    >>> u0 = _init_membership_kmeanspp(X, n_clusters=3, fuzz_coeff=2.0, random_state=42)
    >>> u0.shape
    (3, 100)
    >>> np.allclose(np.sum(u0, axis=0), 1.0)  # Each sample's memberships sum to 1
    True
    >>> bool(np.all((u0 > 0) & (u0 <= 1)))  # All memberships in valid range
    True
    """
    # Get k-means++ centroids directly
    centroids, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=random_state)

    # Calculate distances from each point to each centroid (shape: (n_samples, n_clusters))
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # type: ignore[call-overload, index]

    # Vectorized fuzzy membership calculation
    # Handle zero distances to avoid division by zero
    distances = np.maximum(distances, 1e-10)

    # Calculate the exponent
    exp = 2.0 / (fuzz_coeff - 1)

    # Vectorized calculation:
    # For each point, calculate membership to each cluster: u_ij = 1 / sum_k((d_ij / d_kj)^exp)
    # This is equivalent to: u_ij = d_ij^(-exp) / sum_k(d_kj^(-exp))
    inv_distances = distances ** (-exp)  # shape: (n_samples, n_clusters)
    u0 = inv_distances / np.sum(inv_distances, axis=1, keepdims=True)

    # Transpose to get shape (n_clusters, n_samples)
    u0 = u0.T

    return u0  # type: ignore[no-any-return]
