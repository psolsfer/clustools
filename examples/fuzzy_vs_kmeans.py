# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "clustools",
#     "matplotlib",
# ]
# ///

"""Example comparing fuzzy c-means and k-means clustering."""

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("This example requires matplotlib.")  # noqa: T201
    raise
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from clustools.cluster import FuzzyCMeans

N_CLUSTERS = 10

# Generate sample data
X, y, centers = make_blobs(
    n_samples=300, centers=N_CLUSTERS, cluster_std=1.5, random_state=42, return_centers=True
)

# Fit KMeans
km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
km_labels = km.fit_predict(X)

# Fit FuzzyCMeans
fcm = FuzzyCMeans(n_clusters=N_CLUSTERS, fuzz_coeff=2.0, random_state=42, init="k-means++")
fcm.fit(X)
fcm_labels = fcm.labels_
fcm_probs = fcm.predict_proba()  # It's fcm.u_orig_

# Fit FuzzyCMeans random init
fcm_r = FuzzyCMeans(n_clusters=N_CLUSTERS, fuzz_coeff=2.0, random_state=42, init="random", n_init=2)
fcm_r.fit(X)
fcm_r_labels = fcm.labels_
fcm_r_probs = fcm.predict_proba()  # It's fcm_r.u_orig_

# Plot original clusters
plt.subplot(1, 4, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=30)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="x")
plt.title("Original clusters")


# Plot KMeans
plt.subplot(1, 4, 2)
plt.scatter(X[:, 0], X[:, 1], c=km_labels, cmap="viridis", s=30)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="red", marker="x")
plt.title("KMeans (hard labels)")

# Plot FuzzyCMeans (soft colors via membership to cluster 0)
plt.subplot(1, 4, 3)
alphas = np.max(fcm_probs, axis=1)  # confidence (highest membership)
plt.scatter(X[:, 0], X[:, 1], c=fcm_labels, alpha=alphas, s=30)
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c="red", marker="x")
plt.title("FuzzyCMeans (soft membership)")

# Plot FuzzyCMeans random init (soft colors via membership to cluster 0)
plt.subplot(1, 4, 4)
alphas = np.max(fcm_r_probs, axis=1)  # confidence (highest membership)
plt.scatter(X[:, 0], X[:, 1], c=fcm_r_labels, alpha=alphas, s=30)
plt.scatter(fcm_r.cluster_centers_[:, 0], fcm_r.cluster_centers_[:, 1], c="red", marker="x")
plt.title("FuzzyCMeans (soft membership)")

plt.show()
