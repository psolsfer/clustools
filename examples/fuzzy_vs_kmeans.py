"""Example comparing fuzzy c-means and k-means clustering."""

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("This example requires matplotlib.")  # noqa: T201
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from clustools.cluster._fuzzy import FuzzyCMeans

N_CLUSTERS = 5

# Generate sample data
X, y = make_blobs(n_samples=300, centers=N_CLUSTERS, cluster_std=1.5, random_state=42)

# Fit KMeans
km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
km_labels = km.fit_predict(X)

# Fit FuzzyCMeans
fcm = FuzzyCMeans(n_clusters=N_CLUSTERS, fuzz_coeff=2.0, random_state=42, init="k-means++")
fcm.fit(X)
fcm_labels = fcm.labels_
fcm_probs = fcm.predict_proba()  # Will be fcm.u_orig_

# Plot original clusters
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=30)
plt.title("Original clusters")


# Plot KMeans
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=km_labels, cmap="viridis", s=30)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="red", marker="x")
plt.title("KMeans (hard labels)")

# Plot FuzzyCMeans (soft colors via membership to cluster 0)
plt.subplot(1, 3, 3)
alphas = np.max(fcm_probs, axis=1)  # confidence (highest membership)
plt.scatter(X[:, 0], X[:, 1], c=fcm_labels, alpha=alphas, s=30)
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c="red", marker="x")
plt.title("FuzzyCMeans (soft membership)")

plt.show()
