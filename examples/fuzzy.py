"""Example showing fuzzy membership initialization."""

import sys

import numpy as np
from sklearn.cluster._kmeans import kmeans_plusplus
from sklearn.datasets import make_blobs

from clustools.cluster._fuzzy import _init_membership_kmeanspp

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ruff: noqa: T201

if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=100, centers=5, n_features=2, cluster_std=1.5, random_state=42)

    n_clusters = 5
    fuzz_coeff = 2.0

    # Get fuzzy memberships and centroids
    u0 = _init_membership_kmeanspp(X, n_clusters, fuzz_coeff, random_state=42)

    # For comparison, obtain the k-means++ centroids directly
    centroids, _ = kmeans_plusplus(X, n_clusters=n_clusters, random_state=42)

    print("Shape of fuzzy membership matrix:", u0.shape)
    print("Sum of memberships per point (should be ~1.0):", np.sum(u0, axis=0)[:5])
    print("Centroids:")
    print(centroids)

    # Visualize
    if not HAS_MATPLOTLIB:
        print("Install matplotlib to include a visualization.")
        sys.exit()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Data points colored by highest membership
    max_membership_labels = np.argmax(u0, axis=0)
    colors = ["red", "blue", "green", "yellow", "brown"]
    for i in range(n_clusters):
        mask = max_membership_labels == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.6, label=f"Cluster {i}")
    ax1.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="x",
        s=200,
        linewidths=3,
        label="K-means++ centroids",
    )
    ax1.set_title("Points colored by highest fuzzy membership")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Fuzzy membership strengths for cluster 0
    membership_strength = u0[0, :]  # membership to first cluster
    scatter = ax2.scatter(X[:, 0], X[:, 1], c=membership_strength, cmap="Reds", alpha=0.8)
    ax2.scatter(centroids[0, 0], centroids[0, 1], c="black", marker="x", s=200, linewidths=3)
    ax2.set_title(f"Fuzzy membership strength for Cluster 0\n(fuzziness={fuzz_coeff})")
    plt.colorbar(scatter, ax=ax2, label="Membership strength")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Show some statistics
    print(f"\nFuzziness coefficient: {fuzz_coeff}")
    print(f"Average membership entropy: {np.mean(-np.sum(u0 * np.log(u0 + 1e-10), axis=0)):.3f}")
    print(f"Range of membership values: [{np.min(u0):.3f}, {np.max(u0):.3f}]")
