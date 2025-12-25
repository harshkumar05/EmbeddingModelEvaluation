import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def test_isotropy(embeddings):
    """
    Diagnoses isotropy of an embedding space.
    Args:
        embeddings: np.array of shape (n_samples, n_dimensions)
    """
    print(embeddings)
    n_samples, n_dim = embeddings.shape
    print(f"Analyzing {n_samples} embeddings with {n_dim} dimensions...\n")

    # --- Test 1: Average Random Cosine Similarity ---
    # We take random pairs and check their similarity.
    # In a perfectly isotropic space, this should be close to 0.
    # In anisotropic models (like BERT), this is often > 0.5 or even > 0.9.

    # Calculate pairwise cosine similarity for a random subset (to save memory)
    subset_size = min(1000, n_samples)
    indices = np.random.choice(n_samples, subset_size, replace=False)
    subset = embeddings[indices]

    cos_sim_matrix = cosine_similarity(subset)

    # Get upper triangle values (excluding diagonal)
    sim_values = cos_sim_matrix[np.triu_indices(subset_size, k=1)]
    avg_sim = np.mean(sim_values)

    print(f"--- Metric 1: Narrow Cone Test ---")
    print(f"Avg Cosine Similarity: {avg_sim:.4f}")
    if avg_sim > 0.5:
        print("⚠️  High Anisotropy detected (Embeddings are clustering in a cone).")
    else:
        print("✅  Space appears reasonably distributed.")
    print("-" * 30)

    # --- Test 2: Principal Component Variance (PCA) ---
    # We check how many dimensions are actually doing the work.
    # If 1-3 dimensions explain 90% of the variance, the model is collapsing dimensions.

    pca = PCA(n_components=min(n_dim, 50))  # specific n_components to avoid calculating all
    pca.fit(embeddings)

    explained_variance = pca.explained_variance_ratio_

    # Cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    print(f"\n--- Metric 2: Dimensional Utilization (PCA) ---")
    print(f"Variance explained by Top 1 PC: {explained_variance[0] * 100:.2f}%")
    print(f"Variance explained by Top 3 PCs: {cumulative_variance[2] * 100:.2f}%")

    # Ideally, variance should be spread out.
    # If Top 1 > 20-30% for a high-dim model (768d), it's highly anisotropic.
    if explained_variance[0] > 0.1:
        print(f"⚠️  Top component dominates. The model heavily relies on a specific axis.")
    else:
        print(f"✅  Variance is well distributed across components.")

    return sim_values, explained_variance


# --- Visualization Helper ---
def plot_isotropy(sim_values, explained_variance):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of Cosine Similarities
    ax[0].hist(sim_values, bins=50, color='skyblue', edgecolor='black')
    ax[0].set_title('Distribution of Cosine Similarities')
    ax[0].set_xlabel('Cosine Similarity')
    ax[0].set_ylabel('Frequency')
    ax[0].axvline(x=0, color='r', linestyle='--', label="Ideal Mean (0)")
    ax[0].legend()

    # PCA Variance Plot
    ax[1].plot(explained_variance[:20], marker='o')
    ax[1].set_title('Variance Explained by Top 20 PCs')
    ax[1].set_xlabel('Principal Component')
    ax[1].set_ylabel('Variance Ratio')

    plt.tight_layout()
    plt.show()

# USAGE EXAMPLE:
# Generate dummy anisotropic data (Narrow Cone) to demonstrate
# random_noise = np.random.normal(0, 1, (1000, 768))
# cone_bias = np.random.normal(5, 2, (1000, 1)) * np.ones((1, 768))
# dummy_embeddings = random_noise + cone_bias

# sim_vals, var_exp = test_isotropy(dummy_embeddings)
# plot_isotropy(sim_vals, var_exp)