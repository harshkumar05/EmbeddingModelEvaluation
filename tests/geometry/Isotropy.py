import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import ollama
print("Loading the datasets")
dataset = load_dataset("glue", "stsb", split=f"validation[:{10}]")
df=dataset.to_pandas()
sentences = dataset['sentence1']
print(df)
print(sentences)
MODEL_NAME="embeddinggemma:latest"
embeddings_list = []
for sentence in sentences:
    response = ollama.embeddings(model=MODEL_NAME, prompt=sentence)
    vec = response['embedding']
    embeddings_list.append(vec)

embedding_matrix = np.array(embeddings_list)
n_samples, n_dim = embedding_matrix.shape

print(f" Extraction Completed.")
print(f" Overall Embedding Matrix Shape: {n_samples} rows x {n_dim} dimensions")

print("...Running 'Narrow Cone' Test...")

cos_sim_matrix = cosine_similarity(embedding_matrix)
upper_tri = np.triu_indices(n_samples, k=1)
sim_values = cos_sim_matrix[upper_tri]
avg_sim = np.mean(sim_values)

print(f"   Avg Cosine Similarity: {avg_sim:.4f}")

if avg_sim > 0.5:
    print("DIAGNOSIS: High Anisotropy (The 'Cone' Problem).")
    print("Random sentences look too similar.")
elif avg_sim < 0.3:
    print("DIAGNOSIS: High Isotropy (Healthy Space).")
    print("The model is utilizing the vector space well.")
else:
    print("DIAGNOSIS: Moderate Isotropy.")

pca = PCA(n_components=10)
pca.fit(embedding_matrix)
explained_vars = pca.explained_variance_ratio_

top1_var = explained_vars[0] * 100
top3_var = np.sum(explained_vars[:3]) * 100

print(f"   Top 1 Dimension holds: {top1_var:.2f}% of variance")
print(f"   Top 3 Dimensions hold: {top3_var:.2f}% of variance")

if top1_var > 30:
    print("DIAGNOSIS: Dimensional Collapse.")
    print("One dimension is doing all the work.")
else:
    print("DIAGNOSIS: Well Distributed.")

print("\Generating Report Plots...")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Histogram (The Cone Check)
# Ideal: A bell curve centered near 0 or 0.2
# Bad: A spike near 0.8 or 0.9
ax[0].hist(sim_values, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
ax[0].axvline(avg_sim, color='red', linestyle='--', linewidth=2, label=f"Mean: {avg_sim:.2f}")
ax[0].set_title(f"Isotropy Distribution\n({MODEL_NAME})")
ax[0].set_xlabel("Similarity Score (-1 to 1)")
ax[0].set_ylabel("Frequency")
ax[0].legend()
ax[0].grid(axis='y', alpha=0.3)

# Plot B: PCA Curve (The Dimension Check)
# Ideal: A gentle slope
# Bad: A sharp 'L' shape drop
ax[1].plot(range(1, 11), explained_vars, marker='o', linestyle='-', color='purple', linewidth=2)
ax[1].set_title("Variance by Dimension (PCA)")
ax[1].set_xlabel("Principal Component #")
ax[1].set_ylabel("Variance Ratio")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

