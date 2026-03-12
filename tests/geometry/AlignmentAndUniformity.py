import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances

# Assuming this is your custom local module
from projectUtils.ollamaUtils import get_embeddings_batch

MODEL_NAME = 'embeddinggemma:latest'

# thresholds for sts-b
HIGH_THRESH = 3.0  # similar enough to test alignment
LOW_THRESH = 1.0   # strictly dissimilar to test uniformity
MAX_SAMPLES = 1000 # limit to avoid OOM or massive wait times on local inference

print("Loading STS-B dataset...")
dataset = load_dataset("glue", "stsb", split="train")
df = dataset.to_pandas()

# --- 1. DATA PREP ---
print("Filtering datasets...")

# grab positive pairs for alignment
pos_df = df[df['label'] >= HIGH_THRESH].head(MAX_SAMPLES)
pos_s1 = pos_df['sentence1'].tolist()
pos_s2 = pos_df['sentence2'].tolist()

# grab random sentences for uniformity
# we don't need pairs here, just a pool of unrelated sentences
neg_df = df[df['label'] <= LOW_THRESH].head(MAX_SAMPLES)
uniformity_pool = neg_df['sentence1'].tolist()

print(f"Alignment pairs: {len(pos_s1)}")
print(f"Uniformity pool size: {len(uniformity_pool)}")

# --- 2. GENERATE EMBEDDINGS ---
print("\nGenerating embeddings (this might take a minute)...")

# alignment embeddings
emb_pos1 = np.array(get_embeddings_batch(MODEL_NAME, pos_s1))
emb_pos2 = np.array(get_embeddings_batch(MODEL_NAME, pos_s2))

# uniformity embeddings
emb_unif = np.array(get_embeddings_batch(MODEL_NAME, uniformity_pool))

# --- 3. NORMALIZE ---
# l2 norm maps everything to the surface of the unit hypersphere
norm_pos1 = normalize(emb_pos1, norm='l2')
norm_pos2 = normalize(emb_pos2, norm='l2')
norm_unif = normalize(emb_unif, norm='l2')

# --- 4. ALIGNMENT LOSS ---
# formula: average squared euclidean distance between positive pairs
print("\nCalculating metrics...")

diff = norm_pos1 - norm_pos2
sq_diff = diff ** 2
pair_distances = np.sum(sq_diff, axis=1)
alignment_loss = np.mean(pair_distances)

# --- 5. UNIFORMITY LOSS ---
# formula: log(mean(exp(-t * distance^2))) for all random pairs
t = 2.0 # standard param from Wang & Isola paper

# calculate distance from every point to every other point
dist_matrix = euclidean_distances(norm_unif, norm_unif)
sq_dist_matrix = dist_matrix ** 2

# apply gaussian kernel penalty (the "force field")
penalties = np.exp(-t * sq_dist_matrix)

# zero out the diagonal so we don't penalize a sentence against itself
np.fill_diagonal(penalties, 0)

# calculate average penalty (divide by total possible pairs)
n = len(uniformity_pool)
avg_penalty = np.sum(penalties) / (n * (n - 1))
uniformity_loss = np.log(avg_penalty)

# --- RESULTS ---
print("\n=== FINAL RESULTS ===")
print(f"Model: {MODEL_NAME}")
print(f"Alignment Loss:  {alignment_loss:.4f}  (Goal: closer to 0 is better)")
print(f"Uniformity Loss: {uniformity_loss:.4f}  (Goal: more negative is better, e.g., -2.5)")
