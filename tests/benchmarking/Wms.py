import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
from projectUtils.ollamaUtils import get_embeddings_batch

MODEL_NAME = 'embeddinggemma:latest'

# --- 1. LOAD THE DATASET  ---
print("Loading Word Meaning Similarity dataset (SimLex-999)...")

script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go UP two levels to reach the project root (LLMEvaluation/)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# 3. Build the exact path from the root down to the dataset
file_path = os.path.join(project_root, "datasets", "benchmarking", "SimLex-999.txt")

# Load the tab-separated file using Pandas
print(f"Parsing dataset from: {file_path}")
df = pd.read_csv(file_path, sep='\t')

# Extract the columns (SimLex999 is the human rating out of 10)
word1_list = df['word1'].tolist()
word2_list = df['word2'].tolist()

#priting first 5 rows to verify
print(word1_list[:5])
print(word2_list[:5])

human_scores = df['SimLex999'].tolist()

print(f"✅ Successfully loaded {len(word1_list)} word pairs.")

# --- 2. THE SEAT TEMPLATE WRAPPER ---
# We wrap isolated words in a neutral context so the sentence transformer doesn't panic
print("Wrapping words in neutral context...")
sentences_1 = [f"The word is {w}." for w in word1_list]
sentences_2 = [f"The word is {w}." for w in word2_list]
print(sentences_1[:5])
print(sentences_2[:5])


# --- 3. GENERATE EMBEDDINGS (Safe float64) ---
print(f"Generating contextual embeddings via {MODEL_NAME}...")
emb_1 = np.array(get_embeddings_batch(MODEL_NAME, sentences_1), dtype=np.float64)
emb_2 = np.array(get_embeddings_batch(MODEL_NAME, sentences_2), dtype=np.float64)

# --- 4. THE SAFE MATH ENGINE (Cosine Similarity) ---
print("Calculating Cosine Similarities...")
norm_1 = normalize(emb_1, norm='l2')
norm_2 = normalize(emb_2, norm='l2')

gemma_scores = []
for i in range(len(norm_1)):
    # Calculate dot product for pair i (Equivalent to Cosine Similarity for normalized vectors)
    sim = np.dot(norm_1[i], norm_2[i])
    gemma_scores.append(sim)

# --- 5. THE SPEARMAN CORRELATION ---
print("Running Spearman Correlation against Human Data...")
correlation, p_value = spearmanr(human_scores, gemma_scores)

# --- 6. RESULTS & INTERPRETATION ---
print("\n=== WMS EVALUATION (WORD MEANING SIMILARITY) ===")
print("-" * 55)
print(f"Dataset Used: SimLex-999")
print(f"Total Pairs Tested: {len(word1_list)}")
print(f"Spearman Correlation: {correlation:.4f}")
print(f"P-Value: {p_value:.4e} (Must be < 0.05)")
print("-" * 55)

# Industry benchmarks for WMS on modern sentence transformers
if correlation > 0.60:
    print("Verdict: 🟢 STATE OF THE ART (Safe for production search/routing)")
elif correlation > 0.40:
    print("Verdict: 🟡 ACCEPTABLE (Average performance, beware of antonym confusion)")
else:
    print("Verdict: 🔴 POOR (Do not use this model for semantic mapping)")


# --- ERROR ANALYSIS: Find the biggest confusions ---
# We normalize the human scores (0 to 10) to match the AI scores (0 to 1) just for easy comparison
normalized_human = [score / 10.0 for score in human_scores]

# Calculate the absolute difference between human intuition and AI math
differences = [abs(h - a) for h, a in zip(normalized_human, gemma_scores)]

# Sort the dataset by the biggest mistakes
error_ranking = sorted(zip(word1_list, word2_list, normalized_human, gemma_scores, differences),
                       key=lambda x: x[4], reverse=True)

print("\n🚨 GEMMA'S BIGGEST MISTAKES (Top 5) 🚨")
print(f"{'Word 1':<12} | {'Word 2':<12} | {'Human':<8} | {'Gemma':<8} | {'Difference'}")
print("-" * 60)
for i in range(5):
    w1, w2, h, g, diff = error_ranking[i]
    print(f"{w1:<12} | {w2:<12} | {h:<8.2f} | {g:<8.2f} | {diff:<8.2f}")