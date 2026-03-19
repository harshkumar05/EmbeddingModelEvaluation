import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr
from projectUtils.ollamaUtils import get_embeddings_batch, get_embeddings_batch_via_api

MODEL_NAME = 'embeddinggemma:latest'

# --- 1. DOWNLOAD & LOAD THE DATASET (Hugging Face API) ---
print("Pulling official STS-Benchmark dataset from Hugging Face...")

# Using the GLUE benchmark subset 'stsb'
dataset = load_dataset("sentence-transformers/stsb", split="test")

# Extract the columns (GLUE formats these as 'sentence1', 'sentence2', and 'label' from 0.0 to 5.0)
sent1_list = [str(text) if text else "empty sentence" for text in dataset['sentence1']]
sent2_list = [str(text) if text else "empty sentence" for text in dataset['sentence2']]
human_scores = list(dataset['score']) # These are the human similarity scores (0.0 to 5.0)

print(f"✅ Successfully loaded {len(sent1_list)} real-world sentence pairs.")
print(sent1_list[:5])
print(sent2_list[:5])
print(human_scores[:5])
# --- 3. GENERATE EMBEDDINGS (Safe float64) ---
print(f"Generating contextual embeddings via {MODEL_NAME}...")
emb_1 = np.array(get_embeddings_batch(MODEL_NAME, sent1_list), dtype=np.float64)
emb_2 = np.array(get_embeddings_batch(MODEL_NAME, sent2_list), dtype=np.float64)



# --- 4. THE SAFE MATH ENGINE (Cosine Similarity) ---
print("Calculating Cosine Similarities...")
norm_1 = normalize(emb_1, norm='l2')
norm_2 = normalize(emb_2, norm='l2')


gemma_scores = []
for i in range(len(norm_1)):
    sim = np.dot(norm_1[i], norm_2[i])
    gemma_scores.append(sim)

print(gemma_scores[:5])
print(human_scores[:5])

# --- 4. THE SPEARMAN CORRELATION ---
print("Running Spearman Correlation against Human Context Data...")
correlation, p_value = spearmanr(human_scores, gemma_scores)

# --- 5. RESULTS & INTERPRETATION ---
print("\n=== STS-BENCHMARK EVALUATION (CONTEXT SIMILARITY) ===")
print("-" * 65)
print(f"Dataset Used: GLUE STS-B (Train Split)")
print(f"Total Pairs Tested: {len(sent1_list)}")
print(f"Spearman Correlation: {correlation:.4f}")
print(f"P-Value: {p_value:.4e} (Must be < 0.05)")
print("-" * 65)

# Benchmarks based on Reimers & Gurevych (2019) Sentence-BERT paper
if correlation > 0.82:
    print("Verdict: 🟢 STATE OF THE ART (Matches or beats optimized Sentence Transformers)")
elif correlation > 0.65:
    print("Verdict: 🟡 GOOD (Strong contextual understanding, safe for most RAG tasks)")
elif correlation > 0.46:
    print("Verdict: 🟠 MODERATE (Performs like a raw, untuned LLM. Prone to syntax errors)")
else:
    print("Verdict: 🔴 POOR (Model ignores context entirely)")

# --- 6. ERROR ANALYSIS: Find the biggest contextual confusions ---
# Calculate the absolute difference between human intuition and AI math
# --- 6. ERROR ANALYSIS: Find the biggest contextual confusions ---
# Calculate the absolute difference between human intuition and AI math
# --- 6. ERROR ANALYSIS: Find the biggest contextual confusions ---
# Calculate the absolute difference between human intuition and AI math
differences = [abs(h - g) for h, g in zip(human_scores, gemma_scores)]

# Calculate the Critical Failure Rate (Differences > 0.5)
massive_fails_count = sum(1 for diff in differences if diff > 0.5)
failure_rate = (massive_fails_count / len(differences)) * 100

# Sort the dataset by the biggest mistakes (Difference)
error_ranking = sorted(zip(sent1_list, sent2_list, human_scores, gemma_scores, differences),
                       key=lambda x: x[4], reverse=True)

print("\n🚨 GEMMA'S BIGGEST CONTEXTUAL MISTAKES 🚨")
print(f"Critical Failures (Diff > 0.5): {massive_fails_count} out of {len(sent1_list)} pairs ({failure_rate:.1f}%)")
print("-" * 230)
# Widened the columns to 100 to fit the longer sentences
print(f"{'Sentence 1':<100} | {'Sentence 2':<100} | {'Human':<6} | {'Gemma':<6} | {'Diff'}")
print("-" * 230)

for i in range(5):
    s1, s2, h, g, diff = error_ranking[i]

    # Truncate at 100 characters (97 chars + 3 dots)
    s1_short = (s1[:97] + '...') if len(s1) > 100 else s1
    s2_short = (s2[:97] + '...') if len(s2) > 100 else s2

    # Print with 100-character padding
    print(f"{s1_short:<100} | {s2_short:<100} | {h:<6.2f} | {g:<6.2f} | {diff:<4.2f}")