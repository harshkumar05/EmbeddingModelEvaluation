import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score

# Import your bulletproof batching function
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from projectUtils.ollamaUtils import get_embeddings_batch

MODEL_NAME = 'embeddinggemma:latest'

# --- Dataset ---
corpus = [
    # Baseline Tests
    "The Python programming language was created by Guido van Rossum.",  # 0
    "To fix a leaking pipe, first turn off the main water valve.",  # 1

    # Scenario 1: Lexical Trap
    "Cardiovascular workouts like running or swimming elevate your heart rate.",  # 2 (Correct for Q2)
    "As a citizen, I like to exercise my right to vote in the local elections.",  # 3 (Trap!)

    # Scenario 2: Negation Trap
    "Doctors highly recommend that you avoid consuming caffeine right before bedtime.",  # 4 (Correct for Q3)
    "Drinking caffeine in the morning is a great way to wake up.",  # 5 (Trap!)

    # Scenario 3: Syntax Trap
    "The software engineer reported the critical bug to the product manager.",  # 6 (Correct for Q4)
    "The product manager reported the critical bug to the software engineer.",  # 7 (Trap!)

    # Scenario 4: Polysemy Trap
    "The muddy bank of the river was slippery after the heavy rainstorm.",  # 8 (Correct for Q5)
    "The central bank decided to raise interest rates to combat inflation."  # 9 (Trap!)
]

queries = [
    "Who invented Python?",  # 0 -> Maps to 0
    "How do I repair a dripping sink?",  # 1 -> Maps to 1
    "What is a good way to exercise?",  # 2 -> Maps to 2
    "Is it a good idea to drink coffee before going to sleep?",  # 3 -> Maps to 4
    "Who received the bug report from the developer?",  # 4 -> Maps to 6
    "Why is the ground wet near the water?"  # 5 -> Maps to 8
]

ground_truth_mapping = {
    0: 0,
    1: 1,
    2: 2,  # Lexical Test
    3: 4,  # Negation Test
    4: 6,  # Syntax Test
    5: 8  # Polysemy Test
}

print(f"✅ Loaded Corpus ({len(corpus)} docs) and Queries ({len(queries)}).")

# --- 2. GENERATE EMBEDDINGS ---
print(f"\n🧠 Generating embeddings via {MODEL_NAME}...")
corpus_embs = np.array(get_embeddings_batch(MODEL_NAME, corpus), dtype=np.float64)
query_embs = np.array(get_embeddings_batch(MODEL_NAME, queries), dtype=np.float64)

# Normalize for safe Cosine Similarity
corpus_norm = normalize(corpus_embs, norm='l2')
query_norm = normalize(query_embs, norm='l2')

# --- 3. THE SEARCH ENGINE MATH (Calculate MRR & NDCG) ---
print("\n🧮 Calculating Search Metrics (MRR & NDCG@10)...")

mrr_scores = []
ndcg_scores = []
mistakes = []  # Tracks where the model fell for our traps

print("-" * 85)
print(f"{'Query':<56} | {'Rank'} | {'Sim Score'}")
print("-" * 85)

for query_idx, query_vec in enumerate(query_norm):
    # 1. Similarity against all 10 documents
    similarities = np.dot(corpus_norm, query_vec)

    # 2. Sort documents (Highest similarity to lowest)
    ranked_doc_indices = np.argsort(similarities)[::-1]

    # 3. Target Document ID
    target_doc_id = ground_truth_mapping[query_idx]

    # 4. Calculate MRR
    rank = np.where(ranked_doc_indices == target_doc_id)[0][0] + 1
    mrr_scores.append(1.0 / rank)

    # 5. Calculate NDCG@10
    y_true = np.zeros(len(corpus))
    y_true[target_doc_id] = 1.0

    ndcg = ndcg_score([y_true], [similarities], k=10)
    ndcg_scores.append(ndcg)

    # Print the top result for the terminal
    top_doc_id = ranked_doc_indices[0]
    top_sim = similarities[top_doc_id]

    # --- CATCH THE MISTAKES ---
    if top_doc_id != target_doc_id:
        mistakes.append({
            "query": queries[query_idx],
            "expected_doc": corpus[target_doc_id],
            "actual_doc": corpus[top_doc_id],
            "actual_rank": rank
        })

    query_text = (queries[query_idx][:53] + '...') if len(queries[query_idx]) > 53 else queries[query_idx]
    print(f"{query_text:<56} | #{rank:<3} | {top_sim:.4f}")

# --- 4. FINAL RESULTS ---
final_mrr = np.mean(mrr_scores)
final_ndcg = np.mean(ndcg_scores)

print("-" * 85)
print(f"🎯 FINAL MRR Score:        {final_mrr:.4f} (1.0 is perfect)")
print(f"📊 FINAL NDCG@10 Score:    {final_ndcg:.4f} (1.0 is perfect)")
print("-" * 85)

if final_ndcg > 0.90:
    print("Verdict: 🟢 PRODUCTION READY (Flawless Retrieval)")
elif final_ndcg > 0.70:
    print("Verdict: 🟡 ACCEPTABLE (Good enough for basic Search, might need Reranking)")
else:
    print("Verdict: 🔴 POOR (Model fails to map questions to answers)")

# --- 5. ERROR ANALYSIS (THE AUTOPSY) ---
if mistakes:
    print("\n🚨 GEMMA'S RETRIEVAL MISTAKES (THE TRAPS IT FELL FOR) 🚨")
    for idx, mistake in enumerate(mistakes):
        print("-" * 85)
        print(f"❌ FAILED QUERY: {mistake['query']}")
        print(f"   Target Answer (Pushed down to Rank #{mistake['actual_rank']}):")
        print(f"   -> {mistake['expected_doc']}")
        print(f"   Gemma's #1 Choice (The Trap):")
        print(f"   -> {mistake['actual_doc']}")
    print("-" * 85)
else:
    print("\n🎉 No mistakes! Gemma dodged every single trap.")