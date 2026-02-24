from datasets import load_dataset
import random
import ollama
from sklearn.preprocessing import normalize
from projectUtils.ollamaUtils import get_embedding, get_embeddings_batch
import numpy as np

print("...Loading Datasets...")
dataset = load_dataset("glue", "stsb", split="train")
df=dataset.to_pandas()

#threshold
high_threshold=3.0
low_threshold=2.0
MODEL_NAME='embeddinggemma:latest'


positive_sentence1 = []
positive_sentence2 = []
negative_sentence1 = []
negative_sentence2 = []

positive_pairs_embeddings = []
negative_pairs_embeddings = []

print("...Filtering Pairs...")

sentences = dataset['sentence1']

for data in dataset:
    score=data['label']
    s1 = data['sentence1']
    s2 = data['sentence2']

    if score>=high_threshold:
        positive_sentence1.append(s1)
        positive_sentence2.append(s2)

    elif score<=low_threshold:
        negative_sentence1.append(s1)
        negative_sentence2.append(s2)



# 5. Review what we got
print(f"\n--- Data Extraction Report ---")
print(f"Total Positive Pairs found: {len(positive_sentence1)}")
print(f"Total Negative Pairs found: {len(negative_sentence1)}")

print("\n[Example Positive Pair] (Use for Alignment):")
print(f"A Positive Pairs: {positive_sentence1[0],positive_sentence2[0]}")

print("\n[Example Negative Pair] (Use for Uniformity):")
print(f"B Negative Pairs: {negative_sentence1[0],negative_sentence2[0]}")

print("\n Generating Embedding For 1st Sentences of the positive pairs:")
embedding_list_p1 = get_embeddings_batch("embeddinggemma:latest", positive_sentence1)
print(f"B Negative Pairs: {len(embedding_list_p1)}")

print("\n Generating Embedding For 2st Sentences of the positive pairs:")
embedding_list_p2 = get_embeddings_batch("embeddinggemma:latest", positive_sentence2)
print(f"B Negative Pairs: {len(embedding_list_p2)}")

if len(embedding_list_p1) == len(embedding_list_p2):
    print("\n NO Embedding Lost For Positive Pairs:")

embedding_list_p1= np.array(embedding_list_p1)
embedding_list_p2 = np.array(embedding_list_p2)

normalized_embedding_list_p1 = normalize(embedding_list_p1, norm='l2')
normalized_embedding_list_p2 = normalize(embedding_list_p2, norm='l2')

# 1. Find the difference between each pair
# (Subtracts vector 2 from vector 1 for every item in the list)
diff = normalized_embedding_list_p1 - normalized_embedding_list_p2

# 2. Square the differences
# (This ensures all values are positive and punishes larger differences)
squared_diff = diff ** 2

# 3. Sum the dimensions to get the squared distance for each pair
# (axis=1 means we add up all 768 or 1024 dimensions for each specific sentence pair)
pair_distances = np.sum(squared_diff, axis=1)

# 4. Calculate the Alignment Loss (The Average)
alignment_loss = np.mean(pair_distances)

print(f"Alignment Loss: {alignment_loss:.4f}")