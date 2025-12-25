import ollama
import numpy as np

def get_gemma_embedding(text):
    response = ollama.embed(
        model='embeddinggemma:latest',
        input=text
    )

    return response['embeddings'][0]

def get_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


word1 = "spider"
word2 = "programming"


vec1 = get_gemma_embedding(word1)
vec2 = get_gemma_embedding (word2)

print("word1: ", word1)
print("word2: ", word2)
print(f"Similarity: {get_cosine_similarity(vec1, vec2):.4f}")


#print(get_cosine_similarity(x, y))