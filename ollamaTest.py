import ollama
import numpy as np
from tests import isotropyScore
from tests.geometry import Isotropy
def get_gemma_embedding(text):
    response = ollama.embed(
        model='embeddinggemma:latest',
        input=text
    )

    return response['embeddings'][0]

def get_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


word1 = "man"
word2 = "programming"
print(get_gemma_embedding(word1))


vec1 = get_gemma_embedding(word1)
vec2 = get_gemma_embedding (word2)
print("Length of the vector is: ", len(vec1))
print("Length of the vector is: ", len(vec2))
print(vec1)
print(vec2)
print(get_cosine_similarity(vec1, vec2))
print(f"Similarity: {get_cosine_similarity(vec1, vec2):.4f}")

list_a = [1, 2, 3]
list_b = [4, 5, 6]

# Create a list of lists
matrix = [list_a, list_b]

isotropyScore.test_isotropy(matrix)
Isotropy = Isotropy(matrix)
print(Isotropy)
#print(get_cosine_similarity(x, y))