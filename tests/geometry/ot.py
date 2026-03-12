from projectUtils.ollamaUtils import get_embeddings_batch
from projectUtils.ollamaUtils import get_embedding

#print(get_embedding("embeddinggemma:latest","hi"))

texts = [
    "The sky is blue.",
    "Artificial intelligence is growing rapidly.",
    "Ollama makes running local models easy."
]

embedding_list = get_embeddings_batch("embeddinggemma:latest", texts)
print(embedding_list)