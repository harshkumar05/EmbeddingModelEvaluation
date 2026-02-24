import ollama

def get_embedding(model,sentence):
    try:
        response = ollama.embeddings(model=model, prompt=sentence)
        return response['embedding']
    except:
        return None

def get_embeddings_batch(model,sentences):
    try:
        response = ollama.embed(model=model, input=sentences)
        return response['embeddings']
    except:
        return None