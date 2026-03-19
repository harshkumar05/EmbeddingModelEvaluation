import ollama
import requests

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


def get_embeddings_batch_via_api(model_name, text_list):
    """
    Safely fetches embeddings from local Ollama.
    Automatically handles punctuation, special characters, and batching.
    """
    url = "http://localhost:11434/api/embed"

    # Let the requests library handle the JSON formatting safely!
    payload = {
        "model": model_name,
        "input": text_list
    }

    try:
        # Added a 120-second timeout to prevent silent hanging
        response = requests.post(url, json=payload, timeout=120)

        if response.status_code == 200:
            # Safely extract the embeddings array
            data = response.json()
            return data.get('embeddings', [])
        else:
            print(f"❌ Ollama API Error {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Python Request Failed: {e}")
        return None