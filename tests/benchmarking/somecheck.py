import requests
import json

MODEL_NAME = 'embeddinggemma:latest'

# Your exact 5 sentences
test_sentences = [
    'A girl is styling her hair.',
    'A group of men play soccer on the beach.',
    "One woman is measuring another woman's ankle.",
    'A man is cutting up a cucumber.',
    'A man is playing a harp.'
]

print("Sending 5 sentences directly to Ollama...")

# We use the /api/embed endpoint (the modern batching endpoint for Ollama)
url = "http://localhost:11434/api/embed"
payload = {
    "model": MODEL_NAME,
    "input": test_sentences  # Sending pure python list of strings
}

try:
    response = requests.post(url, json=payload)

    # If Ollama throws an error, this will print the exact reason!
    if response.status_code != 200:
        print(f"❌ Ollama Rejected the Request! Status: {response.status_code}")
        print(f"Reason: {response.text}")
    else:
        data = response.json()
        embeddings = data.get('embeddings', [])
        print(f"✅ SUCCESS! Generated {len(embeddings)} embeddings.")
        print(f"Shape of first vector: {len(embeddings[0])} dimensions")

except Exception as e:
    print(f"❌ Python crashed before it even reached Ollama: {e}")