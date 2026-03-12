import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from datasets import load_dataset

# Assuming this is your custom local module
from projectUtils.ollamaUtils import get_embeddings_batch

MODEL_NAME = 'embeddinggemma:latest'
SAMPLES_PER_CLASS = 300  # 1200 total dots on the map

print("Loading labeled dataset (AG News)...")
# ag_news has 4 clear classes: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
dataset = load_dataset("ag_news", split="train")
df = dataset.to_pandas()

# --- 1. DATA PREP ---
print("Filtering data by category...")

# grab an equal number of samples from each category
world_df = df[df['label'] == 0].head(SAMPLES_PER_CLASS)
sports_df = df[df['label'] == 1].head(SAMPLES_PER_CLASS)
biz_df = df[df['label'] == 2].head(SAMPLES_PER_CLASS)
tech_df = df[df['label'] == 3].head(SAMPLES_PER_CLASS)

# combine and extract the text and labels
final_df = pd.concat([world_df, sports_df, biz_df, tech_df])
sentences = final_df['text'].tolist()
labels = final_df['label'].tolist()

# --- 2. GENERATE EMBEDDINGS ---
print(f"Generating {len(sentences)} embeddings (this might take a minute)...")
embeddings = np.array(get_embeddings_batch(MODEL_NAME, sentences))

# --- 3. UMAP REDUCTION ---
# squashing 768 dimensions down to exactly 2 dimensions
print("Running UMAP dimensionality reduction...")

# n_neighbors controls how UMAP balances local vs global structure (15 is standard)
# min_dist controls how tightly UMAP packs points together (0.1 is standard)
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# --- 4. PLOTTING ---
print("Plotting the results...")

plt.figure(figsize=(10, 8))
categories = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple'}

# plot each category one by one so we get a nice legend
for label_id, label_name in categories.items():
    # find all indices where the label matches
    idx = [i for i, lbl in enumerate(labels) if lbl == label_id]

    # scatter plot those specific points
    plt.scatter(
        embeddings_2d[idx, 0],  # X coordinates
        embeddings_2d[idx, 1],  # Y coordinates
        c=colors[label_id],
        label=label_name,
        alpha=0.6,  # slight transparency so we can see overlaps
        s=15  # dot size
    )

plt.title(f"UMAP Projection of AG News using {MODEL_NAME}")
plt.legend()
plt.tight_layout()
plt.show()