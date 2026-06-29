from pathlib import Path
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

# Load products
products = pd.read_csv(
    ROOT / "data" / "products_metadata.csv"
)

# Use product names
texts = products["product_name"].astype(str).tolist()

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Normalize
faiss.normalize_L2(embeddings)

# Build index
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

# Save
faiss.write_index(
    index,
    str(ROOT / "models" / "faiss.index")
)

np.save(
    ROOT / "models" / "product_embeddings.npy",
    embeddings
)

print("Done")
print("Products:", len(products))