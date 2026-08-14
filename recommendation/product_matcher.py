from pathlib import Path
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------
# Load metadata
# ----------------------------------------------------------

products = pd.read_csv(
    ROOT / "data" / "products_metadata.csv"
)

# ----------------------------------------------------------
# Load FAISS index
# ----------------------------------------------------------

index = faiss.read_index(
    str(ROOT / "models" / "faiss.index")
)

# ----------------------------------------------------------
# Load embedding model
# ----------------------------------------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ----------------------------------------------------------
# Semantic Product Matching
# ----------------------------------------------------------

def get_candidate_products(query, top_k=10):

    embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(embedding)

    scores, indices = index.search(
        embedding,
        top_k
    )

    candidates = []

    for score, idx in zip(scores[0], indices[0]):

        candidates.append({

            "product_id":
                int(products.iloc[idx]["product_id"]),

            "product_name":
                products.iloc[idx]["product_name"],

            "similarity":
                float(score)

        })

    return pd.DataFrame(candidates)


# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

if __name__ == "__main__":

    print(get_candidate_products("milk"))