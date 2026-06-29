from pathlib import Path
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

products = pd.read_csv(ROOT / "data" / "products_metadata.csv")

index = faiss.read_index(str(ROOT / "models" / "faiss.index"))

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def recommend(query, top_k=5):

    embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    faiss.normalize_L2(embedding)

    scores, indices = index.search(embedding, top_k)

    recommendations = []

    for score, idx in zip(scores[0], indices[0]):

        recommendations.append({

            "product_id": int(products.iloc[idx]["product_id"]),

            "product_name": products.iloc[idx]["product_name"],

            "score": float(score)

        })

    return recommendations


if __name__ == "__main__":

    recs = recommend("milk")

    for r in recs:
        print(r)