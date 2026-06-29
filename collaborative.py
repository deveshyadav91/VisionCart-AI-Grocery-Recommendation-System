from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Load trained model
model = joblib.load(ROOT / "models" / "als_model.pkl")

# Load mappings
user_mapping = joblib.load(ROOT / "models" / "user_mapping.pkl")
item_mapping = joblib.load(ROOT / "models" / "item_mapping.pkl")

# Reverse mapping
product_to_idx = {v: k for k, v in item_mapping.items()}
idx_to_product = item_mapping
def recommend(product_id, top_k=5):

    if product_id not in product_to_idx:
        return []

    item_idx = product_to_idx[product_id]

    ids, scores = model.similar_items(
        item_idx,
        N=top_k + 1
    )

    recommendations = []

    for idx, score in zip(ids, scores):

        if idx == item_idx:
            continue

        recommendations.append({

        "product_id": int(idx_to_product[idx]),

        "product_name":
        id_to_name.get(
            idx_to_product[idx],
            "Unknown"
        ),

        "score": float(score)

        })

    return recommendations[:top_k]
import pandas as pd

products = pd.read_csv(
    ROOT / "data" / "products_metadata.csv"
)

id_to_name = dict(
    zip(products.product_id,
        products.product_name)
)
if __name__ == "__main__":

    recs = recommend(24852)

    for r in recs:
        print(r)