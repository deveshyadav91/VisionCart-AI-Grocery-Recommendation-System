from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]

def parse_frozenset(text):
    numbers = re.findall(r"\d+", text)
    return [int(x) for x in numbers]

rules = pd.read_csv(ROOT / "models" / "association_rules.csv")
products = pd.read_csv(ROOT / "data" / "products_metadata.csv")

id_to_name = dict(zip(products.product_id, products.product_name))

def recommend_from_product(product_id, top_k=10):

    recommendations = []

    for _, row in rules.iterrows():

        antecedents = parse_frozenset(row["antecedents"])
        consequents = parse_frozenset(row["consequents"])

        if product_id in antecedents:

            for pid in consequents:

                recommendations.append({
                    "product_id": int(pid),
                    "product_name": id_to_name.get(pid, "Unknown"),
                    "lift": float(row["lift"]),
                    "confidence": float(row["confidence"]),
                    "support": float(row["support"])
                })

    unique = {}

    for rec in recommendations:
        pid = rec["product_id"]
        if pid not in unique or rec["lift"] > unique[pid]["lift"]:
            unique[pid] = rec

    recommendations = sorted(
        unique.values(),
        key=lambda x: x["lift"],
        reverse=True
    )

    return recommendations[:top_k]


if __name__ == "__main__":

    recs = recommend_from_product(13176)

    for r in recs:
        print(r)