from pathlib import Path
import json
import pandas as pd

from recommendation.product_matcher import get_candidate_products
from recommendation.association_inference import recommend_from_product
from recommendation.collaborative import recommend as als_recommend


ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Configuration
# ============================================================

CONFIG = {

    "weights": {

        "association": 0.65,

        "als": 0.35

    },

    "top_k": {

    "association": 20,

    "als": 20,

    "final": 5

}

}

# ============================================================
# Product Metadata
# ============================================================

products = pd.read_csv(
    ROOT / "data" / "products_metadata.csv"
)

id_to_name = dict(
    zip(
        products.product_id,
        products.product_name
    )
)

# ============================================================
# Score Normalization
# ============================================================

def normalize(items):

    if len(items) == 0:
        return items

    values = [x["score"] for x in items]

    mn = min(values)
    mx = max(values)

    if mn == mx:

        for item in items:
            item["score"] = 1.0

        return items

    for item in items:

        item["score"] = (

            item["score"] - mn

        ) / (

            mx - mn

        )

    return items
# ============================================================
# Hybrid Recommendation from Product ID
# ============================================================

def hybrid_from_product(product_id):

    query = id_to_name.get(product_id)

    if query is None:
        return []

    final_scores = {}

    # --------------------------------------------------------
    # FP-Growth Recommendations
    # --------------------------------------------------------

    assoc = recommend_from_product(
        product_id,
        top_k=CONFIG["top_k"]["association"]
    )

    assoc = [

        {
            "product_id": item["product_id"],
            "product_name": item["product_name"],
            "score": item["lift"]
        }

        for item in assoc

    ]

    assoc = normalize(assoc)

    for item in assoc:

        pid = item["product_id"]

        final_scores[pid] = {

            "product_name": item["product_name"],

            "score":

                item["score"]

                * CONFIG["weights"]["association"]

        }

    # --------------------------------------------------------
    # ALS Recommendations
    # --------------------------------------------------------

    als = als_recommend(

        product_id,

        top_k=CONFIG["top_k"]["als"]

    )

    als = normalize(als)

    for item in als:

        pid = item["product_id"]

        if pid not in final_scores:

            final_scores[pid] = {

                "product_name": item["product_name"],

                "score": 0.0

            }

        final_scores[pid]["score"] += (

            item["score"]

            * CONFIG["weights"]["als"]

        )

    # --------------------------------------------------------
    # Convert Dictionary → List
    # --------------------------------------------------------

    merged = []

    for pid, info in final_scores.items():

        merged.append({

            "product_id": pid,

            "product_name": info["product_name"],

            "score": round(info["score"], 4)

        })

    # --------------------------------------------------------
    # Remove Query Product
    # --------------------------------------------------------

    merged = [

        item

        for item in merged

        if item["product_id"] != product_id

    ]

        # --------------------------------------------------------
    # Sort by Hybrid Score
    # --------------------------------------------------------

    merged.sort(

        key=lambda x: x["score"],

        reverse=True

    )

    return merged[:CONFIG["top_k"]["final"]]

    # --------------------------------------------------------
    # CrossEncoder Re-ranking
    # --------------------------------------------------------

    final = rerank(

        query,

        merged[:CONFIG["top_k"]["rerank"]],

        top_k=CONFIG["top_k"]["final"]

    )

    return final
# ============================================================
# Hybrid Recommendation from Category
# ============================================================

def hybrid_recommend(category):

    print("\n==========================")
    print("Category:", category)

    candidates = get_candidate_products(category, top_k=5)

    print("\nCandidates:")
    print(candidates)

    if len(candidates) == 0:
        return []

    product_id = int(candidates.iloc[0]["product_id"])

    print("\nMatched Product ID:", product_id)

    assoc = recommend_from_product(product_id, top_k=5)

    print("\nAssociation:")
    print(assoc)

    als = als_recommend(product_id, top_k=5)

    print("\nALS:")
    print(als)

    return hybrid_from_product(product_id)

# ============================================================
# Basket Recommendation
# ============================================================

def recommend_basket(basket):

    recommendations = {}

    for category in basket:

        recommendations[category] = hybrid_recommend(category)

    return recommendations


# ============================================================
# Save Recommendations
# ============================================================

def save_recommendations(basket, recommendations):

    output = {

        "basket": basket,

        "hybrid_recommendations": recommendations

    }

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "hybrid_recommendations.json"

    with open(output_path, "w") as f:

        json.dump(
            output,
            f,
            indent=4
        )

    print(f"\nRecommendations saved to:\n{output_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    basket_path = ROOT / "outputs" / "basket.json"

    if not basket_path.exists():

        print("basket.json not found!")

        exit()

    with open(basket_path, "r") as f:

        basket = json.load(f)["basket"]

    print("\nDetected Basket")
    print("=" * 60)

    for item in basket:
        print("•", item)

    recommendations = recommend_basket(basket)

    print("\nHybrid Recommendations")
    print("=" * 60)

    for category, recs in recommendations.items():

        print(f"\n{category.upper()}")

        if len(recs) == 0:

            print("No recommendations found.")

            continue

        for r in recs:

            print(

                f"• {r['product_name']} "

                f"({r['score']:.4f})"

            )

    save_recommendations(
        basket,
        recommendations
    )

    print("\nDone!")