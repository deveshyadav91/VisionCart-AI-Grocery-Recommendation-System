import sys
from pathlib import Path
import random
import json
import joblib
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from recommendation.hybrid import hybrid_from_product
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    ndcg_at_k,
    hit_rate,
    coverage
)

# ============================================================
# Configuration
# ============================================================

TOP_K = 5
NUM_SAMPLES = 1000
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ============================================================
# Load Files
# ============================================================

print("Loading files...")

ground_truth = joblib.load(
    ROOT / "models" / "ground_truth.pkl"
)

# Convert keys to Python int
ground_truth = {
    int(k): [int(x) for x in v]
    for k, v in ground_truth.items()
}

products = pd.read_csv(
    ROOT / "data" / "products_metadata.csv"
)

id_to_name = dict(
    zip(
        products.product_id,
        products.product_name
    )
)

all_products = list(ground_truth.keys())

TOTAL_PRODUCTS = len(all_products)

print(f"Products loaded : {TOTAL_PRODUCTS:,}")

# ============================================================
# Random Sampling
# ============================================================

if NUM_SAMPLES > TOTAL_PRODUCTS:
    NUM_SAMPLES = TOTAL_PRODUCTS

sample_products = random.sample(
    all_products,
    NUM_SAMPLES
)

print(f"Evaluating {NUM_SAMPLES} products...\n")

# ============================================================
# Metric Storage
# ============================================================

precision_scores = []
recall_scores = []
map_scores = []
ndcg_scores = []
hitrate_scores = []

recommended_products = set()

results = []

# ============================================================
# Evaluate Sampled Products
# ============================================================

for product_id in tqdm(sample_products):

    # ---------------------------------------------
    # Ground Truth
    # ---------------------------------------------

    actual = ground_truth.get(product_id, [])

    if len(actual) == 0:
        continue

    # ---------------------------------------------
    # Generate Recommendations
    # ---------------------------------------------

    try:

        recommendations = hybrid_from_product(product_id)

    except Exception as e:

        print(f"Error for Product {product_id}: {e}")

        continue

    if len(recommendations) == 0:
        continue

    predicted = [

        int(item["product_id"])

        for item in recommendations[:TOP_K]

    ]

    # ---------------------------------------------
    # Store Recommended Products
    # ---------------------------------------------

    recommended_products.update(predicted)

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    precision = precision_at_k(

        predicted,

        actual,

        TOP_K

    )

    recall = recall_at_k(

        predicted,

        actual,

        TOP_K

    )

    ap = average_precision(

        predicted,

        actual,

        TOP_K

    )

    ndcg = ndcg_at_k(

        predicted,

        actual,

        TOP_K

    )

    hr = hit_rate(

        predicted,

        actual,

        TOP_K

    )

    precision_scores.append(precision)

    recall_scores.append(recall)

    map_scores.append(ap)

    ndcg_scores.append(ndcg)

    hitrate_scores.append(hr)

    # ---------------------------------------------
    # Save Product Result
    # ---------------------------------------------

    results.append({

        "product_id": product_id,

        "product_name":

            id_to_name.get(product_id, "Unknown"),

        "precision@5": precision,

        "recall@5": recall,

        "map@5": ap,

        "ndcg@5": ndcg,

        "hitrate@5": hr

    })


# ============================================================
# Average Metrics
# ============================================================

def safe_mean(values):

    if len(values) == 0:
        return 0

    return sum(values) / len(values)


overall = {

    "Precision@5":

        round(

            safe_mean(precision_scores),

            4

        ),

    "Recall@5":

        round(

            safe_mean(recall_scores),

            4

        ),

    "MAP@5":

        round(

            safe_mean(map_scores),

            4

        ),

    "NDCG@5":

        round(

            safe_mean(ndcg_scores),

            4

        ),

    "HitRate@5":

        round(

            safe_mean(hitrate_scores),

            4

        ),

    "Coverage":

        round(

            coverage(

                recommended_products,

                TOTAL_PRODUCTS

            ),

            4

        )

}

print("\nMetric computation complete.\n")
# ============================================================
# Print Results
# ============================================================

print("=" * 60)
print("           VisionCart Evaluation Results")
print("=" * 60)

for metric, value in overall.items():
    print(f"{metric:<20}: {value:.4f}")

print("=" * 60)
print(f"Products Evaluated : {len(results)}")
print("=" * 60)

# ============================================================
# Save Results
# ============================================================

output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

# ----------------------------
# JSON Summary
# ----------------------------

summary = {
    "configuration": {
        "top_k": TOP_K,
        "num_samples": NUM_SAMPLES,
        "random_seed": RANDOM_SEED
    },
    "metrics": overall,
    "products_evaluated": len(results)
}

with open(output_dir / "evaluation_results.json", "w") as f:
    json.dump(summary, f, indent=4)

# ----------------------------
# CSV (Per-product Metrics)
# ----------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    output_dir / "evaluation_results.csv",
    index=False
)

print("\nSaved Files")
print("-" * 60)
print(output_dir / "evaluation_results.json")
print(output_dir / "evaluation_results.csv")

print("\nEvaluation Complete!")