from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]

print("Loading Instacart dataset...")

# ---------------------------------------------------
# Load prior orders
# ---------------------------------------------------

prior = pd.read_csv(
    ROOT / "data" / "instacart" / "order_products__prior.csv",
    usecols=["order_id", "product_id"]
)

print("Building baskets...")

baskets = prior.groupby("order_id")["product_id"].apply(list)

print(f"Total baskets: {len(baskets):,}")

# ---------------------------------------------------
# Build co-occurrence counts
# ---------------------------------------------------

co_occurrence = defaultdict(Counter)

for basket in baskets:

    basket = list(set(basket))

    for item in basket:

        for other in basket:

            if item != other:

                co_occurrence[item][other] += 1

print("Building Top-20 Ground Truth...")

ground_truth = {}

TOP_N = 20

for product, counter in co_occurrence.items():

    top_products = [

        pid

        for pid, _ in counter.most_common(TOP_N)

    ]

    ground_truth[product] = top_products

print("Saving...")

joblib.dump(

    ground_truth,

    ROOT / "models" / "ground_truth.pkl"

)

print(f"Ground truth built for {len(ground_truth):,} products.")

print("Saved to models/ground_truth.pkl")