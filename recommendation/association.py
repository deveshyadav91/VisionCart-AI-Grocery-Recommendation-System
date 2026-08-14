from pathlib import Path
import pandas as pd

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

ROOT = Path(__file__).resolve().parents[1]

print("Loading data...")

orders = pd.read_csv(
    ROOT / "data/instacart/order_products__prior.csv",
    nrows=1000000
)

products = pd.read_csv(
    ROOT / "data/instacart/products.csv"
)

print("Creating baskets...")

basket = (
    orders
    .groupby(["order_id", "product_id"])
    .size()
    .unstack(fill_value=0)
)

basket = basket.astype(bool)

print("Mining frequent itemsets...")

frequent = fpgrowth(
    basket,
    min_support=0.001,
    use_colnames=True
)

print("Generating rules...")

rules = association_rules(
    frequent,
    metric="lift",
    min_threshold=1.0
)

rules.to_csv(
    ROOT / "models/association_rules.csv",
    index=False
)

import pickle

id_to_name = dict(zip(products.product_id, products.product_name))

with open(ROOT / "models/id_to_name.pkl", "wb") as f:
    pickle.dump(id_to_name, f)

print("Rules Generated:", len(rules))