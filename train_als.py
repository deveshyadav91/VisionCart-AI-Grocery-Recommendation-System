from pathlib import Path
import pandas as pd
import scipy.sparse as sparse
import joblib

from implicit.als import AlternatingLeastSquares

ROOT = Path(__file__).resolve().parents[1]

print("Loading data...")

orders = pd.read_csv(
    ROOT / "data" / "instacart" / "orders.csv",
    usecols=["order_id", "user_id"]
)

prior = pd.read_csv(
    ROOT / "data" / "instacart" / "order_products__prior.csv",
    usecols=["order_id", "product_id"]
)

print("Merging...")

data = prior.merge(orders, on="order_id")

print(data.head())
print("Encoding IDs...")

user_codes = data["user_id"].astype("category")
item_codes = data["product_id"].astype("category")

data["user_idx"] = user_codes.cat.codes
data["item_idx"] = item_codes.cat.codes

user_mapping = dict(enumerate(user_codes.cat.categories))
item_mapping = dict(enumerate(item_codes.cat.categories))

print("Users:", len(user_mapping))
print("Items:", len(item_mapping))
print("Building sparse matrix...")

matrix = sparse.csr_matrix(
    (
        [1] * len(data),
        (
            data["user_idx"],
            data["item_idx"]
        )
    ),
    shape=(len(user_mapping), len(item_mapping))
)



print(matrix.shape)
print("Training ALS...")

model = AlternatingLeastSquares(

    factors=64,
    regularization=0.01,
    iterations=20,
    random_state=42

)

model.fit(matrix)

print("Training Complete!")
joblib.dump(
    model,
    ROOT / "models" / "als_model.pkl"
)

joblib.dump(
    user_mapping,
    ROOT / "models" / "user_mapping.pkl"
)

joblib.dump(
    item_mapping,
    ROOT / "models" / "item_mapping.pkl"
)

print("Saved model.")
