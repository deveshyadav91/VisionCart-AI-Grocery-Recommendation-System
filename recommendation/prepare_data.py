import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load Instacart files
products = pd.read_csv(ROOT / "data" / "instacart" / "products.csv")
aisles = pd.read_csv(ROOT / "data" / "instacart" / "aisles.csv")
departments = pd.read_csv(ROOT / "data" / "instacart" / "departments.csv")

# Merge metadata
products = products.merge(aisles, on="aisle_id")
products = products.merge(departments, on="department_id")

# Keep useful columns
metadata = products[
    [
        "product_id",
        "product_name",
        "aisle",
        "department"
    ]
]

metadata.to_csv(
    ROOT / "data" / "products_metadata.csv",
    index=False
)

print(metadata.head())