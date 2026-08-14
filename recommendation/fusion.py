import numpy as np
from collections import defaultdict

# ============================================================
# Score Normalization (per-source safe scaling)
# ============================================================

def normalize_scores(scores_dict):
    """
    Input format:
    {
        "faiss": [(pid, score), ...],
        "als": [(pid, score), ...],
        "association": [(pid, score), ...]
    }
    """

    normalized = {}

    for source, items in scores_dict.items():
        if not items:
            continue

        values = np.array([s for _, s in items])

        min_s = values.min()
        max_s = values.max()

        denom = (max_s - min_s) + 1e-8

        normalized[source] = [
            (pid, (score - min_s) / denom)
            for pid, score in items
        ]

    return normalized


# ============================================================
# Fusion Feature Builder
# ============================================================

def build_fusion_table(normalized_scores):
    """
    Converts multi-source signals into unified feature table
    """

    table = defaultdict(lambda: {
        "faiss": 0.0,
        "als": 0.0,
        "association": 0.0,
        "count": 0
    })

    for source, items in normalized_scores.items():
        for pid, score in items:

            table[pid][source] = max(table[pid][source], score)
            table[pid]["count"] += 1

    return table


# ============================================================
# Feature Vector Builder (for ML rankers later)
# ============================================================

def to_features(table):
    """
    Converts fusion table into ML-ready matrix
    """

    X = []
    items = []

    for pid, f in table.items():
        X.append([
            f["faiss"],
            f["als"],
            f["association"],
            f["count"]
        ])
        items.append(pid)

    return np.array(X), items


# ============================================================
# Rule-based Fusion (current production fallback)
# ============================================================

def rule_fuse(table, weights=None):
    """
    Safe default fusion (no ML required)
    """

    if weights is None:
        weights = {
            "faiss": 0.5,
            "als": 0.3,
            "association": 0.2
        }

    results = []

    for pid, f in table.items():

        score = (
            weights["faiss"] * f["faiss"] +
            weights["als"] * f["als"] +
            weights["association"] * f["association"]
        )

        # agreement boost
        score *= (1 + 0.1 * f["count"])

        results.append({
            "product_id": pid,
            "score": float(score)
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results


# ============================================================
# ML-based Fusion (optional upgrade path)
# ============================================================

def ml_fuse(model, X, items):
    """
    Plug-in for XGBoost / LightGBM / Logistic models
    """

    scores = model.predict_proba(X)[:, 1]

    ranked = sorted(
        zip(items, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {"product_id": pid, "score": float(score)}
        for pid, score in ranked
    ]


# ============================================================
# Final Fusion Entry Point
# ============================================================

def fuse_recommendations(scores_dict, model=None, use_ml=False):
    """
    Main API for hybrid system

    Parameters:
    - scores_dict: raw outputs from FAISS / ALS / FP-Growth
    - model: trained ML ranker (optional)
    - use_ml: switch between rule-based or ML-based fusion
    """

    # Step 1: Normalize
    normalized = normalize_scores(scores_dict)

    # Step 2: Build unified table
    table = build_fusion_table(normalized)

    # Step 3: ML-based fusion (optional)
    if use_ml and model is not None:

        X, items = to_features(table)
        return ml_fuse(model, X, items)

    # Step 4: Rule-based fusion (default safe mode)
    return rule_fuse(table)