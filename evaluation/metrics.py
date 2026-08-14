import math


# ============================================================
# Precision@K
# ============================================================

def precision_at_k(predicted, actual, k=5):

    predicted = predicted[:k]

    if len(predicted) == 0:
        return 0.0

    hits = len(set(predicted) & set(actual))

    return hits / k


# ============================================================
# Recall@K
# ============================================================

def recall_at_k(predicted, actual, k=5):

    if len(actual) == 0:
        return 0.0

    predicted = predicted[:k]

    hits = len(set(predicted) & set(actual))

    return hits / len(actual)


# ============================================================
# Average Precision@K (AP)
# ============================================================

def average_precision(predicted, actual, k=5):

    predicted = predicted[:k]

    score = 0.0
    hits = 0

    for i, item in enumerate(predicted):

        if item in actual:

            hits += 1

            score += hits / (i + 1)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


# ============================================================
# DCG@K
# ============================================================

def dcg_at_k(predicted, actual, k=5):

    predicted = predicted[:k]

    dcg = 0.0

    for i, item in enumerate(predicted):

        if item in actual:

            dcg += 1 / math.log2(i + 2)

    return dcg


# ============================================================
# NDCG@K
# ============================================================

def ndcg_at_k(predicted, actual, k=5):

    dcg = dcg_at_k(predicted, actual, k)

    ideal_hits = min(len(actual), k)

    if ideal_hits == 0:
        return 0.0

    idcg = 0.0

    for i in range(ideal_hits):

        idcg += 1 / math.log2(i + 2)

    return dcg / idcg


# ============================================================
# Hit Rate@K
# ============================================================

def hit_rate(predicted, actual, k=5):

    predicted = predicted[:k]

    return float(len(set(predicted) & set(actual)) > 0)


# ============================================================
# Coverage
# ============================================================

def coverage(recommended_products, total_products):

    if total_products == 0:
        return 0.0

    return len(recommended_products) / total_products