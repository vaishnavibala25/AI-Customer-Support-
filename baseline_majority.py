import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ==========================================
# 1. LOAD GOLDEN SET
# ==========================================

df = pd.read_csv("golden_candidates.csv")
df = df.dropna(subset=["true_intent"])

print("Total examples:", len(df))


# ==========================================
# 2. FIND MAJORITY CLASS
# ==========================================

majority_intent = (
    df["true_intent"]
    .value_counts()
    .idxmax()
)

majority_count = (
    df["true_intent"]
    .value_counts()
    .max()
)

print("\nMajority intent:", majority_intent)
print("Examples in majority class:", majority_count)


# ==========================================
# 3. PREDICT MAJORITY FOR EVERYTHING
# ==========================================

predictions = [
    majority_intent
    for _ in range(len(df))
]


# ==========================================
# 4. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    df["true_intent"],
    predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    df["true_intent"],
    predictions,
    average="macro",
    zero_division=0
)


# ==========================================
# 5. RESULTS
# ==========================================

print("\n============================================================")
print("MAJORITY-CLASS BASELINE")
print("============================================================")

print("Accuracy:", round(accuracy, 4))
print("Accuracy (%):", round(accuracy * 100, 2))

print("Macro Precision:", round(precision, 4))
print("Macro Recall:", round(recall, 4))
print("Macro F1:", round(f1, 4))