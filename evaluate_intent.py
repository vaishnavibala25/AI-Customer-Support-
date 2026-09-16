import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from intent_classifier import classify_intent


# ==========================================
# 1. LOAD GOLDEN SET
# ==========================================

df = pd.read_csv("golden_candidates.csv")

print("Total golden examples:", len(df))

df = df.dropna(subset=["true_intent"])

print("Labelled examples:", len(df))


# ==========================================
# 2. RUN CLASSIFIER
# ==========================================

predictions = []
confidences = []

print("\nClassifying golden examples...")

for i, message in enumerate(df["customer_message"]):

    result = classify_intent(str(message))

    predictions.append(result["intent"])
    confidences.append(result["confidence"])

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(df)}")


df["predicted_intent"] = predictions
df["confidence"] = confidences


# ==========================================
# 3. BASIC METRICS
# ==========================================

accuracy = accuracy_score(
    df["true_intent"],
    df["predicted_intent"]
)

precision, recall, f1, support = precision_recall_fscore_support(
    df["true_intent"],
    df["predicted_intent"],
    average="macro",
    zero_division=0
)


# ==========================================
# 4. PRINT OVERALL RESULTS
# ==========================================

print("\n============================================================")
print("INTENT CLASSIFICATION RESULTS")
print("============================================================")

print("Total examples:", len(df))

print(
    "Correct:",
    int((df["true_intent"] == df["predicted_intent"]).sum())
)

print(
    "Incorrect:",
    int((df["true_intent"] != df["predicted_intent"]).sum())
)

print("Accuracy:", round(accuracy, 4))
print("Accuracy (%):", round(accuracy * 100, 2))

print("\nMacro Precision:", round(precision, 4))
print("Macro Recall:", round(recall, 4))
print("Macro F1:", round(f1, 4))


# ==========================================
# 5. PER-INTENT REPORT
# ==========================================

print("\n============================================================")
print("PER-INTENT CLASSIFICATION REPORT")
print("============================================================")

print(
    classification_report(
        df["true_intent"],
        df["predicted_intent"],
        zero_division=0
    )
)


# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

labels = sorted(
    df["true_intent"].unique()
)

cm = confusion_matrix(
    df["true_intent"],
    df["predicted_intent"],
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n============================================================")
print("CONFUSION MATRIX")
print("============================================================")

print(cm_df)


# ==========================================
# 7. SHOW INCORRECT PREDICTIONS
# ==========================================

errors = df[
    df["true_intent"] != df["predicted_intent"]
]

print("\n============================================================")
print("INCORRECT PREDICTIONS")
print("============================================================")

print("Total errors:", len(errors))

for _, row in errors.head(20).iterrows():

    print("\nCustomer:", row["customer_message"])
    print("True intent:", row["true_intent"])
    print("Predicted:", row["predicted_intent"])
    print("Confidence:", round(row["confidence"], 4))


# ==========================================
# 8. SAVE RESULTS
# ==========================================

df.to_csv(
    "intent_evaluation_results.csv",
    index=False
)

cm_df.to_csv(
    "intent_confusion_matrix.csv"
)

print("\nSaved:")
print("intent_evaluation_results.csv")
print("intent_confusion_matrix.csv")