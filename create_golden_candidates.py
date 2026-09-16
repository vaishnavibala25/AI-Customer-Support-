import pandas as pd
from intent_classifier import classify_intent


# ---------------------------------------------------------
# 1. LOAD AMAZONHELP DATA
# ---------------------------------------------------------

df = pd.read_csv("amazonhelp_retrieval.csv")

df = df[
    ["customer_message"]
].dropna().drop_duplicates()


print("Available messages:", len(df))


# ---------------------------------------------------------
# 2. CLASSIFY EVERY MESSAGE
# ---------------------------------------------------------

results = []

for i, message in enumerate(df["customer_message"]):

    result = classify_intent(message)

    results.append({
        "customer_message": message,
        "suggested_intent": result["intent"],
        "similarity": result["confidence"]
    })

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} messages")


classified = pd.DataFrame(results)


# ---------------------------------------------------------
# 3. SAMPLE CANDIDATES FOR EACH INTENT
# ---------------------------------------------------------

golden_parts = []

for intent in classified["suggested_intent"].unique():

    candidates = classified[
        classified["suggested_intent"] == intent
    ].sort_values(
        "similarity",
        ascending=False
    )

    # Take a mixture of strong and weaker examples
    if len(candidates) > 15:

        strong = candidates.head(8)

        remaining = candidates.iloc[8:]

        random_part = remaining.sample(
            min(7, len(remaining)),
            random_state=42
        )

        selected = pd.concat(
            [strong, random_part]
        )

    else:
        selected = candidates

    selected = selected.head(15)

    golden_parts.append(selected)


golden = pd.concat(
    golden_parts,
    ignore_index=True
)


# ---------------------------------------------------------
# 4. ADD MANUAL LABEL COLUMN
# ---------------------------------------------------------

golden["true_intent"] = ""


# ---------------------------------------------------------
# 5. SAVE
# ---------------------------------------------------------

golden.to_csv(
    "golden_candidates.csv",
    index=False
)


print("\n============================================")
print("Golden candidate set created")
print("============================================")
print("Rows:", len(golden))
print("File: golden_candidates.csv")
print("\nSuggested intent distribution:")
print(golden["suggested_intent"].value_counts())
