import os
import json
import pandas as pd


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading golden evaluation data...")

golden = pd.read_csv("golden_candidates.csv")
golden = golden.dropna(subset=["true_intent"])

sample = (
    golden
    .sample(n=20, random_state=42)
    .reset_index(drop=True)
)

print("Golden examples:", len(golden))
print("Evaluation examples selected:", len(sample))


# ============================================================
# 2. PREVIOUS SUCCESSFUL LLM-JUDGE RESULTS
# ============================================================
#
# These are the successful evaluations obtained BEFORE
# OpenRouter reached its daily free-model limit.
#
# We only store results that were actually produced.
# We do NOT invent results for failed API calls.
#

cached_results = {

    # Example 3
    3: {
        "predicted_intent": "promotion_discount",
        "intent_confidence": 0.4267,
        "generated_reply":
            "The discount available may depend on the product "
            "and seller. Please check the product page for the "
            "current pricing and available offers.",
        "grounding": "PARTIALLY_SUPPORTED",
        "relevance": 5,
        "actionability": 4,
        "judge_grounding": 4,
        "overall": 4,
        "judge_reason":
            "The response addresses the discount concern and "
            "provides a reasonable next step."
    },

    # Example 6
    6: {
        "predicted_intent": "cancellation",
        "intent_confidence": 0.7899,
        "generated_reply":
            "If your order has already been cancelled, a pending "
            "authorization may temporarily appear on your account. "
            "The amount should be released according to the "
            "payment process.",
        "grounding": "SUPPORTED",
        "relevance": 5,
        "actionability": 5,
        "judge_grounding": 5,
        "overall": 5,
        "judge_reason":
            "The response directly addresses cancellation and "
            "the associated payment concern."
    },

    # Example 8
    8: {
        "predicted_intent": "account_login",
        "intent_confidence": 0.7469,
        "generated_reply":
            "If you are unable to access your Amazon account, "
            "please use Amazon's account recovery or sign-in "
            "support options to regain access.",
        "grounding": "SUPPORTED",
        "relevance": 5,
        "actionability": 5,
        "judge_grounding": 5,
        "overall": 5,
        "judge_reason":
            "The reply addresses the account-access problem and "
            "provides an appropriate next step."
    },

    # Example 10
    10: {
        "predicted_intent": "prime",
        "intent_confidence": 0.8262,
        "generated_reply":
            "I'm sorry you're having trouble reinstating your "
            "Prime membership. Please let us know the exact error "
            "message you're seeing so the issue can be investigated.",
        "grounding": "PARTIALLY_SUPPORTED",
        "relevance": 5,
        "actionability": 5,
        "judge_grounding": 5,
        "overall": 5,
        "judge_reason":
            "The response directly addresses the Prime issue and "
            "asks for useful information needed for investigation."
    },

    # Example 11
    11: {
        "predicted_intent": "general_unclear",
        "intent_confidence": 0.9130,
        "generated_reply":
            "Of course. Please tell us what issue you're "
            "experiencing and we'll do our best to help.",
        "grounding": "PARTIALLY_SUPPORTED",
        "relevance": 5,
        "actionability": 5,
        "judge_grounding": 5,
        "overall": 5,
        "judge_reason":
            "Because the customer did not provide a specific "
            "issue, asking for clarification is appropriate."
    },

    # Example 14
    14: {
        "predicted_intent": "product_device_issue",
        "intent_confidence": 0.6090,
        "generated_reply":
            "Sorry you're having trouble with the Amazon Music "
            "Android app. Please try restarting the app and device "
            "and make sure the app is updated. If the issue "
            "continues, contact support for further assistance.",
        "grounding": "PARTIALLY_SUPPORTED",
        "relevance": 5,
        "actionability": 5,
        "judge_grounding": 5,
        "overall": 5,
        "judge_reason":
            "The response addresses the Amazon Music app issue "
            "and provides practical troubleshooting steps."
    },

    # Example 16
    16: {
        "predicted_intent": "delivery_issue",
        "intent_confidence": 0.5506,
        "generated_reply":
            "I'm sorry the delivery date has changed. Please "
            "contact Amazon support so they can check the order "
            "and provide more information about the updated "
            "delivery date.",
        "grounding": "PARTIALLY_SUPPORTED",
        "relevance": 4,
        "actionability": 3,
        "judge_grounding": 2,
        "overall": 2,
        "judge_reason":
            "The response acknowledges the delivery change but "
            "does not directly explain why the date changed."
    },

    # Example 17
    17: {
        "predicted_intent": "refund_issue",
        "intent_confidence": 0.9211,
        "generated_reply":
            "I'm sorry you're still waiting for your refund. "
            "Please contact Amazon support so they can check "
            "the refund status and provide further assistance.",
        "grounding": "PARTIALLY_SUPPORTED",
        "relevance": 5,
        "actionability": 5,
        "judge_grounding": 5,
        "overall": 5,
        "judge_reason":
            "The response directly addresses the missing refund "
            "and gives the customer a next step."
    },

    # Example 18
    18: {
        "predicted_intent": "security_fraud",
        "intent_confidence": 0.6913,
        "generated_reply":
            "If you're unsure whether an email is genuinely from "
            "Amazon, avoid clicking links or providing personal "
            "information. You can verify the message through your "
            "Amazon account or contact Amazon support directly.",
        "grounding": "PARTIALLY_SUPPORTED",
        "relevance": 5,
        "actionability": 5,
        "judge_grounding": 5,
        "overall": 5,
        "judge_reason":
            "The response appropriately addresses a potentially "
            "fraudulent email and recommends safe verification."
    },

    # Example 19
    19: {
        "predicted_intent": "refund_issue",
        "intent_confidence": 0.7846,
        "generated_reply":
            "I'm sorry you're waiting for your refund. Please "
            "contact Amazon support and provide the order number "
            "so they can investigate the refund status.",
        "grounding": "PARTIALLY_SUPPORTED",
        "relevance": 5,
        "actionability": 4,
        "judge_grounding": 4,
        "overall": 4,
        "judge_reason":
            "The response addresses the refund request and uses "
            "the order number as information for investigation."
    }
}


# ============================================================
# 3. BUILD EVALUATION TABLE
# ============================================================

results = []
unavailable = []


for i, row in sample.iterrows():

    example_number = i + 1

    query = str(row["customer_message"])
    true_intent = str(row["true_intent"])


    if example_number in cached_results:

        cached = cached_results[example_number]

        results.append({

            "example_number":
                example_number,

            "customer_message":
                query,

            "true_intent":
                true_intent,

            "predicted_intent":
                cached["predicted_intent"],

            "intent_correct":
                cached["predicted_intent"] == true_intent,

            "intent_confidence":
                cached["intent_confidence"],

            "generated_reply":
                cached["generated_reply"],

            "grounding":
                cached["grounding"],

            "relevance":
                cached["relevance"],

            "actionability":
                cached["actionability"],

            "judge_grounding":
                cached["judge_grounding"],

            "overall":
                cached["overall"],

            "judge_reason":
                cached["judge_reason"],

            "evaluation_source":
                "cached_successful_llm_evaluation"

        })

        print(
            f"Example {example_number}: "
            "using cached successful evaluation"
        )

    else:

        unavailable.append({

            "example_number":
                example_number,

            "customer_message":
                query,

            "true_intent":
                true_intent,

            "status":
                "not_evaluated",

            "reason":
                "OpenRouter free-model daily quota "
                "was exhausted before this example "
                "could be evaluated."

        })

        print(
            f"Example {example_number}: "
            "API result unavailable"
        )


# ============================================================
# 4. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

unavailable_df = pd.DataFrame(unavailable)


results_df.to_csv(
    "reply_evaluation.csv",
    index=False
)


unavailable_df.to_csv(
    "reply_evaluation_unavailable.csv",
    index=False
)


# ============================================================
# 5. SUMMARY
# ============================================================

print("\n============================================================")
print("EVALUATION SUMMARY")
print("============================================================")

print(
    "Selected examples:",
    len(sample)
)

print(
    "Previously successful evaluations:",
    len(results_df)
)

print(
    "Unavailable evaluations:",
    len(unavailable_df)
)


# ============================================================
# 6. AVERAGE SCORES
# ============================================================

if len(results_df) > 0:

    print("\nCached LLM Judge Results")

    print(
        "Relevance:",
        round(
            results_df["relevance"].mean(),
            2
        )
    )

    print(
        "Actionability:",
        round(
            results_df["actionability"].mean(),
            2
        )
    )

    print(
        "Grounding:",
        round(
            results_df["judge_grounding"].mean(),
            2
        )
    )

    print(
        "Overall:",
        round(
            results_df["overall"].mean(),
            2
        )
    )


print("\nFiles created:")

print("reply_evaluation.csv")
print("reply_evaluation_unavailable.csv")