# decision.py

# Intents that should normally be escalated
SENSITIVE_INTENTS = {
    "security_fraud",
    "account_login",
}

# Intents where an agent may need to investigate
INVESTIGATION_INTENTS = {
    "delivery_misdelivery",
    "refund_issue",
    "payment_billing",
}

# Minimum confidence required for automatic handling
CONFIDENCE_THRESHOLD = 0.70


def make_decision(intent, confidence, evidence_count):
    """
    Decide whether the AI should auto-handle the customer message
    or escalate it to a human agent.

    Returns:
        decision: AUTO_HANDLE or ESCALATE
        reason: explanation for the decision
    """

    # 1. Low-confidence predictions are unsafe
    if confidence < CONFIDENCE_THRESHOLD:
        return (
            "ESCALATE",
            f"Low intent confidence ({confidence:.2f})"
        )

    # 2. Unclear messages should go to a human
    if intent == "general_unclear":
        return (
            "ESCALATE",
            "Customer intent is unclear"
        )

    # 3. Security/fraud cases are sensitive
    if intent in SENSITIVE_INTENTS:
        return (
            "ESCALATE",
            "Security or account-sensitive issue requires human review"
        )

    # 4. Some issues require account/order investigation
    if intent in INVESTIGATION_INTENTS:
        return (
            "ESCALATE",
            "Issue may require account, order, payment, or refund investigation"
        )

    # 5. No historical evidence means the response cannot be grounded
    if evidence_count == 0:
        return (
            "ESCALATE",
            "No relevant historical evidence found"
        )

    # 6. Otherwise, the system can suggest a response
    return (
        "AUTO_HANDLE",
        "Clear intent with sufficient confidence and historical evidence"
    )


# Simple test
if __name__ == "__main__":

    test_cases = [
        ("delivery_issue", 0.91, 3),
        ("return_issue", 0.88, 3),
        ("security_fraud", 0.95, 3),
        ("general_unclear", 0.80, 2),
        ("delivery_issue", 0.45, 3),
        ("prime", 0.86, 0),
    ]

    for intent, confidence, evidence_count in test_cases:

        decision, reason = make_decision(
            intent,
            confidence,
            evidence_count
        )

        print("\nIntent:", intent)
        print("Confidence:", confidence)
        print("Evidence:", evidence_count)
        print("Decision:", decision)
        print("Reason:", reason)

