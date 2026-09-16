import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# KEYWORD-BASED BASELINE
# ============================================================

df = pd.read_csv("golden_candidates.csv")
df = df.dropna(subset=["true_intent"])

print("Total examples:", len(df))


def keyword_predict(message):

    message = str(message).lower()


    # Delivery issue
    if any(word in message for word in [
        "late", "delayed", "not arrived", "didn't arrive",
        "did not arrive", "not received", "haven't received",
        "hasn't arrived", "where is my package"
    ]):
        return "delivery_issue"


    # Tracking
    if any(word in message for word in [
        "tracking", "track my", "tracking number",
        "tracking id", "shipment status"
    ]):
        return "order_tracking_status"


    # Return
    if any(word in message for word in [
        "return", "send back", "returning", "pickup"
    ]):
        return "return_issue"


    # Refund
    if any(word in message for word in [
        "refund", "money back", "refunded"
    ]):
        return "refund_issue"


    # Cancellation
    if any(word in message for word in [
        "cancel", "cancellation"
    ]):
        return "cancellation"


    # Payment / billing
    if any(word in message for word in [
        "payment", "charged", "charge", "billing",
        "credit card", "debit card", "duplicate charge"
    ]):
        return "payment_billing"


    # Account / login
    if any(word in message for word in [
        "login", "log in", "sign in", "password",
        "account access", "can't access my account"
    ]):
        return "account_login"


    # Prime
    if "prime" in message:
        return "prime"


    # Promotion / discount
    if any(word in message for word in [
        "discount", "coupon", "promo", "promotion",
        "cashback", "voucher"
    ]):
        return "promotion_discount"


    # Product / device
    if any(word in message for word in [
        "broken", "damaged", "defective", "not working",
        "wrong product", "wrong item"
    ]):
        return "product_device_issue"


    # Security / fraud
    if any(word in message for word in [
        "fraud", "phishing", "scam", "suspicious",
        "hacked", "unauthorized"
    ]):
        return "security_fraud"


    # General Amazon question
    if any(word in message for word in [
        "amazon", "website", "available", "availability",
        "cash on delivery", "cod"
    ]):
        return "general_amazon_query"


    # Fallback
    return "general_unclear"


predictions = [
    keyword_predict(message)
    for message in df["customer_message"]
]


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


print("\n============================================================")
print("KEYWORD-BASED BASELINE")
print("============================================================")
print("Accuracy:", round(accuracy, 4))
print("Accuracy (%):", round(accuracy * 100, 2))
print("Macro Precision:", round(precision, 4))
print("Macro Recall:", round(recall, 4))
print("Macro F1:", round(f1, 4))