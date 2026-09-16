# grounding_checker.py

import re


# Words that represent actions commonly seen in support responses
ACTION_GROUPS = {
    "investigate": {
        "investigate",
        "investigation",
        "check",
        "look into",
        "review"
    },

    "contact_support": {
        "contact",
        "reach",
        "phone",
        "chat",
        "support",
        "directly"
    },

    "return": {
        "return",
        "send back",
        "returning"
    },

    "refund": {
        "refund",
        "refunded",
        "money back"
    },

    "track": {
        "track",
        "tracking",
        "delivery status",
        "shipment status"
    },

    "cancel": {
        "cancel",
        "cancellation"
    }
}


def normalize(text):
    """Convert text into lowercase normalized form."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def find_supported_actions(generated_reply, evidence):

    reply = normalize(generated_reply)

    historical_text = " ".join(
        normalize(item["agent"])
        for item in evidence
    )

    supported_actions = []
    unsupported_actions = []

    # Check investigation / checking
    investigation_words = [
        "investigate",
        "investigation",
        "look into",
        "check",
        "review"
    ]

    if any(word in reply for word in investigation_words):

        if any(word in historical_text for word in investigation_words):

            supported_actions.append("investigate")

        else:
            unsupported_actions.append("investigate")


    # Check contacting support
    contact_words = [
        "contact",
        "reach",
        "phone",
        "chat",
        "support",
        "directly"
    ]

    if any(word in reply for word in contact_words):

        if any(word in historical_text for word in contact_words):

            supported_actions.append("contact_support")

        else:
            unsupported_actions.append("contact_support")


    # Check return
    return_words = [
        "return",
        "send back",
        "returning"
    ]

    if any(word in reply for word in return_words):

        if any(word in historical_text for word in return_words):

            supported_actions.append("return")

        else:
            unsupported_actions.append("return")


    # Check refund
    refund_words = [
        "refund",
        "refunded",
        "money back"
    ]

    if any(word in reply for word in refund_words):

        if any(word in historical_text for word in refund_words):

            supported_actions.append("refund")

        else:
            unsupported_actions.append("refund")


    # Check cancellation
    cancel_words = [
        "cancel",
        "cancellation"
    ]

    if any(word in reply for word in cancel_words):

        if any(word in historical_text for word in cancel_words):

            supported_actions.append("cancel")

        else:
            unsupported_actions.append("cancel")


    return supported_actions, unsupported_actions



def check_grounding(
    customer_message,
    generated_reply,
    evidence
):

    # No evidence means we cannot prove grounding
    if not evidence:

        return {
            "grounding": "UNSUPPORTED",
            "reason": "No historical evidence was retrieved.",
            "supported_actions": [],
            "unsupported_actions": []
        }


    supported_actions, unsupported_actions = find_supported_actions(
        generated_reply,
        evidence
    )


    # If the response contains unsupported actions
    if unsupported_actions:

        return {
            "grounding": "PARTIALLY_SUPPORTED",
            "reason": (
                "The response contains actions that are not "
                "supported by the retrieved historical evidence."
            ),
            "supported_actions": supported_actions,
            "unsupported_actions": unsupported_actions
        }


    # If at least one meaningful action is supported
    if supported_actions:

        return {
            "grounding": "SUPPORTED",
            "reason": (
                "The response contains actions consistent with "
                "the retrieved historical support responses."
            ),
            "supported_actions": supported_actions,
            "unsupported_actions": []
        }


    # Evidence exists but no recognizable action was found
    return {
        "grounding": "PARTIALLY_SUPPORTED",
        "reason": (
            "Relevant historical evidence exists, but no "
            "specific supported support action was detected."
        ),
        "supported_actions": [],
        "unsupported_actions": []
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_evidence = [

        {
            "similarity": 0.868,
            "customer": "My package didn't arrive",
            "agent": "Please reach us directly so we can investigate."
        },

        {
            "similarity": 0.800,
            "customer": "Did not receive my package.",
            "agent": "Could you please confirm if we have missed the mentioned time frame?"
        },

        {
            "similarity": 0.748,
            "customer": "My package has not arrived yet.",
            "agent": "We'd like to help. Please reach us directly so we can investigate."
        }
    ]


    customer_message = "My package has not arrived yet."


    generated_reply = (
        "I'm sorry to hear your package hasn't arrived yet. "
        "To help investigate this for you, could you please provide "
        "your order number or the email address associated with your "
        "Amazon account? This will allow me to look into the delivery "
        "status and assist you further."
    )


    result = check_grounding(
        customer_message,
        generated_reply,
        test_evidence
    )


    print("\nGrounding Result:")
    print(result)
