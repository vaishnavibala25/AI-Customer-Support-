import os
import pandas as pd
import numpy as np
import faiss
import requests

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from intent_classifier import classify_intent
from decision import make_decision
from grounding_checker import check_grounding


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")

if not LLM_API_KEY:
    raise ValueError(
        "LLM_API_KEY not found. Make sure it is present in your .env file."
    )


# ============================================================
# 2. LOAD RETRIEVAL DATA
# ============================================================

print("Loading AmazonHelp data...")

df = pd.read_csv("amazonhelp_retrieval.csv")

print("Total historical examples:", len(df))


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# 4. LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index("customer_index.faiss")


# ============================================================
# 5. GET CUSTOMER QUERY
# ============================================================

query = input("\nEnter customer query: ")

print("\n============================================================")
print("CUSTOMER QUERY")
print("============================================================")
print(query)


# ============================================================
# 6. INTENT CLASSIFICATION
# ============================================================

print("\nClassifying intent...")

intent_result = classify_intent(query)

intent = intent_result["intent"]
confidence = intent_result["confidence"]

print("\n============================================================")
print("INTENT CLASSIFICATION")
print("============================================================")
print("Intent:", intent)
print("Confidence:", round(confidence, 4))
print(
    "Second best:",
    intent_result["second_best"],
    "(",
    round(intent_result["second_score"], 4),
    ")"
)


# ============================================================
# 7. CONVERT QUERY INTO EMBEDDING
# ============================================================

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)


# ============================================================
# 8. RETRIEVE TOP 3 HISTORICAL CASES
# ============================================================

scores, indices = index.search(
    query_embedding,
    3
)


print("\n============================================================")
print("HISTORICAL EVIDENCE")
print("============================================================")


evidence = []
examples = []


for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    customer_message = str(
        df.iloc[idx]["customer_message"]
    )

    agent_response = str(
        df.iloc[idx]["agent_response"]
    )

    similarity = float(score)

    print(f"\n--- Evidence {rank} ---")
    print("Similarity:", round(similarity, 4))
    print("Customer:", customer_message)
    print("Agent:", agent_response)


    # Store structured evidence for grounding checker
    evidence.append({
        "similarity": similarity,
        "customer": customer_message,
        "agent": agent_response
    })


    # Store evidence for LLM prompt
    examples.append(
        f"""
Historical Example {rank}:
Customer: {customer_message}
Agent Response: {agent_response}
"""
    )


# ============================================================
# 9. BUILD GROUNDED LLM PROMPT
# ============================================================

historical_examples = "\n".join(examples)


prompt = f"""
You are an AmazonHelp customer support assistant.

Customer message:
{query}

Detected intent:
{intent}

Below are historical AmazonHelp support conversations.

Use these historical examples as evidence for how similar
issues were handled.

{historical_examples}

Instructions:

1. Write a concise and professional customer-facing response.
2. Use the historical examples as guidance.
3. Do not blindly copy historical responses.
4. Do not invent refunds, policies, prices, dates, tracking
   information, or other unsupported facts.
5. If information is insufficient, ask for the necessary
   information or direct the customer to appropriate support.
6. Do not claim that you performed an action that you cannot perform.
7. Return ONLY the customer-facing response.
"""


# ==========================================
# ==========================================
# 9. CALL OPENROUTER
# ==========================================

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")

if not LLM_API_KEY:
    raise ValueError(
        "LLM_API_KEY not found. Make sure it is present in your .env file."
    )

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {LLM_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "inclusionai/ling-3.0-flash-sante:free",
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ],
    "temperature": 0.2,
    "max_tokens": 300
}

print("\nGenerating response...")

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )
except Exception as e:
    print("\nLLM request failed:")
    print(e)
    raise SystemExit


# ==========================================
# 10. CHECK API RESPONSE
# ==========================================

print("API Status:", response.status_code)

if response.status_code != 200:
    print("\nAPI Error:")
    print(response.text)
    raise SystemExit


result = response.json()

message = result["choices"][0]["message"]
generated_response = message.get("content")

if not generated_response:
    print("\nNo response content returned.")
    print(
        "Finish reason:",
        result["choices"][0].get("finish_reason")
    )
    raise SystemExit


# ==========================================
# 11. SHOW GENERATED RESPONSE
# ==========================================

print("\n============================================================")
print("SUGGESTED AMAZONHELP RESPONSE")
print("============================================================")

print(generated_response)


# ==========================================
# 12. CHECK HISTORICAL GROUNDING
# ==========================================

print("\nChecking historical grounding...")

grounding_result = check_grounding(
    query,
    generated_response,
    evidence
)

print("\n============================================================")
print("GROUNDING CHECK")
print("============================================================")

if isinstance(grounding_result, dict):

    grounding = grounding_result["grounding"]
    grounding_reason = grounding_result["reason"]

    print("Grounding:", grounding)
    print("Reason:", grounding_reason)

    print(
        "Supported actions:",
        grounding_result.get("supported_actions", [])
    )

    print(
        "Unsupported actions:",
        grounding_result.get("unsupported_actions", [])
    )

else:

    grounding = "UNKNOWN"

    print("Grounding:", grounding)
    print("Reason:", grounding_result)


# ==========================================
# 13. MAKE HANDLING DECISION
# ==========================================

print("\nMaking handling decision...")

decision, decision_reason = make_decision(
    intent,
    confidence,
    len(evidence)
)


# If the response cannot be grounded,
# always send it to a human.
if grounding in ["UNSUPPORTED", "UNKNOWN"]:

    decision = "ESCALATE"

    decision_reason = (
        "Generated response could not be sufficiently grounded "
        "in historical evidence."
    )


# If the response asks for customer-specific information
# or requires further investigation, escalate to a human.
investigation_phrases = [
    "provide your order number",
    "provide your account",
    "provide your delivery address",
    "need more information",
    "need additional information",
    "contact our support team",
    "reach our support team",
    "phone or chat",
    "investigate this",
    "further investigation"
]

response_lower = generated_response.lower()

requires_investigation = any(
    phrase in response_lower
    for phrase in investigation_phrases
)

if requires_investigation:

    decision = "ESCALATE"

    decision_reason = (
        "Response requires customer-specific information "
        "or further investigation."
    )

# ==========================================
# 14. FINAL SYSTEM OUTPUT
# ==========================================

print("\n============================================================")
print("HANDLING DECISION")
print("============================================================")

print("Decision:", decision)
print("Reason:", decision_reason)


print("\n============================================================")
print("FINAL SYSTEM OUTPUT")
print("============================================================")

print("Intent:", intent)
print("Confidence:", round(confidence, 4))
print("Grounding:", grounding)
print("Decision:", decision)
print("Reason:", decision_reason)

print("============================================================")