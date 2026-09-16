
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. LOAD INTENT DEFINITIONS
# ---------------------------------------------------------

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)


# ---------------------------------------------------------
# 2. LOAD EMBEDDING MODEL
# ---------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 3. CREATE EMBEDDINGS FOR INTENT EXAMPLES
# ---------------------------------------------------------

intent_embeddings = {}

for intent_name, intent_data in intents.items():

    examples = intent_data["examples"]

    embeddings = model.encode(
        examples,
        normalize_embeddings=True
    )

    intent_embeddings[intent_name] = embeddings


# ---------------------------------------------------------
# 4. CLASSIFY CUSTOMER MESSAGE
# ---------------------------------------------------------

def classify_intent(message):

    query_embedding = model.encode(
        [message],
        normalize_embeddings=True
    )[0]

    scores = {}

    for intent_name, embeddings in intent_embeddings.items():

        similarities = np.dot(
            embeddings,
            query_embedding
        )

        best_similarity = np.max(similarities)

        scores[intent_name] = float(best_similarity)


    # Rank all intents
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    best_intent = ranked[0][0]
    best_score = ranked[0][1]

    second_intent = ranked[1][0]
    second_score = ranked[1][1]


    return {
        "intent": best_intent,
        "confidence": best_score,
        "second_best": second_intent,
        "second_score": second_score,
        "top_3": ranked[:3]
    }
