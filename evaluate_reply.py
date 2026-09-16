import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")

BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://openrouter.ai/api/v1"
)

MODEL = os.getenv(
    "LLM_MODEL",
    "inclusionai/ling-3.0-flash-sante:free"
)


# ============================================================
# CHECK API KEY
# ============================================================

if not API_KEY:
    raise ValueError(
        "LLM_API_KEY not found. Check your .env file."
    )


# ============================================================
# LOAD GOLDEN DATA
# ============================================================

df = pd.read_csv("golden_candidates.csv")
df = df.dropna(subset=["true_intent"])

print("Total labelled examples:", len(df))


# ============================================================
# JUDGE FUNCTION
# ============================================================

def judge_reply(customer_message, generated_reply, evidence):

    evidence_text = "\n\n".join(
        [
            f"Historical Customer: {item['customer']}\n"
            f"Historical Agent: {item['agent']}\n"
            f"Similarity: {item['similarity']:.4f}"
            for item in evidence
        ]
    )

    prompt = f"""
You are evaluating an AI customer-support reply.

Your task is to judge whether the suggested reply is good
based ONLY on the customer message and historical support
evidence.

Customer message:
{customer_message}

Suggested reply:
{generated_reply}

Historical evidence:
{evidence_text}

Evaluate these four criteria from 1 to 5:

1. Relevance:
Does the reply address the customer's actual issue?

2. Actionability:
Does the reply give a useful next step?

3. Grounding:
Is the reply consistent with the historical support evidence?
Do not assume that historical evidence proves current policy.

4. Overall quality:
Considering the above, how useful is the reply?

Return ONLY valid JSON:

{{
    "relevance": <1-5>,
    "actionability": <1-5>,
    "grounding": <1-5>,
    "overall": <1-5>,
    "reason": "<short explanation>"
}}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 500
    }

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    # --------------------------------------------------------
    # DEBUG: SHOW ACTUAL API RESPONSE
    # --------------------------------------------------------

    print("\nDEBUG API RESPONSE:")
    print(json.dumps(result, indent=2))

    # --------------------------------------------------------
    # EXTRACT CONTENT
    # --------------------------------------------------------

    message = result["choices"][0]["message"]

    content = message.get("content")

    if content is None:

        # Some reasoning models may put their output elsewhere.
        reasoning = message.get("reasoning")

        if reasoning:
            print("\nModel returned reasoning but no content.")

        raise ValueError(
            "LLM returned no text content. "
            "Check the DEBUG API RESPONSE above."
        )

    # --------------------------------------------------------
    # CLEAN POSSIBLE MARKDOWN JSON
    # --------------------------------------------------------

    content = content.strip()

    if content.startswith("```json"):
        content = content[7:]

    if content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    # --------------------------------------------------------
    # PARSE JSON
    # --------------------------------------------------------

    try:

        return json.loads(content)

    except json.JSONDecodeError:

        print("\nMODEL CONTENT:")
        print(content)

        raise ValueError(
            "LLM returned text, but it was not valid JSON."
        )


# ============================================================
# TEST WITH ONE EXAMPLE
# ============================================================

customer_message = "My package has not arrived yet."

generated_reply = """
I'm sorry to hear your package hasn't arrived yet.
To help investigate this, could you please provide
your order number? Alternatively, you can reach our
support team directly via phone or chat for a more
detailed investigation.
"""

evidence = [
    {
        "customer": "My package didn’t arrive",
        "agent": "We've responded to your DM, Julian! Please be sure to review our response on that platform!",
        "similarity": 0.8676
    },
    {
        "customer": "Did not received my package.",
        "agent": "Could you please confirm if we have missed the mentioned time frame? 2/2",
        "similarity": 0.8001
    },
    {
        "customer": "Hey so my package has not arrived yet. Can I get this sorted please",
        "agent": "While we can't access order or account details through social media, we'd still like to help! Please reach us directly via phone or chat so we can investigate:",
        "similarity": 0.7473
    }
]


# ============================================================
# RUN JUDGE
# ============================================================

print("\n============================================================")
print("LLM-AS-A-JUDGE")
print("============================================================")

try:

    result = judge_reply(
        customer_message,
        generated_reply,
        evidence
    )

    print("\nJudge result:")
    print(json.dumps(result, indent=2))

except Exception as e:

    print("\nJudge failed:")
    print(str(e))