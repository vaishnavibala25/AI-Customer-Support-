# AI Customer Support Agent

An AI-powered customer support copilot built for the Hiver SDE Intern Assignment.

The system classifies incoming customer messages, retrieves historically similar AmazonHelp conversations, generates an evidence-grounded response, checks the response against historical evidence, and recommends whether the case should be auto-handled or escalated to a human.

> **Important:** The historical dataset represents AmazonHelp Twitter support conversations from the dataset period. Historical responses are used as evidence of past resolution patterns, not as proof of current Amazon policies or live customer/account information.

---

## 1. Problem

Customer support messages are often short, noisy, ambiguous, and context-dependent.

The goal of this project is to build a support copilot that can:

1. Classify an incoming customer message into a small set of support intents.
2. Retrieve historically similar customer-support cases.
3. Generate a concise response grounded in those historical cases.
4. Check whether the generated response is consistent with the retrieved evidence.
5. Decide whether the response can be auto-handled or should be escalated to a human, with a reason.

---

## 2. Dataset

The primary dataset is the **Customer Support on Twitter (CST)** dataset.

The target brand selected for this project is **AmazonHelp** because it provides a sufficiently large number of historical support interactions.

The full dataset contains millions of tweets, so a focused AmazonHelp subset was reconstructed into customer-agent conversation pairs.

### Retrieval dataset

The project uses:

* `amazonhelp_retrieval.csv`
* 4,411 historical customer-agent examples

Each example contains a customer message and the corresponding historical agent response.

The original full dataset was not included in the repository because the assignment explicitly allows and encourages working with a subsample.

---

## 3. System Architecture


                    Customer Query
                          |
                          v
                 Intent Classification
                          |
                          v
                  Query Embedding
                          |
                          v
                    FAISS Search
                          |
                          v
              Top Historical Cases
                          |
                          v
                  Grounded Prompt
                          |
                          v
                    LLM Generation
                          |
                          v
                  Suggested Response
                          |
                          v
                 Grounding Checker
                          |
                          v
                  Handling Decision
                    /          \
                   /            \
           AUTO_HANDLE        ESCALATE


---

## 4. Components

### Intent Classification

The project defines 14 support intents:

1. `delivery_issue`
2. `order_tracking_status`
3. `return_issue`
4. `refund_issue`
5. `cancellation`
6. `payment_billing`
7. `account_login`
8. `prime`
9. `promotion_discount`
10. `product_device_issue`
11. `delivery_misdelivery`
12. `security_fraud`
13. `general_amazon_query`
14. `general_unclear`

The classifier uses the `all-MiniLM-L6-v2` sentence embedding model.

Each intent contains representative examples. The incoming message is embedded and compared with the examples using cosine similarity.

The highest-scoring intent is selected.

---

### Historical Retrieval

The incoming customer message is converted into an embedding.

FAISS is used to retrieve the top three historically similar AmazonHelp customer messages.

The retrieved customer-agent pairs are then supplied to the LLM as historical evidence.

---

### Response Generation

The system uses an OpenRouter-hosted LLM.

The prompt instructs the model to:

* write a concise customer-facing response
* use historical examples as guidance
* avoid blindly copying historical responses
* avoid inventing unsupported facts
* ask for necessary information when evidence is insufficient
* avoid claiming actions that the system cannot perform

The API model used during development was:

`inclusionai/ling-3.0-flash-sante:free`

---

### Grounding Check

The grounding checker performs a deterministic evidence/action consistency check.

It identifies whether actions suggested by the generated response are supported by the retrieved historical responses.

Possible results include:

* `SUPPORTED`
* `PARTIALLY_SUPPORTED`
* `UNSUPPORTED`
* `UNKNOWN`

This should be interpreted as **historical evidence consistency**, not verification of current Amazon policy.

---

### Handling Decision

The decision layer uses conservative escalation rules.

Cases may be escalated when:

* intent confidence is below `0.70`
* the intent is `general_unclear`
* the issue involves security or account-sensitive information
* the issue may require order, payment, or refund investigation
* historical evidence is unavailable
* the generated response is insufficiently grounded
* the response requires customer-specific information or further investigation

---

# 5. Example

### Input


My package has not arrived yet


### Intent


delivery_issue


### Historical Evidence


Similarity: 0.8676
Customer: My package didn’t arrive

Similarity: 0.8001
Customer: Did not received my package .

Similarity: 0.7473
Customer: Hey so my package has not arrived yet.


### Example Generated Response

I'm sorry to hear your package hasn't arrived yet. To help investigate this,
could you please provide your order number? Alternatively, you can reach our
support team directly via phone or chat for a more detailed investigation.


### Grounding


SUPPORTED


### Handling Decision


ESCALATE


### Reason


Response requires customer-specific information or further investigation.


---

# 6. Evaluation

The intent classifier was evaluated on a manually labelled golden set of 210 examples.

The golden set contains 15 candidate examples for each of the 14 intents.

## Intent Classification Results

| Approach             | Accuracy | Macro Precision | Macro Recall | Macro F1 |
| -------------------- | -------: | --------------: | -----------: | -------: |
| Majority baseline    |   18.10% |           1.29% |        7.14% |    2.19% |
| Keyword baseline     |   49.05% |          59.51% |       50.24% |   52.08% |
| Embedding classifier |   57.62% |          57.62% |       66.11% |   58.85% |

The embedding-based classifier improves accuracy and Macro F1 over both the trivial majority baseline and the simple keyword baseline on this evaluation set.

The keyword baseline has higher Macro Precision, showing that accuracy alone does not capture every aspect of classifier behaviour.

---

# 7. Golden Evaluation Set

The project uses:


golden_candidates.csv


with 210 manually labelled examples.

Candidate selection combined classifier-driven high-similarity examples with randomly selected examples from each predicted intent.

This was designed to expose difficult and borderline cases rather than to create a statistically representative sample of all AmazonHelp traffic.

The labels were manually reviewed and assigned using the project's intent definitions.

---

# 8. Reply Evaluation

The project also includes an LLM-as-judge evaluation for generated replies.

The judge evaluates:

* Relevance
* Actionability
* Grounding
* Overall quality

A human audit was performed on 10 successfully judged examples.

| Metric        | Human Mean | LLM Judge Mean |
| ------------- | ---------: | -------------: |
| Relevance     |       3.70 |           4.90 |
| Actionability |       3.55 |           4.60 |
| Grounding     |       3.80 |           4.50 |
| Overall       |       3.75 |           4.50 |

The human audit showed that the LLM judge tended to assign higher scores than the human evaluator.

Exact agreement was limited:

| Metric        | Exact Agreement | Agreement Within ±1 |
| ------------- | --------------: | ------------------: |
| Relevance     |             10% |                 70% |
| Actionability |             10% |                 60% |
| Grounding     |             20% |                 70% |
| Overall       |             20% |                 60% |

Therefore, the LLM judge is treated as a supplementary evaluation signal rather than a replacement for human evaluation.

The reply evaluation contains fewer successful LLM-judged cases than originally planned because the free LLM API quota was exhausted during evaluation.

---

# 9. Baselines

Two baselines were implemented.

### Baseline 1 — Majority Class

Always predicts the most frequent intent in the golden evaluation set.

Result:


Accuracy: 18.10%


### Baseline 2 — Keyword Classifier

Uses simple keyword/rule matching to assign intents.

Result:


Accuracy: 49.05%
Macro F1: 52.08%


### Our Embedding Classifier

Uses semantic sentence embeddings rather than exact keyword matching.

Result:


Accuracy: 57.62%
Macro F1: 58.85%


---

# 10. Top 5 Failure Modes

## 1. Delivery issues confused with return-related issues

Example:


I have still not received my courier!


True intent:


delivery_issue


Predicted:


return_issue


Hypothesis:

Delivery delays and returns share vocabulary around orders and couriers. More contrastive training examples could improve the boundary.

---

## 2. Unclear messages over-classified as security/fraud

Examples such as:


I am receiving the same mails...


and


here attaching the inbox page...

were sometimes classified as `security_fraud`.

Hypothesis:

The classifier needs stronger examples distinguishing ordinary email/account conversations from genuine phishing or fraud signals.

---

## 3. Tracking status and delivery delay overlap

Messages containing tracking information can also describe a delivery problem.

The current classifier has difficulty separating:


Where is my package?


from:


My package is late.


Hypothesis:

A two-stage decision could first determine whether the customer is asking for status or reporting a delivery failure.

---

## 4. Multilingual messages reduce reliability

Some Portuguese, French, and Italian examples were classified incorrectly.

Hypothesis:

The current intent examples and embedding approach were developed mainly around the available English-language support patterns. Multilingual examples or multilingual embeddings could improve robustness.

---

## 5. Generated replies can be generic or exceed available evidence

Some generated responses were appropriately cautious, while others could introduce assumptions or generic instructions that are not directly supported by the retrieved historical evidence.

Hypothesis:

Generation should be more strictly constrained by retrieved evidence, and unsupported claims or invented links should trigger escalation.

---

# 11. What Is Misleading About My Headline Number?

The headline result is:


57.62% intent classification accuracy


This number should **not** be interpreted as expected accuracy on all AmazonHelp customer traffic.

Reasons:

1. The evaluation contains 210 manually labelled examples.
2. The set contains 15 candidate examples for each of 14 intents rather than the natural traffic distribution.
3. Candidate selection intentionally included classifier-driven examples and borderline cases.
4. The number measures intent classification, not complete end-to-end customer support quality.
5. Reply evaluation was performed on a much smaller successfully judged sample because of the LLM API quota limitation.

Therefore:

> **57.62% is the measured intent-classification accuracy on this particular evaluation set, not a production accuracy estimate.**

---

# 12. What We Chose Not to Build

We intentionally did not build a fully autonomous customer-service agent with access to customer accounts, orders, payment systems, or internal Amazon support tools.

The system is designed as a support copilot:


Classify
   ↓
Retrieve historical evidence
   ↓
Draft response
   ↓
Check evidence consistency
   ↓
Recommend auto-handle or escalation

This keeps the project focused on capabilities that can be evaluated using the historical dataset without pretending that historical Twitter conversations provide live customer or account access.

---

# 13. What I Would Build Next Week

1. Expand the human-labelled evaluation set using independent representative sampling.
2. Improve the boundaries between delivery, return, and tracking intents.
3. Improve the distinction between unclear messages and genuine security/fraud messages.
4. Add stronger multilingual support.
5. Improve grounding verification for unsupported claims and invented links.
6. Calibrate the escalation threshold using validation data.
7. Increase human-rated response examples for better LLM-judge calibration.
8. Add regression tests for known failure cases.

---

# 14. Decision Log

### 1. Selected AmazonHelp

AmazonHelp was selected because it provides a large volume of historical support interactions.

### 2. Used a subset of the full dataset

The full dataset is very large, so a focused subset was used for practical development and evaluation.

### 3. Reconstructed customer-agent pairs

Conversation relationships in the original dataset were used to connect customer messages with historical support responses.

### 4. Used semantic embeddings

Embeddings were selected because customer messages can express the same issue using different words.

### 5. Used FAISS

FAISS provides efficient vector similarity search over historical examples.

### 6. Defined 14 intents

The taxonomy was designed around recurring support issue categories observed in the selected data.

### 7. Used an embedding classifier

Semantic similarity was used instead of an LLM-based classifier to keep classification lightweight and reproducible.

### 8. Created a 210-example golden set

15 examples were selected for each of the 14 intents.

### 9. Kept borderline examples

Difficult examples were retained because they reveal realistic classifier weaknesses.

### 10. Added two baselines

A majority baseline and keyword baseline were implemented to measure whether semantic classification provides useful improvement.

### 11. Used deterministic grounding checks

A deterministic checker was used for the final grounding layer because LLM API reliability and quota limitations made an entirely LLM-based checker less dependable.

### 12. Used conservative escalation

Security, account-sensitive, investigation-heavy, unclear, low-confidence, and insufficiently grounded cases are routed toward human review.

### 13. Treated historical responses as evidence, not current policy

The dataset represents historical support behaviour and does not provide live Amazon policies or customer information.

### 14. Treated the LLM judge as supplementary

Human evaluation showed disagreement with the LLM judge, so the judge is not treated as ground truth.

### 15. Reported limitations explicitly

Evaluation limitations, API quota limitations, and dataset sampling limitations are included rather than hidden.

---

# 15. Project Structure


AI Customer Support/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── main.py
├── generate_response.py
├── intent_classifier.py
├── grounding_checker.py
├── decision.py
├── intents.json
│
├── amazonhelp_retrieval.csv
├── customer_index.faiss
│
├── create_golden_candidates.py
├── golden_candidates.csv
│
├── evaluate_intent.py
├── baseline_majority.py
├── baseline_keyword.py
├── evaluate_reply.py
├── evaluate_system.py
│
├── intent_evaluation_results.csv
├── intent_confusion_matrix.csv
└── reply_evaluation.csv


---

# 16. Setup

Create and activate a virtual environment:

python -m venv venv


Windows:


venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


Create a `.env` file:


LLM_API_KEY=your_openrouter_api_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=inclusionai/ling-3.0-flash-sante:free


Never commit `.env` or the API key to GitHub.

---

# 17. Run the Demo

Run:


python main.py


Then enter a customer message, for example:


My package has not arrived yet


The system displays:

* detected intent
* confidence
* retrieved historical evidence
* suggested response
* grounding result
* handling decision
* escalation reason

---

# 18. Run Evaluation

Intent evaluation:


python evaluate_intent.py


Majority baseline:


python baseline_majority.py


Keyword baseline:


python baseline_keyword.py

The reply evaluation scripts are included in the repository along with the available evaluation results.

---

# 19. Reproducibility

The repository contains the processed retrieval dataset, FAISS index, intent definitions, evaluation set, evaluation scripts, and result files required to reproduce the reported experiments.

The full original CST dataset is not included because of its size. The project uses the processed AmazonHelp subset described above.

The LLM-generated response stage requires an API key and may be affected by provider availability or free-tier limits.

---

# 20. AI Coding Tools Used

AI coding assistants were used during development for code assistance, debugging, and implementation support.

All major project components were reviewed and understood before inclusion.

The final system architecture, intent definitions, evaluation methodology, failure analysis, and reported limitations were reviewed as part of the project development process.

---

# 21. Limitations

* The retrieval dataset is a historical subset rather than the complete CST dataset.
* Historical Twitter responses may not represent current Amazon support policies.
* Intent classification accuracy is limited on ambiguous and multilingual messages.
* The golden evaluation set is intentionally balanced by intent and is not representative of natural traffic distribution.
* The grounding checker evaluates evidence/action consistency rather than real-world policy truth.
* LLM-generated responses depend on external model availability.
* The free LLM API quota limited the number of successful reply evaluations.
* No live customer account, order, payment, or internal support-system integration is implemented.

---

## Conclusion

The resulting system demonstrates an end-to-end AI customer-support copilot:

Customer Message
       ↓
Intent Classification
       ↓
Historical Retrieval
       ↓
Evidence-Grounded Generation
       ↓
Grounding Check
       ↓
Auto-handle / Human Escalation


The evaluation focuses on demonstrating measurable behaviour against simple baselines while explicitly documenting failure cases and limitations.
