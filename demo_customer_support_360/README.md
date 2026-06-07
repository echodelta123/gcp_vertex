# Customer Support 360 Demo

> **Goal:** Synthesize multi-turn retail customer support logs into *actionable personas* using Retrieval‑Augmented Generation (RAG) and Gemini structured output.

## Business Context

Fashion retailers (like H&M) receive large streams of unstructured support interactions (emails, chats, social media public posts, and DMs). The demo shows how to:

- **Ingest** raw logs (JSONL) into a vector store (Weaviate) with rich metadata (customer ID, issue type, timestamp).
- **Retrieve** the top‑k most relevant interactions for a given query using hybrid keyword + semantic search.
- **Synthesize** a concise **persona** (pain points, sentiment, recommended actions) via Gemini Flash (`structured_output=True`).

The result mimics a **customer‑experience manager** dashboard that can prioritize support tickets or design proactive service improvements.

---

## Data Source

This demo is based on the Kaggle **H&M Personalized Fashion Recommendations** competition dataset: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations. The pipeline mirrors the three core raw inputs from that source - customer reviews, transactions, and article metadata.

---

## Architecture Overview

Data Ingestion
+--------------------------------+
| Raw Support Logs (JSONL)       |
| Log Parser (Python)            |
+--------------------------------+
              |
              v
Vector Store
+----------------------+
| Weaviate (v3)        |
+----------------------+
              |
              v
Retrieval + Generation
+------------------------------------------------------+
| User Query (FastAPI)                                 |
| Hybrid Search (keyword + embeddings)                 |
| Gemini Structured Output                             |
+------------------------------------------------------+
              |
              v
Frontend
+----------------------+
| Streamlit Dashboard  |
+----------------------+

### Key Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Weaviate** | Scalable vector DB with metadata filters | `shared/config.py` provides `WEAVIATE_URL` and `WEAVIATE_API_KEY`. The client is instantiated in `services.py`.
| **Gemini Structured Output** | Guarantees JSON‑compatible persona schema, validated with Pydantic (`PersonaResponse`). | Wrapped in `shared/gemini_client.py` with live/mock dual mode.
| **FastAPI** | Exposes `/persona` endpoint (POST query) and `/health`. | `backend.py` (router + CORS).
| **Streamlit UI** | Interactive query textbox, displays persona cards and top retrieved logs. | `frontend.py`.

---

## Getting Started

### 1. Install & Configure

```bash
cd vertex-ai-customer-demo
pip install -r requirements.txt
cp .env.example .env   # Edit if you have real Weaviate credentials
```

- If `DEMO_MODE=true` (default), the backend uses a **mock Weaviate** implementation with pre‑generated random embeddings – no external service needed.
- To use a real Weaviate instance, set `WEAVIATE_URL` and `WEAVIATE_API_KEY` in `.env`.

### 2. Load Sample Data

```bash
# Load the bundled demo logs (Kaggle Customer Support dataset) into Weaviate
python -m demo_customer_support_360.services.load_demo_data
```

The script reads `data/support_logs_sample.jsonl` and upserts each entry with fields:

```json
{ "customer_id": "C1234", "timestamp": "2023-04-01T12:34:56Z", "issue": "Shipping delay", "transcript": "..." }
```

### 3. Run the API

```bash
make demo-3-api   # FastAPI on http://localhost:8003
```

Open `http://localhost:8003/docs` for the OpenAPI UI.

### 4. Run the UI

```bash
make demo-3-ui    # Streamlit on http://localhost:8503
```

Enter a query like:

- *"What are the most common complaints about shipping and package delivery?"*
- *"Summarize the sentiment for customers returning clothing items due to size mismatches."*

---

## API Reference

### POST `/persona`

**Request** (`application/json`):
```json
{ "query": "string", "top_k": 5 }
```
- `query`: Natural‑language question.
- `top_k` (optional, default 5): Number of relevant logs to retrieve.

**Response** (`200 OK`):
```json
{
  "persona": {
    "title": "Baggage‑Handling Complaints",
    "summary": "Customers frequently report delayed baggage, leading to missed connections and negative sentiment.",
    "pain_points": ["Lost luggage", "Late delivery"],
    "sentiment_score": -0.73,
    "recommended_action": "Introduce real‑time baggage tracking and proactive notifications."
  },
  "retrieved": [
    { "customer_id": "C001", "issue": "Baggage lost", "transcript": "..." },
    ...
  ]
}
```

The `persona` object is validated against `PersonaResponse` Pydantic model.

### GET `/health`

Simple health‑check returning `{ "status": "ok" }`.

---

## Implementation Details

### Vector Store Schema

```json
{
  "class": "SupportLog",
  "properties": [
    { "name": "customer_id", "dataType": ["string"] },
    { "name": "timestamp",   "dataType": ["date"] },
    { "name": "issue",       "dataType": ["text"] },
    { "name": "transcript",  "dataType": ["text"] },
    { "name": "vector",      "dataType": ["number[]"] }
  ]
}
```

- **Hybrid Search**: Weaviate's `nearText` (semantic) + `where` (keyword filter) combined.
- **Embedding**: Gemini embeddings (`models/text-embedding-004`). In mock mode we generate deterministic vectors using a hash of the transcript.

### Gemini Prompt (Template)

```jinja2
You are an H&M fashion retail CX expert. Summarize the top themes, sentiment, and recommended actions from the following support logs:

{% for log in logs %}
- {{ log.transcript }}
{% endfor %}

Provide a JSON response matching this schema:
{{ schema_json }}
```

The prompt is stored in `prompts.py` and rendered with `jinja2`.

### Error Handling

- **Weaviate connection errors** → fallback to mock store (if `DEMO_MODE=true`).
- **Gemini JSON parsing errors** → return a 502 with a human‑readable fallback persona.
- **Input validation** → FastAPI + Pydantic ensures `query` length > 3 characters.

---

## Testing

```bash
make test   # Runs pytest suite for all demos
```

Key test cases for this demo:
- `test_generate_persona_success` – validates successful RAG generation.
- `test_vector_store_fallback_mock` – ensures the mock store loads when real Weaviate is unreachable.
- `test_input_validation` – ensures short queries return 422.

---
