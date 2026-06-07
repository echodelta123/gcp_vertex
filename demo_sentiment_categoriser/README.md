# Demo 1: Sentiment Intelligence Engine

> Aspect‑level sentiment analysis with structured LLM output — because a single "positive/negative" label isn't enough for enterprise customer feedback. It segments unstructured feedback into granular, categorized business insights.

## Problem Statement: The Feedback Black Hole - how to aggregate feedback and keep the nuance?

Customer feedback is collected across numerous disparate channels: app store reviews, support tickets, NPS surveys, and social media platforms. Standard sentiment analysis models typically summarize these feedback strings with a single, top‑level document label (such as *positive*, *negative*, or *neutral*).

However, a single label is often insufficient for nuanced customer reviews. For example:
> "The dress design is gorgeous and fits perfectly, but the zipper broke after wearing it just once. Customer support was helpful but I'm still frustrated."

A traditional sentiment classifier would mark this feedback as "Mixed" or "Neutral," obfuscating actionable insights. To be operationalized effectively, this feedback contains three distinct, category‑specific signals:
- **Product Design & Fit**: Positive
- **Product Quality & Durability**: Negative (zipper malfunction)
- **Customer Support Experience**: Positive (helpful support team)

*This demo uses curated data inspired by the Kaggle **[E‑Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews-2023)** dataset to address this problem.*

## System Overview

The Sentiment Intelligence Engine dissects customer feedback into structured, categorized data points:
1. **Aspect‑Level Extraction**: Google Gemini parses feedback text to identify explicit aspects/categories (e.g., fit, quality, price) and assigns individual sentiment tags to each category.
2. **Strict Output Contracts**: The application utilizes Pydantic validation to enforce a structured JSON schema. This guarantees that all model responses conform to a predictable data model suitable for automated pipeline ingestion.
3. **Entity and Urgency Classification**: The system identifies key entities (e.g., specific products) and computes an urgency rating (Low to Critical) based on the severity and category of negative feedback.

### The Pipeline

```
Customer Text
   │
   ▼
┌──────────────────┐
│ Prompt Builder   │ ← Injects Structured Output Schema (JSON Contract)
│ (prompts.py)     │
└───────┬──────────┘
        │
        ▼
┌──────────────────┐     ┌──────────────────┐
│ Gemini API       │────▶│ Pydantic Parser  │
│ (structured      │     │ (models.py)      │
│  JSON output)    │     │ Type validation  │
└──────────────────┘     └───────┬──────────┘
                                   │
                       ┌─────────────┼──────────────┐
                       ▼             ▼              ▼
                 ┌──────────┐ ┌──────────┐   ┌──────────-┐
                 │ Sentiment│ │ Aspects  │   │ Urgency   │
                 │ + Score  │ │ Breakdown│   │ + Entities│
                 └──────────┘ └──────────┘   └──────────-┘
```

## Example Input and Output

### System Input (Unstructured Customer Review)
```
"The dress design is gorgeous and fits perfectly, but the zipper broke after wearing it just once. Customer support was helpful but I'm still frustrated."
```

### System Output (Structured Sentiment Payload)
```json
{
  "overall_sentiment": "MIXED",
  "confidence_score": 0.95,
  "aspects": [
    {"aspect": "design", "sentiment": "POSITIVE", "supporting_text": "design is gorgeous and fits perfectly"},
    {"aspect": "durability", "sentiment": "NEGATIVE", "supporting_text": "zipper broke after wearing it just once"},
    {"aspect": "customer_support", "sentiment": "POSITIVE", "supporting_text": "Customer support was helpful"}
  ],
  "entities": [{"name": "dress", "type": "PRODUCT"}],
  "urgency": "MEDIUM"
}
```

### Demo Mode Fallback

To ensure the service can run immediately without requiring external API keys, it defaults to `DEMO_MODE=true`. This uses a robust **heuristic analyzer**:
- Keyword‑based sentiment detection with weighted positive/negative vocabularies.
- Pattern‑matched aspect extraction (detects "fit/sizing", "fabric quality", "durability").
- Deterministic urgency scoring based on negative signal density.

## Technical Decisions & Trade‑offs

### Analysis Methodology

| Approach                     | Advantages                                    | Disadvantages                                 |
|------------------------------|-----------------------------------------------|-----------------------------------------------|
| **Few‑Shot Classification** | Very low latency, simple pattern matching      | Limited to single overall labels; cannot extract multiple aspects |
| **Fine‑Tuned BERT Models**   | Fast inference speeds, zero API dependency    | Requires extensive training datasets and lacks zero‑shot adaptation |
| **Local PyTorch EmbeddingBag** ✅ | Sub-5ms CPU inference, zero host API costs, easily run locally. | Requires historical labels for training; lacks deep contextual nuance. |
| **Gemini Structured Output** ✅ | Deep, multi‑signal semantic analysis; zero‑shot capability | Higher relative latency and dependency on external API services |

### Schema Enforcement with Pydantic

To prevent parsing errors downstream, all LLM outputs are validated against a strict Pydantic model contract at the service boundary:
```python
class SentimentResult(BaseModel):
    sentiment: SentimentLabel          # Enum: POSITIVE | NEGATIVE | NEUTRAL | MIXD
    confidence: float = Field(ge=0.0, le=1.0)
    aspects: list[AspectSentiment]     # Categorized aspect‑level breakdown
    iab_aspects: list[IABAspectSentiment]  # IAB‑Tech flat aspect format
    entities: list[Entity]             # Named entities extracted
    urgency: UrgencyLevel              # Enum: LOW → CRITICAL
```

## IAB‑Tech Output Format

The IAB‑Tech schema expects a flat list of aspect objects where each entry contains `aspect`, `sentiment`, `score`, and an explicit `confidence` field (the same numeric value as `score`). This aligns with industry‑standard ad‑tech pipelines and makes downstream aggregation trivial.

Our mock service now populates the `iab_aspects` field by copying the richer `aspects` data, ensuring both the internal rich model and the IAB‑Tech contract are satisfied.

If the generated output deviates from the schema, validation catch blocks capture the failure, permitting retry strategies or safe fallback defaults to prevent system failures.

## Architecture

```
 demo_sentiment/
 ├── models.py      # Pydantic request/response schemas (The Contract)
 ├── prompts.py     # Gemini prompt templates with JSON schema injection
 ├── services.py    # Business logic: live analysis + mock fallback (now supports Gemini & Ollama)
 ├── backend.py     # FastAPI app with /analyze, /analyze/batch, /history
 └── frontend.py    # Streamlit dashboard with Plotly charts
```

## Running Locally

```bash
# Terminal 1 — API Backend
make demo-1-api          # Runs on http://localhost:8001

# Terminal 2 — UI Frontend
make demo-1-ui           # Runs on http://localhost:8501
```

## Cost & Efficiency Options

| Backend   | Model (default)                | Typical Cost per 1 k tokens* | Latency (ms) | Remarks |
|-----------|--------------------------------|------------------------------|--------------|---------|
| **Gemini** | `gemini-2.5-flash-lite` (via Google) | ~$0.00035 (USD) | ~400‑600 | High‑quality zero‑shot output; cheap tier of Gemini. |
| **Ollama**| `mistral:7b-instruct-q4_K_M` (local) | $0 (local compute)           | ~200‑300     | No API costs; requires local Docker/CPU/GPU. Ideal for dev & low‑budget prod. |
| **GPT‑OSS** | Open‑source Llama 2‑7B‑Chat (via vLLM) | $0 (compute) | ~200‑400 | Free model; run on modest CPU/GPU instance. |

> \*Costs are approximate based on public pricing at the time of writing and assume 1 k input + output tokens.

### Estimated Monthly Cost for 1 Million Analyses

Assumptions:
- Average request uses ~700 tokens (≈0.7 k).  
- No discount tiers; pricing is linear.

| Backend | Cost per request (≈0.7 k tokens) | Cost for 1 M requests |
|--------|--------------------------------|-----------------------|
| Gemini | $0.00035 × 0.7 ≈ $0.000245 | **$245** |
| Ollama | $0 (compute only) – estimate $0.05 /hr for a modest CPU instance. Running 24/7 for a month ≈ $36. Assuming the model can handle the load, **≈ $40** total (infrastructure) |
| GPT‑OSS | $0 (compute) – estimate $0.05 /hr for a modest CPU instance. Running 24/7 for a month ≈ $36. Assuming the model can handle the load, **≈ $40** total (infrastructure) |

#### Optimisation Strategies to Reduce Cost

- **Batching**: Group up to `MAX_BATCH_SIZE` (default 8) requests into a single API call. For Gemini/BISON this reduces token overhead because the prompt is shared, saving ~10‑15 % on token‑based pricing.
- **OpenAI Batch API**: Submit an array of requests as a JSONL file; the batch endpoint processes jobs asynchronously (often 10‑15 min) and returns results at **½ the standard per‑token price**. Ideal for large offline analyses.
- **Google Gemini Batch API**: Use the asynchronous batch endpoint to submit bulk jobs; Gemini processes them in parallel and charges the same per‑token rate but with reduced overhead and higher throughput.
- **Prompt Caching**: Re‑use the same prompt template for identical inputs; cache the generated JSON schema to avoid re‑sending it each call.
- **Quantised Local Model**: Deploy a quantised Ollama model (8‑bit) – cuts CPU cycles by ~30 % and reduces required instance size, lowering infrastructure cost.
- **ONNX Runtime**: Convert a transformer (e.g., DistilBERT) to ONNX and run with the ONNX Runtime for up to 2‑3× faster inference, allowing fewer CPU cores.
- **Async Parallelism**: Use `asyncio.gather` with a semaphore to saturate the backend without exceeding rate limits, improving throughput and allowing lower‑cost provisioned resources.
- **Cost‑Aware Routing**: Route low‑priority bulk jobs to the free Ollama backend, reserving Gemini for high‑value, SLA‑critical analyses.

### Configuration

The `shared/config.py` file drives backend selection:
- `MODEL_BACKEND` – choose `GEMINI`, `OLLAMA`, or `BISON`.
- `OLLAMA_MODEL` – specify the local model name (e.g., `mistral:7b-instruct-q4_K_M`).
- `MAX_BATCH_SIZE` – maximum number of items processed per batch request (default 8).

Set these via environment variables or a `.env` file at the repository root.

## Future Considerations: Quantization & ONNX

- **Quantization**: Applying 8‑bit or 4‑bit quantization to a local model (e.g., via `bitsandbytes` or `torch.quantization`) can cut memory usage by >50 % and improve inference speed on CPUs.
- **ONNX Runtime**: Exporting a transformer model to ONNX and running it with the ONNX Runtime enables accelerated inference on both CPU and GPU, often surpassing pure PyTorch speed.
- **Batch Optimisation**: The current `/analyze/batch` endpoint processes items sequentially. In production we will switch to `asyncio.gather` with a semaphore, enabling true concurrent calls to the chosen backend while respecting rate limits.
- **Dynamic Backend Routing**: Route low‑priority bulk jobs to our quantized local PyTorch model, reserving the premium Gemini backend for complex or SLA‑critical requests.

## Candidate Small Sentiment Models (Hugging Face)

| Model | Size | Typical Latency (CPU) | Accuracy (SST‑2) | Why Consider |
|-------|------|-----------------------|------------------|--------------|
| `distilbert-base-uncased-finetuned-sst-2-english` | 66 M params | ~30 ms | 91 % | Industry‑standard small model; well‑supported and easy to deploy. |
| `Varnikasiva/sentiment-classification-bert-mini` | 11 M params | ~10 ms | 85 % | Ultra‑lightweight; great for edge or high‑throughput scenarios. |
| `tanaos/tanaos-sentiment-analysis-v1` | ~5 M params | <5 ms | 78 % | Extreme speed; suitable for massive batch jobs where perfect accuracy is not critical. |
| `cardiffnlp/twitter-roberta-base-sentiment` | 125 M params | ~45 ms | 93 % | Optimised for short social‑media text; handles emojis and slang well. |

**How to use** (example with DistilBERT):
```python
from transformers import pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
print(classifier("I love using fast models!"))
```

These models can be loaded locally and wrapped in a lightweight client that mimics the `generate_json` interface used by `SentimentService`, enabling seamless swapping between LLM‑based and classic transformer backends.

## Production Evolution Path

1. **Native Structured Output Mode**: Use Gemini's native `response_mime_type="application/json"` feature to enforce schema at the model layer, reducing output parsing overhead.
2. **Async Batch Processing**: The current batch endpoint processes sequentially. Production would use `asyncio.gather()` with a semaphore and exponential backoff to handle rate limits during parallel Gemini/Ollama calls.
3. **Persistent Storage Pipeline**: Route the validated Pydantic models directly into BigQuery or Snowflake for long‑term historical trend analysis.
4. **Streaming Responses**: For very long texts or transcripts, stream partial results to the UI via WebSockets to improve perceived latency.
5. **Cost‑aware Routing**: Dynamically select the backend per request based on token budget or SLA (e.g., cheap local Ollama for high‑volume low‑priority jobs, Gemini for premium analysis).


---

## Iteration 1 - PyTorch Sentiment & Aspect Classifier

### Hybrid Routing & Architectural Trade-offs

Transitioning from expensive/slow LLM prototypes to cost-effective, high-throughput hybrid systems.

> *"The **Sentiment Intelligence Engine** provides deep, aspect-level extraction, but running LLMs for millions of reviews is cost-prohibitive. To solve this, I integrated a lightweight, quantized **PyTorch classifier** (`nn.EmbeddingBag`). It shows how we can handle 90% of simple/high-volume traffic locally on standard CPUs at sub-5ms latency and zero API cost, routing only low-confidence or complex cases to Gemini."*

#### Integration Use Cases

1. **A Fast, Local Aspect Detector (First-Stage Filter)**
   * **Approach**: Train the local `EmbeddingBag` model as a multi-label aspect classifier to predict which aspects (e.g., `design`, `durability`, `customer_support`) are active.
   * **Benefit**: If no negative or critical aspects are active, skip sending the request to Gemini entirely, or only instruct Gemini to analyze the specific aspects identified by the local model, saving significant prompt tokens.
2. **A High-Throughput Sentiment Router**
   * **Approach**: Train the local model on historical sentiment outputs (`positive`, `negative`, `mixed`).
   * **Benefit**: If the local model is $>95\%$ confident that a review is simple and purely positive (e.g., *"Loved the color!"*), process it locally on the CPU for free. Route only mixed or low-confidence reviews to Gemini.

#### Comparative Trade-offs

| Dimension | Gemini 2.5 Flash Lite | PyTorch `EmbeddingBag` (Document Classifier) |
| :--- | :--- | :--- |
| **Cost** | ~$0.25 per 1,000 runs (variable API cost) | **$0 (runs locally on standard CPU)** |
| **Latency** | ~400ms – 600ms | **< 5ms** (especially when quantized to `int8`) |
| **Data Requirements** | **Zero-shot** (works out of the box with prompt) | Requires **labeled training data** (e.g., 500+ tagged reviews) |
| **Contextual Nuance** | Excellent. Handles negation (*"not bad"*), sarcasm, and extracts exact supporting quotes. | Basic. `EmbeddingBag` averages word embeddings, so it struggles with negation and cannot extract supporting text snippets. |

#### Implementation

Added a local sentiment and aspect classifier built with PyTorch's `nn.EmbeddingBag` architecture. It is optimized for high-throughput, CPU-bound routing.

*   **Inference Integration**: [pytorch_client.py](pytorch_client.py)
*   **Model Architecture**: [model.py](pytorch_model/model.py)
*   **Interactive Demo Notebook**: [sentiment_training_demo.ipynb](pytorch_model/sentiment_training_demo.ipynb)

To run the training pipeline locally:
```bash
python -m pytorch_model.model --output-dir ./pytorch_model/output
```
This script trains a float model, evaluates micro-F1 accuracy, applies **dynamic quantization (INT8)** to reduce size by >50%, and outputs the quantized weights.

---
