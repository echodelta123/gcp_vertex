"""
Streamlit frontend for the Sentiment Intelligence Engine.

Features:
  - Single text analysis with rich results display
  - Batch analysis via CSV upload with aggregate dashboard
  - Interactive sentiment gauge, aspect charts, and trend visualization
  - Sample data for instant demo without setup

Run: streamlit run demo_sentiment_categoriser/frontend.py --server.port 8501
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

from shared.gemini_client import GeminiClient
from shared.config import settings
from shared.mock_data import generate_reviews
from demo_sentiment_categoriser.services import SentimentService
from demo_sentiment_categoriser.models import SentimentLabel

# --- Page Config ---
st.set_page_config(
    page_title="Sentiment Intelligence Engine | Vertex AI Demo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .sentiment-positive { color: #00e676; }
    .sentiment-negative { color: #ff5252; }
    .sentiment-neutral { color: #ffd740; }
    .sentiment-mixed { color: #448aff; }
    .aspect-chip {
        display: inline-block; padding: 6px 14px; margin: 4px;
        border-radius: 20px; font-size: 0.85rem; font-weight: 500;
    }
    .chip-positive { background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid rgba(0,230,118,0.3); }
    .chip-negative { background: rgba(255,82,82,0.15); color: #ff5252; border: 1px solid rgba(255,82,82,0.3); }
    .chip-neutral { background: rgba(255,215,64,0.15); color: #ffd740; border: 1px solid rgba(255,215,64,0.3); }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# --- Initialize Service ---
@st.cache_resource
def get_service():
    client = GeminiClient()
    return SentimentService(client)

service = get_service()

SENTIMENT_COLORS = {
    "POSITIVE": "#00e676", "NEGATIVE": "#ff5252",
    "NEUTRAL": "#ffd740", "MIXED": "#448aff",
}


# --- Helper: Sentiment Gauge ---
def render_sentiment_gauge(sentiment: str, confidence: float):
    color = SENTIMENT_COLORS.get(sentiment, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        title={"text": f"<b>{sentiment}</b>", "font": {"size": 18, "color": color}},
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#444"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1a2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "rgba(255,82,82,0.1)"},
                {"range": [33, 66], "color": "rgba(255,215,64,0.1)"},
                {"range": [66, 100], "color": "rgba(0,230,118,0.1)"},
            ],
        },
    ))
    fig.update_layout(
        height=250, margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ccc"},
    )
    return fig


def render_aspect_chart(aspects):
    if not aspects:
        return None
    df = pd.DataFrame([
        {"Aspect": a.aspect.title(), "Score": a.score, "Sentiment": a.sentiment.value}
        for a in aspects
    ])
    colors = [SENTIMENT_COLORS.get(s, "#888") for s in df["Sentiment"]]
    fig = px.bar(
        df, x="Score", y="Aspect", orientation="h",
        color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
        title="Aspect-Level Sentiment Breakdown",
    )
    fig.update_layout(
        height=max(200, len(aspects) * 50 + 80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#ccc"}, xaxis={"range": [0, 1], "gridcolor": "#333"},
        yaxis={"gridcolor": "#333"}, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Check health / availability
    gemini_live = service.gemini_client.is_live
    pytorch_ready = service.pytorch_client.is_ready
    
    # Options mapping to keys expected by the backend
    backend_options = {
        "🟢 Live (Gemini)": "GEMINI",
        "🦙 Local LLM (Ollama)": "OLLAMA",
        "⚡ Local PyTorch (EmbeddingBag)": "LOCAL_PYTORCH",
        "🔶 Heuristic (Demo Mode)": "HEURISTIC"
    }
    
    selected_label = st.selectbox(
        "Active Inference Backend:",
        options=list(backend_options.keys()),
        index=0 if gemini_live else (2 if pytorch_ready else 3)
    )
    selected_backend = backend_options[selected_label]

    # Show availability status
    st.markdown("### Backend Status")
    st.markdown(f"- **Gemini API:** {'🟢 Live' if gemini_live else '🔴 Offline / Demo'}")
    st.markdown(f"- **Local PyTorch:** {'🟢 Ready' if pytorch_ready else '🔴 Not Trained'}")
    st.markdown("- **Ollama Service:** 🔘 Local API assumed")

    # Estimated Cost & Latency table
    st.markdown("---")
    st.markdown("### 📊 Backend Reference Specs")
    specs_data = [
        {"Backend": "Gemini 2.5", "Latency": "~500ms", "Cost / 1K": "$0.00025"},
        {"Backend": "Ollama 7B", "Latency": "~300ms", "Cost / 1K": "Free"},
        {"Backend": "PyTorch CPU", "Latency": "< 5ms", "Cost / 1K": "Free"},
        {"Backend": "Heuristics", "Latency": "< 1ms", "Cost / 1K": "Free"}
    ]
    st.dataframe(pd.DataFrame(specs_data), hide_index=True)

    st.markdown("---")
    st.markdown("## 📊 Sample Data")
    if st.button("Load Sample Reviews", use_container_width=True):
        st.session_state["sample_loaded"] = True

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Sentiment Intelligence Engine**  
    Powered by Google Vertex AI (Gemini)
    
    Analyzes customer feedback with:
    - Aspect-based sentiment
    - Entity extraction
    - Urgency scoring
    - Batch processing
    
    """)


# --- Main Content ---
st.markdown("# 💬 Sentiment Intelligence Engine")
st.markdown("*Enterprise-grade customer feedback analysis powered by Google Vertex AI*")

# Portfolio Narrative Box
with st.expander("📚 **Portfolio Narrative: Quantization & Dynamic Routing**", expanded=False):
    st.markdown("""
    ### Transitioning from LLMs to High-Throughput CPU Models
    
    The **Sentiment Intelligence Engine** provides deep aspect-level sentiment extraction using LLMs (Gemini). However, processing millions of reviews per day at scale via LLMs is often cost-prohibitive.
    
    To solve this, we integrated a lightweight **Document Classifier** (`nn.EmbeddingBag` model built with PyTorch) trained on historical LLM labels. 
    
    #### ⚖️ The Architectural Trade-offs:
    * **Latency:** Gemini takes **400ms – 600ms**, while the dynamic quantized PyTorch model runs on the local CPU in **< 5ms**.
    * **Cost:** Gemini API usage has a variable dollar cost, while the PyTorch CPU model is **100% free ($0)** to run on existing servers.
    * **Contextual Nuance:** Gemini handles negation, sarcasm, and extracts exact quotes. PyTorch averages word embeddings, making it excellent for classification but missing detailed text extraction.
    
    #### 🔀 Smart Dynamic Routing Pattern:
    In production, a router sends simple, confident text to the fast local PyTorch model, and falls back to Gemini only for low-confidence classifications or highly complex feedback.
    """)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📊 Batch Analysis", "📈 Dashboard"])

# --- Tab 1: Single Analysis ---
with tab1:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("### Analyze Text")
        text_input = st.text_area(
            "Paste a customer review, support ticket, or feedback:",
            height=200,
            placeholder="e.g., 'The dress design is gorgeous and fits perfectly, but the zipper broke after wearing it just once. Customer support was helpful but I'm still frustrated.'",
        )
        context = st.selectbox(
            "Context (optional):", ["auto-detect", "product review", "support ticket", "social media", "survey response"]
        )
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    with col_result:
        if analyze_btn and text_input:
            import time
            start_time = time.time()
            with st.spinner("Analyzing sentiment..."):
                import asyncio
                ctx = None if context == "auto-detect" else context
                result = asyncio.run(service.analyze(text_input, ctx, backend=selected_backend))
            elapsed_ms = (time.time() - start_time) * 1000

            # Performance stats
            col_perf1, col_perf2 = st.columns(2)
            col_perf1.metric("Response Time", f"{elapsed_ms:.1f} ms")
            cost_map = {
                "GEMINI": "$0.00025",
                "OLLAMA": "Free (Local)",
                "LOCAL_PYTORCH": "Free (Local CPU)",
                "HEURISTIC": "Free (Local Heuristic)"
            }
            col_perf2.metric("Estimated Cost", cost_map.get(selected_backend, "Free"))

            # Gauge
            st.plotly_chart(render_sentiment_gauge(result.sentiment.value, result.confidence), use_container_width=True)

            # Urgency badge
            urgency_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
            st.markdown(f"**Urgency:** {urgency_colors.get(result.urgency.value, '⚪')} {result.urgency.value}")

            # Summary
            if result.summary:
                st.info(f"**Summary:** {result.summary}")

            # Aspects
            if result.aspects:
                fig = render_aspect_chart(result.aspects)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # Key Phrases
            if result.key_phrases:
                st.markdown("**Key Phrases:**")
                chips = ""
                for phrase in result.key_phrases:
                    chips += f'<span class="aspect-chip chip-neutral">{phrase}</span>'
                st.markdown(chips, unsafe_allow_html=True)

            # Entities
            if result.entities:
                st.markdown("**Entities Detected:**")
                for ent in result.entities:
                    st.markdown(f"- **{ent.name}** ({ent.type})")

        elif analyze_btn:
            st.warning("Please enter text to analyze.")

# --- Tab 2: Batch Analysis ---
with tab2:
    st.markdown("### Batch Analysis")
    st.markdown("Upload a CSV with a `text` column, or use sample data.")

    use_sample = st.session_state.get("sample_loaded", False)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file or use_sample:
        if use_sample:
            reviews = generate_reviews(15)
            df = pd.DataFrame(reviews)
            text_col = "body"
        else:
            df = pd.read_csv(uploaded_file)
            text_cols = [c for c in df.columns if "text" in c.lower() or "body" in c.lower() or "review" in c.lower() or "content" in c.lower()]
            text_col = text_cols[0] if text_cols else df.columns[0]

        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Analysis", type="primary"):
            import asyncio
            import time
            batch_start_time = time.time()
            progress = st.progress(0)
            results = []
            texts = df[text_col].dropna().tolist()[:50]  # Cap at 50

            for i, text in enumerate(texts):
                result = asyncio.run(service.analyze(str(text), "product review", backend=selected_backend))
                results.append(result)
                progress.progress((i + 1) / len(texts))

            batch_elapsed = time.time() - batch_start_time
            st.success(f"✅ Analyzed {len(results)} items in {batch_elapsed:.2f} seconds ({ (batch_elapsed/len(results))*1000 :.1f} ms/item)")

            # Aggregate stats
            stats = service.get_aggregate_stats(results)

            col1, col2, col3, col4 = st.columns(4)
            dist = stats.get("sentiment_distribution", {})
            col1.metric("Positive", dist.get("POSITIVE", 0))
            col2.metric("Negative", dist.get("NEGATIVE", 0))
            col3.metric("Neutral", dist.get("NEUTRAL", 0))
            col4.metric("Mixed", dist.get("MIXED", 0))

            # Distribution pie chart
            fig_pie = px.pie(
                names=list(dist.keys()), values=list(dist.values()),
                title="Sentiment Distribution",
                color=list(dist.keys()),
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ccc"})
            st.plotly_chart(fig_pie, use_container_width=True)

            # Aspect summary
            aspect_summary = stats.get("aspect_summary", {})
            if aspect_summary:
                st.markdown("### Aspect Summary Across All Reviews")
                asp_df = pd.DataFrame([
                    {"Aspect": k.title(), "Avg Score": v["avg_score"],
                     "Dominant Sentiment": v["dominant_sentiment"], "Mentions": v["mentions"]}
                    for k, v in aspect_summary.items()
                ]).sort_values("Mentions", ascending=False)
                st.dataframe(asp_df, use_container_width=True, hide_index=True)

            # Results table
            st.markdown("### Detailed Results")
            results_df = pd.DataFrame([
                {"Text": r.source_text[:100] + "...", "Sentiment": r.sentiment.value,
                 "Confidence": f"{r.confidence:.0%}", "Urgency": r.urgency.value,
                 "Aspects": ", ".join(a.aspect for a in r.aspects)}
                for r in results
            ])
            st.dataframe(results_df, use_container_width=True, hide_index=True)

# --- Tab 3: Dashboard ---
with tab3:
    st.markdown("### Analysis Dashboard")
    history = service.get_history()
    if not history:
        st.info("No analysis history yet. Run some analyses to see trends here.")
    else:
        st.metric("Total Analyses", len(history))

        sentiments = [r.sentiment.value for r in history]
        sent_counts = {s: sentiments.count(s) for s in set(sentiments)}

        fig = px.bar(
            x=list(sent_counts.keys()), y=list(sent_counts.values()),
            color=list(sent_counts.keys()), color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment Distribution (Session History)",
            labels={"x": "Sentiment", "y": "Count"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ccc"}, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")