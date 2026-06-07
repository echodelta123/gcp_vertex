"""
Streamlit frontend for the Semantic Recommendation Engine.

Features:
  - Natural language product search
  - Visual product cards with similarity scores
  - AI-generated recommendation explanations
  - Product catalog browser
  - Ingestion status panel
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import asyncio
import torch

from demo_recommendation_engine.pytorch_model.monitoring.drift_detector import DriftReport
from demo_recommendation_engine.pytorch_model.retraining.trigger import evaluate_retraining_need
from shared.gemini_client import GeminiClient
from shared.config import settings
from demo_recommendation_engine.services import RecommenderService

# --- Page Config ---
st.set_page_config(
    page_title="Recommendation Engine | Vertex AI Demo",
    page_icon="🛍️",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .product-card {
        background: linear-gradient(135deg, #1e1e30 0%, #2a2a40 100%);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
        padding: 24px; margin-bottom: 16px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .product-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
    .product-name { font-size: 1.2rem; font-weight: 700; color: #e0e0e0; margin-bottom: 8px; }
    .product-category { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .product-price { font-size: 1.4rem; font-weight: 700; color: #00e676; }
    .match-score { font-size: 0.9rem; color: #448aff; font-weight: 600; }
    .reason-tag {
        display: inline-block; padding: 4px 12px; margin: 3px;
        border-radius: 16px; font-size: 0.75rem;
        background: rgba(68,138,255,0.15); color: #448aff;
        border: 1px solid rgba(68,138,255,0.3);
    }
    .explanation-box {
        background: rgba(255,255,255,0.03); border-left: 3px solid #448aff;
        padding: 12px 16px; margin-top: 12px; border-radius: 0 8px 8px 0;
        font-size: 0.9rem; color: #bbb;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    gemini = GeminiClient()
    svc = RecommenderService(gemini)
    svc.ingest_catalog()
    return svc

service = get_service()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    mode = "🟢 Live" if service.gemini.is_live else "🔶 Demo"
    st.info(f"**Mode:** {mode}")
    st.metric("Catalog Size", service.get_catalog_size())

    st.markdown("---")
    st.markdown("### 💡 Try These Queries")
    sample_queries = [
        "summer dresses for a garden wedding",
        "smart casual wide-leg trousers for the office",
        "cozy winter knitwear layering pieces",
        "edgy leather jacket for going out",
        "vintage 90s style denim",
    ]
    for q in sample_queries:
        if st.button(f"→ {q}", key=f"sq_{q}"):
            st.session_state["query"] = q

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Semantic Recommendation Engine**  
    **Semantic Recommendation Engine**  
    Uses vector embeddings (LanceDB) for semantic product search
    and Gemini for personalized explanations.
    
    Inspired by [Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview).
    """)

# --- Main ---
st.markdown("# 🛍️ Semantic Recommendation Engine")
st.markdown("*Personalized product discovery powered by vector search & Google Gemini*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Discover Products", "📦 Product Catalog", "⚡ Two-Tower Recommender (PyTorch)"])

with tab1:
    query = st.text_input(
        "What are you looking for?",
        value=st.session_state.get("query", ""),
        placeholder="e.g., 'I need a floral summer dress for a beach wedding'",
    )
    col_k, col_btn = st.columns([1, 3])
    with col_k:
        top_k = st.slider("Results", 3, 10, 5)
    with col_btn:
        search_btn = st.button("🔍 Find Recommendations", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner("Searching catalog and generating explanations..."):
            results = asyncio.run(service.recommend(query, top_k))

        if results:
            st.markdown(f"### Top {len(results)} Recommendations for: *\"{query}\"*")

            # Similarity scores chart
            fig = go.Figure(go.Bar(
                x=[r.product.name for r in results],
                y=[r.product.similarity_score or 0.5 for r in results],
                marker_color=["#448aff" if i == 0 else "#5c6bc0" for i in range(len(results))],
                text=[f"{(r.product.similarity_score or 0.5):.0%}" for r in results],
                textposition="outside",
            ))
            fig.update_layout(
                title="Semantic Match Scores", yaxis_title="Similarity",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ccc"}, yaxis={"range": [0, 1.1], "gridcolor": "#333"},
                xaxis={"gridcolor": "#333"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Product cards
            for i, rec in enumerate(results):
                p = rec.product
                score_pct = f"{(p.similarity_score or 0.5):.0%}"
                reasons_html = "".join(f'<span class="reason-tag">{r}</span>' for r in rec.match_reasons)

                st.markdown(f"""
                <div class="product-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <div class="product-category">{p.category}</div>
                            <div class="product-name">{p.name}</div>
                        </div>
                        <div style="text-align:right;">
                            <div class="product-price">${p.price:.2f}</div>
                            <div class="match-score">Match: {score_pct}</div>
                        </div>
                    </div>
                    <p style="color:#999;margin:8px 0;font-size:0.9rem;">{p.description}</p>
                    <div>{reasons_html}</div>
                    <div class="explanation-box">💡 {rec.explanation}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Try a different query.")

with tab2:
    st.markdown("### Product Catalog")
    catalog = service.get_catalog()
    if catalog:
        df = pd.DataFrame(catalog)
        display_cols = ["product_id", "name", "category", "price", "tags"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # Category distribution
        fig = px.pie(df, names="category", title="Products by Category",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#ccc"})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## ⚡ Two-Tower Recommender (PyTorch)")
    st.markdown(
        "*High-throughput dual-encoder retrieval running locally on CPU in sub-5ms.*"
    )

    # Narrative overview
    with st.expander("📚 **Architecture Overview: Dual-Encoder (Two-Tower) Model**", expanded=False):
        st.markdown("""
        ### Two-Tower Architecture & High-Scale MLOps
        
        The **Two-Tower** model consists of:
        1. **User Tower**: Deep neural network mapping user IDs and features into a shared embedding space.
        2. **Item Tower**: Deep neural network mapping item IDs and features into the same shared embedding space.
        
        #### ⚡ Performance & Scale:
        - **Retrieval:** Matching is computed using a simple dot product (or cosine similarity) between user and item embeddings. At scale, this is accelerated via Approximate Nearest Neighbor (ANN) indexers (e.g. Vertex AI Vector Search).
        - **Latency:** Unlike LLMs (~500ms) or heavy ranking models, the Two-Tower retrieval stage runs on CPU in **< 1ms**, ideal for serving millions of users.
        - **In-Batch Negatives:** Trained using custom softmax loss where other items in the batch act as implicit negative examples, avoiding expensive negative sampling loops.
        """)

    st.markdown("---")

    # Layout: left for training & drift, right for interactive demo
    col_demo, col_mlops = st.columns([5, 4])

    with col_demo:
        st.markdown("### 🔍 Test Recommender Model")
        
        user_id = st.number_input("Select User ID (0 - 499):", min_value=0, max_value=499, value=42)
        
        # User features interactive selection
        pref_cat = st.selectbox(
            "User Favorite Category Preference:",
            ["Dresses", "Trousers", "Knitwear", "Outerwear", "Denim", "Tops", "Skirts"]
        )
        
        # Construct synthetic user features
        categories_list = ["Dresses", "Trousers", "Knitwear", "Outerwear", "Denim", "Tops", "Skirts"]
        # 8-dimensional user features: 7 dims correspond to categories, 8th is a general bias
        user_features = [0.0] * 8
        if pref_cat in categories_list:
            user_features[categories_list.index(pref_cat)] = 2.0
        user_features[7] = 0.5  # general bias
        
        top_k_rec = st.slider("Number of Recommendations:", 3, 10, 5, key="top_k_rec")
        
        rec_btn = st.button("🚀 Get PyTorch Recommendations", use_container_width=True)
        
        if rec_btn:
            if not service.pytorch_client.is_ready:
                st.error("❌ Two-Tower model is not trained. Please train the model in the panel on the right first!")
            else:
                with st.spinner("Retrieving from PyTorch model..."):
                    import time
                    start_time = time.time()
                    resp = service.get_pytorch_recommendations(user_id, user_features, top_k_rec)
                    elapsed = (time.time() - start_time) * 1000
                
                if "error" in resp:
                    st.error(f"❌ {resp['error']}")
                else:
                    st.success(f"✅ Retrieved top recommendations in **{elapsed:.2f} ms**")
                    
                    st.markdown("#### Top Recommended Products")
                    catalog = service.get_catalog()
                    
                    for i, item_id in enumerate(resp["recommended_items"]):
                        score = resp["scores"][i]
                        # Deterministic mapping to real catalog items
                        p_dict = catalog[item_id % len(catalog)]
                        
                        st.markdown(f"""
                        <div class="product-card">
                            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                                <div>
                                    <div class="product-category">{p_dict['category']}</div>
                                    <div class="product-name">[ID: {item_id}] {p_dict['name']}</div>
                                </div>
                                <div style="text-align:right;">
                                    <div class="product-price">${p_dict['price']:.2f}</div>
                                    <div class="match-score">Similarity: {score:.3f}</div>
                                </div>
                            </div>
                            <p style="color:#999;margin:8px 0;font-size:0.9rem;">{p_dict['description']}</p>
                            <div>{"".join(f'<span class="reason-tag">{t}</span>' for t in p_dict['tags'])}</div>
                        </div>
                        """, unsafe_allow_html=True)

    with col_mlops:
        st.markdown("### ⚙️ Model Management")
        
        # Training panel
        with st.container():
            is_ready = service.pytorch_client.is_ready
            status_text = "🟢 Trained & Ready" if is_ready else "🔴 Untrained / Missing weights"
            st.markdown(f"**Model Status:** {status_text}")
            
            epochs = st.slider("Training Epochs:", 3, 20, 5)
            train_btn = st.button("🏋️ Train Two-Tower Model", use_container_width=True)
            
            if train_btn:
                with st.spinner("Training model on CPU (synthetic data)..."):
                    res = service.train_pytorch_model(epochs=epochs)
                if res["success"]:
                    st.success("🎉 Model trained successfully!")
                    metrics = res["metrics"]
                    
                    # Display training metrics
                    hist = metrics["test_metrics"]
                    st.markdown("#### Test Set Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Hit Rate@10", f"{hist['hit_rate@10']:.2%}")
                    col2.metric("NDCG@10", f"{hist['ndcg@10']:.2%}")
                    col3.metric("MRR", f"{hist['mrr']:.2f}")
                    
                    # Display Quantization metrics
                    st.markdown("#### ⚡ Dynamic Quantization Benefits")
                    st.markdown(f"- **Original Size:** {metrics['model_size_bytes']/1024:.1f} KB")
                    st.markdown(f"- **Quantized Size:** {metrics['quantized_size_bytes']/1024:.1f} KB")
                    st.markdown(f"- **Storage Reduction:** **{metrics['size_reduction_pct']}%** smaller")
                    
                    # Plot training history if available
                    model_path = service.pytorch_client.output_dir / "model.pt"
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    history_data = checkpoint.get("training_history", {})
                    
                    if history_data:
                        epochs_list = list(range(1, len(history_data["train_loss"]) + 1))
                        # Plot Loss
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(x=epochs_list, y=history_data["train_loss"], name="Train Loss", line=dict(color="#ff5252", width=2)))
                        fig_loss.update_layout(
                            title="Training Loss Curve", xaxis_title="Epoch", yaxis_title="Loss",
                            height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#ccc"}, yaxis={"gridcolor": "#333"}, xaxis={"gridcolor": "#333", "tickmode": "linear"}
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                        
                        # Plot Metrics
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(x=epochs_list, y=history_data["val_hit@10"], name="Hit Rate@10", line=dict(color="#448aff", width=2)))
                        fig_acc.add_trace(go.Scatter(x=epochs_list, y=history_data["val_ndcg@10"], name="NDCG@10", line=dict(color="#00e676", width=2)))
                        fig_acc.update_layout(
                            title="Validation Accuracy Trend", xaxis_title="Epoch", yaxis_title="Score",
                            height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#ccc"}, yaxis={"gridcolor": "#333"}, xaxis={"gridcolor": "#333", "tickmode": "linear"}
                        )
                        st.plotly_chart(fig_acc, use_container_width=True)
                        
                else:
                    st.error(f"❌ Training failed: {res['error']}")
        
        st.markdown("---")
        st.markdown("### 📈 MLOps Drift & Retraining Simulator")
        st.markdown("*Adjust sliders to simulate shifts in production user traffic or behavior.*")
        
        # Retraining simulator inputs
        feature_drift = st.slider("Feature Drift (PSI):", 0.0, 0.5, 0.08, 0.01)
        pred_drift = st.slider("Prediction Drift (KS):", 0.0, 0.20, 0.02, 0.01)
        ctr_decay = st.slider("CTR Metric Decay (Relative %):", 0.0, 0.50, 0.02, 0.01)
        days_stale = st.slider("Days Since Last Train:", 1, 30, 5, 1)
        new_data_count = st.slider("New User Interactions:", 0, 20000, 12000, 500)
        
        # Instantiate DriftReport structure
        report = DriftReport(
            feature_drift_score=feature_drift,
            prediction_drift_score=pred_drift,
            business_metric_current=1.0 - ctr_decay,
            business_metric_baseline=1.0,
            needs_retraining=False,
            reasons=[],
            timestamp=""
        )
        reasons = []
        if feature_drift > 0.2:
            reasons.append(f"Feature PSI={feature_drift:.3f} exceeds threshold")
        if pred_drift > 0.05:
            reasons.append(f"Prediction KS={pred_drift:.3f} exceeds threshold")
        if ctr_decay > 0.1:
            reasons.append(f"CTR Decay={ctr_decay*100:.1f}% exceeds threshold")
        report.needs_retraining = len(reasons) > 0
        report.reasons = reasons
        
        decision = evaluate_retraining_need(
            drift_report=report,
            days_since_last_training=days_stale,
            new_interactions_count=new_data_count,
            min_interactions_for_retrain=10000,
            max_days_without_retrain=14
        )
        
        # Display decision
        priority_colors = {
            "critical": "🔴 CRITICAL RETRAIN",
            "high": "🟠 HIGH PRIORITY RETRAIN",
            "normal": "🟡 SCHEDULED RETRAIN",
            "skip": "🟢 MODEL STABLE"
        }
        
        color_class = {
            "critical": "rgba(255,82,82,0.15)",
            "high": "rgba(255,167,38,0.15)",
            "normal": "rgba(255,215,64,0.15)",
            "skip": "rgba(0,230,118,0.15)"
        }
        
        border_color = {
            "critical": "#ff5252",
            "high": "#ffa726",
            "normal": "#ffd740",
            "skip": "#00e676"
        }
        
        reasons_list_html = "".join(f"<li>{r}</li>" for r in decision.trigger_reasons)
        if not reasons_list_html:
            reasons_list_html = "<li>No triggers detected. Baseline and drift features are stable.</li>"
            
        st.markdown(f"""
        <div style="background:{color_class[decision.priority]}; border-left: 5px solid {border_color[decision.priority]}; padding: 15px; border-radius: 4px; margin-top: 15px;">
            <h4 style="margin: 0 0 10px 0; color: {border_color[decision.priority]};">{priority_colors[decision.priority]}</h4>
            <p style="margin: 0 0 5px 0; font-size: 0.9rem;"><b>Decision:</b> {"Trigger retraining pipeline" if decision.should_retrain else "Skip retraining"}</p>
            <p style="margin: 0; font-size: 0.85rem;"><b>Reasons Evaluated:</b></p>
            <ul style="margin: 5px 0 10px 20px; font-size: 0.85rem; color: #ccc;">
                {reasons_list_html}
            </ul>
            <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 4px; font-family: monospace; font-size: 0.8rem;">
                <b>Recommended Retraining Config:</b><br/>
                - Epochs: {decision.recommended_config.get('epochs')}<br/>
                - Learning Rate: {decision.recommended_config.get('learning_rate')}<br/>
                - Batch Size: {decision.recommended_config.get('batch_size')}<br/>
                - Full Dataset: {decision.recommended_config.get('use_full_dataset')}<br/>
                - Extended Eval: {decision.recommended_config.get('run_extended_eval')}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")