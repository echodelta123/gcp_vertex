"""
Streamlit frontend for Customer 360 Intelligence Platform.

Features:
  - Customer selector with profile cards
  - AI-generated persona with churn risk and sentiment trend
  - Interaction timeline visualization
  - RAG source attribution panel
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import asyncio

from shared.gemini_client import GeminiClient
from demo_customer_support_360.services import Customer360Service

st.set_page_config(page_title="Customer Support 360 | Vertex AI Demo", page_icon="👤", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .profile-card {
        background: linear-gradient(135deg, #1e1e30, #2a2a40);
        border: 1px solid rgba(255,255,255,0.08); border-radius: 16px;
        padding: 24px; text-align: center;
    }
    .profile-name { font-size: 1.6rem; font-weight: 700; color: #e0e0e0; }
    .profile-segment { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1.5px; }
    .risk-low { color: #00e676; } .risk-medium { color: #ffd740; } .risk-high { color: #ff5252; }
    .trend-improving { color: #00e676; } .trend-stable { color: #ffd740; } .trend-declining { color: #ff5252; }
    .insight-card {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 16px; margin: 8px 0;
    }
    .tag { display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 16px;
           font-size: 0.8rem; background: rgba(68,138,255,0.12); color: #448aff;
           border: 1px solid rgba(68,138,255,0.2); }
    .pain-tag { background: rgba(255,82,82,0.12); color: #ff5252; border-color: rgba(255,82,82,0.2); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    return Customer360Service(GeminiClient())

service = get_service()

# Sidebar
with st.sidebar:
    st.markdown("## 👤 Select Customer")
    profiles = service.get_profiles()
    customer_options = {f"{p.name} ({p.id})": p.id for p in profiles}
    selected = st.selectbox("Customer:", list(customer_options.keys()))
    selected_id = customer_options[selected]

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Customer Support 360 Intelligence**  
    RAG pipeline synthesizes interaction history into actionable personas.
    
    Powered by LanceDB + Google Gemini.
    
    Inspired by [GCP Customer Experience Modernization](https://github.com/GoogleCloudPlatform/customer-experience-modernization).
    """)

# Main
st.markdown("# 👤 Customer Support 360 Intelligence Platform")
st.markdown("*RAG-powered customer understanding from interaction history*")
st.markdown("---")

# Generate persona
if st.button("🧠 Generate Customer Persona", type="primary", use_container_width=True):
    with st.spinner("Retrieving interactions and synthesizing persona..."):
        profile, persona, rag_count = asyncio.run(service.generate_persona(selected_id))

    if profile and persona:
        st.session_state["persona"] = persona
        st.session_state["profile"] = profile
        st.session_state["rag_count"] = rag_count

if "persona" in st.session_state:
    persona = st.session_state["persona"]
    profile = st.session_state["profile"]
    rag_count = st.session_state["rag_count"]

    # Profile header
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"""<div class="profile-card">
        <div class="profile-name">{profile.name}</div>
        <div class="profile-segment">{persona.segment}</div>
    </div>""", unsafe_allow_html=True)

    risk_cls = f"risk-{persona.churn_risk.lower()}"
    trend_cls = f"trend-{persona.sentiment_trend.lower()}"
    col2.metric("Lifetime Value", f"${profile.ltv:,.2f}", delta=persona.lifetime_value_tier)
    col3.markdown(f"**Churn Risk**<br><span class='{risk_cls}' style='font-size:1.8rem;font-weight:700;'>{persona.churn_risk}</span>", unsafe_allow_html=True)
    col4.markdown(f"**Sentiment Trend**<br><span class='{trend_cls}' style='font-size:1.8rem;font-weight:700;'>{persona.sentiment_trend}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Summary
    st.markdown(f"### 📋 Executive Summary")
    st.info(persona.summary)
    st.caption(f"Based on {rag_count} interactions retrieved via RAG pipeline")

    # Two columns: preferences/pain points + recommendations
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### ✅ Preferences")
        prefs_html = "".join(f'<span class="tag">{p}</span>' for p in persona.preferences)
        st.markdown(prefs_html, unsafe_allow_html=True)

        st.markdown("### ⚠️ Pain Points")
        pains_html = "".join(f'<span class="tag pain-tag">{p}</span>' for p in persona.pain_points)
        st.markdown(pains_html, unsafe_allow_html=True)

    with col_b:
        st.markdown("### 🎯 Recommended Actions")
        for i, rec in enumerate(persona.recommendations, 1):
            st.markdown(f"**{i}.** {rec}")

    # Interaction timeline
    st.markdown("---")
    st.markdown("### 📅 Interaction Timeline")
    interactions = service.get_interactions(selected_id)
    if interactions:
        timeline_df = pd.DataFrame(interactions)
        timeline_df["date"] = pd.to_datetime(timeline_df["date"])
        timeline_df = timeline_df.sort_values("date")

        type_colors = {"purchase": "#00e676", "phone_support": "#448aff", "email_ticket": "#ffd740",
                       "app_feedback": "#e040fb", "survey_response": "#ff9100", "twitter_public": "#ff5252",
                       "twitter_dm": "#00bcd4"}

        fig = px.scatter(
            timeline_df, x="date", y="type", color="type",
            color_discrete_map=type_colors, size_max=12,
            hover_data=["summary"], title="Customer Journey Timeline",
        )
        fig.update_traces(marker=dict(size=14, line=dict(width=1, color="#333")))
        fig.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ccc"}, yaxis={"gridcolor": "#222"}, xaxis={"gridcolor": "#222"},
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Details table
        st.dataframe(
            timeline_df[["date", "type", "summary", "satisfaction_score"]].sort_values("date", ascending=False),
            use_container_width=True, hide_index=True,
        )

st.markdown("---")
