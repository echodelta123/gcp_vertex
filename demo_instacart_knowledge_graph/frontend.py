"""
Streamlit frontend for Knowledge Graph Explorer.

Features:
  - Natural language graph queries
  - Interactive pyvis graph visualization
  - Generated Cypher display
  - AI-powered result explanations
  - Graph schema explorer
  - Query history
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import asyncio
import tempfile

from shared.gemini_client import GeminiClient
from demo_instacart_knowledge_graph.services import GraphExplorerService

st.set_page_config(page_title="Instacart Knowledge Graph | Vertex AI Demo", page_icon="🕸️", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .cypher-block {
        background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
        padding: 16px; font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem; color: #00e676; white-space: pre-wrap;
        line-height: 1.6;
    }
    .explanation-box {
        background: linear-gradient(135deg, rgba(68,138,255,0.08), rgba(68,138,255,0.02));
        border-left: 4px solid #448aff; padding: 16px 20px;
        border-radius: 0 8px 8px 0; margin: 16px 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #1e1e30, #2a2a40);
        border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
        padding: 16px; text-align: center;
    }
    .stat-value { font-size: 2rem; font-weight: 700; color: #e040fb; }
    .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service():
    return GraphExplorerService(GeminiClient())

service = get_service()


def render_graph_pyvis(nodes, edges):
    """Render interactive graph using pyvis."""
    try:
        from pyvis.network import Network

        net = Network(height="500px", width="100%", bgcolor="#0d1117",
                      font_color="#e0e0e0", directed=True)
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=200)

        # Color map
        type_colors = {
            "Product": "#448aff", "Category": "#e040fb",
            "Aisle": "#00e676", "default": "#ffd740",
        }

        for node in nodes:
            color = type_colors.get(node.type, type_colors["default"])
            title = f"{node.label}\nType: {node.type}"
            if node.properties:
                for k, v in node.properties.items():
                    title += f"\n{k}: {v}"
            net.add_node(
                node.id, label=node.label, color=color,
                title=title, size=25 if node.type == "Product" else 18,
                font={"size": 12, "color": "#e0e0e0"},
            )

        edge_colors = {
            "FREQUENTLY_BOUGHT_WITH": "#00e676",
            "STYLE_MATCH": "#448aff",
            "SEASONAL_PAIR": "#ffd740",
            "SIMILAR_STYLE": "#e040fb",
            "LIFESTYLE_MATCH": "#ff9100",
            "IN_CATEGORY": "#666",
        }
        for edge in edges:
            color = edge_colors.get(edge.relation, "#555")
            label = edge.relation.replace("_", " ").title()
            net.add_edge(
                edge.source, edge.target,
                title=f"{label} ({edge.confidence:.0%})",
                color=color, width=max(1, edge.confidence * 3),
                label=f"{edge.confidence:.0%}" if edge.relation != "IN_CATEGORY" else "",
            )

        # Save to temp file and display
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            net.save_graph(f.name)
            with open(f.name, "r") as rf:
                html = rf.read()
            return html
    except ImportError:
        return None


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📊 Graph Statistics")
    stats = service.get_stats()
    st.markdown(f"""
    <div class="stat-card"><div class="stat-value">{stats.total_products}</div><div class="stat-label">Products</div></div>
    """, unsafe_allow_html=True)
    st.markdown("")
    col_a, col_b = st.columns(2)
    col_a.metric("Categories", stats.total_categories)
    col_b.metric("Relationships", stats.total_relationships)

    st.markdown("**Relationship Types:**")
    for rt in stats.relationship_types:
        st.markdown(f"  → `{rt}`")

    st.markdown("---")
    st.markdown("### 💡 Try These Queries")
    samples = [
        "What products are frequently bought with Organic Bananas?",
        "Show me expensive produce items",
        "Which products are frequently bought together?",
        "Find ingredients often used in the same recipe",
        "What organic products do you have?",
    ]
    for q in samples:
        if st.button(f"→ {q}", key=f"gq_{q}"):
            st.session_state["graph_query"] = q

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Instacart Knowledge Graph**  
    NL → Cypher → Neo4j pipeline for Market Basket Analysis.
    
    Inspired by Kaggle's Instacart Market Basket challenge.
    """)

# --- Main ---
st.markdown("# 🕸️ Instacart Knowledge Graph")
st.markdown("*Natural language queries over a product knowledge graph*")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Query Graph", "🗺️ Full Graph View"])

with tab1:
    query = st.text_input(
        "Ask a question about the product graph:",
        value=st.session_state.get("graph_query", ""),
        placeholder="e.g., 'What products are frequently bought with denim jeans?'",
    )

    if st.button("🔍 Query Graph", type="primary", use_container_width=True) and query:
        with st.spinner("Translating to Cypher and querying graph..."):
            result = asyncio.run(service.query(query))

        # Generated Cypher
        st.markdown("### 🔧 Generated Cypher")
        st.markdown(f'<div class="cypher-block">{result["cypher"]}</div>', unsafe_allow_html=True)

        # Explanation
        st.markdown(f'<div class="explanation-box">💡 <b>Insight:</b> {result["explanation"]}</div>',
                     unsafe_allow_html=True)

        # Graph visualization
        if result["nodes"]:
            st.markdown("### 🕸️ Graph Visualization")
            html = render_graph_pyvis(result["nodes"], result["edges"])
            if html:
                components.html(html, height=520, scrolling=False)
            else:
                st.info("Install `pyvis` for interactive graph visualization.")

        # Results table
        if result["results"]:
            st.markdown("### 📋 Query Results")
            df = pd.DataFrame(result["results"])
            st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Full Product Knowledge Graph")
    all_nodes, all_edges = service.get_all_graph_data()
    html = render_graph_pyvis(all_nodes, all_edges)
    if html:
        components.html(html, height=600, scrolling=False)
    else:
        st.info("Install `pyvis` for interactive graph visualization: `pip install pyvis`")

    # Legend
    st.markdown("""
    **Legend:**
    🔵 Product  |  🟣 Category  |  🟢 Aisle
    
    **Edge Colors:**
    🟢 Frequently Bought With  |  🔵 Pairs Well With  |  🟡 Recipe Match  |  🟣 Diet Basket
    """)

st.markdown("---")
