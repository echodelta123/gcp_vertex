"""
Graph Explorer service: NL → Cypher → execute → explain.

Supports two graph backends:
  1. Neo4j Aura (when NEO4J_URI is configured)
  2. In-memory mock graph (DEMO_MODE, no Neo4j required)
"""
import logging
from shared.gemini_client import GeminiClient
from shared.config import settings
from shared.mock_data import get_graph_data
from demo_instacart_knowledge_graph.models import GraphNode, GraphEdge, GraphStats
from demo_instacart_knowledge_graph.prompts import (
    NL2CYPHER_SYSTEM, NL2CYPHER_PROMPT, EXPLAIN_RESULTS_SYSTEM,
    EXPLAIN_RESULTS_PROMPT, get_mock_cypher,
)

logger = logging.getLogger(__name__)


class GraphExplorerService:
    """NL-to-Cypher graph exploration service."""

    def __init__(self, gemini: GeminiClient):
        self.gemini = gemini
        self._graph_data = get_graph_data()
        self._neo4j_driver = None
        self._setup_neo4j()

    def _setup_neo4j(self):
        """Try connecting to Neo4j if configured."""
        if settings.NEO4J_PASSWORD and not settings.effective_demo_mode:
            try:
                from neo4j import GraphDatabase
                self._neo4j_driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                )
                self._neo4j_driver.verify_connectivity()
                logger.info("✅ Connected to Neo4j")
            except Exception as e:
                logger.warning(f"⚠️ Neo4j connection failed: {e}")
                self._neo4j_driver = None
        else:
            logger.info("🔶 Graph Explorer using in-memory mock graph")

    def get_stats(self) -> GraphStats:
        nodes = self._graph_data["nodes"]
        edges = self._graph_data["edges"]
        rel_types = list(set(e["relation"] for e in edges))
        return GraphStats(
            total_products=len(nodes["products"]),
            total_categories=len(nodes["categories"]),
            total_aisles=len(nodes["aisles"]),
            total_relationships=len(edges),
            relationship_types=rel_types,
        )

    async def query(self, nl_query: str) -> dict:
        """Full NL → Cypher → Execute → Explain pipeline."""
        # Step 1: Generate Cypher
        cypher = await self._generate_cypher(nl_query)

        # Step 2: Execute query
        results, nodes, edges = self._execute_query(cypher, nl_query)

        # Step 3: Explain results
        explanation = await self._explain_results(nl_query, results)

        return {
            "cypher": cypher,
            "results": results,
            "nodes": nodes,
            "edges": edges,
            "explanation": explanation,
        }

    async def _generate_cypher(self, nl_query: str) -> str:
        """Translate NL to Cypher via Gemini."""
        if self.gemini.is_live:
            prompt = NL2CYPHER_PROMPT.format(query=nl_query)
            raw = self.gemini.generate(prompt, system_instruction=NL2CYPHER_SYSTEM)
            # Clean up response
            cypher = raw.strip().strip("`").strip()
            if cypher.startswith("cypher\n"):
                cypher = cypher[7:]
            return cypher

        return get_mock_cypher(nl_query)

    def _execute_query(self, cypher: str, nl_query: str) -> tuple[list[dict], list[GraphNode], list[GraphEdge]]:
        """Execute Cypher on Neo4j or simulate on mock graph."""
        if self._neo4j_driver:
            return self._execute_neo4j(cypher)
        return self._execute_mock(nl_query)

    def _execute_neo4j(self, cypher: str) -> tuple[list, list, list]:
        """Execute against real Neo4j."""
        try:
            with self._neo4j_driver.session() as session:
                result = session.run(cypher)
                records = [dict(r) for r in result]
                # Extract nodes/edges from results (simplified)
                return records, [], []
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return [{"error": str(e)}], [], []

    def _execute_mock(self, nl_query: str) -> tuple[list[dict], list[GraphNode], list[GraphEdge]]:
        """Simulate query execution on mock graph data."""
        products = self._graph_data["nodes"]["products"]
        edges = self._graph_data["edges"]
        query_lower = nl_query.lower()

        # Build lookup
        prod_map = {p["id"]: p for p in products}

        # Find relevant edges
        relevant_edges = []
        for e in edges:
            src = prod_map.get(e["source"], {})
            tgt = prod_map.get(e["target"], {})
            src_text = f"{src.get('name', '')} {src.get('category', '')} {src.get('aisle', '')}".lower()
            tgt_text = f"{tgt.get('name', '')} {tgt.get('category', '')} {tgt.get('aisle', '')}".lower()

            # Check if query relates to any node in this edge
            query_words = query_lower.split()
            if any(w in src_text or w in tgt_text or w in e["relation"].lower() for w in query_words):
                relevant_edges.append(e)

        if not relevant_edges:
            relevant_edges = edges[:5]  # Default: show first 5

        # Build results, nodes, edges for visualization
        results = []
        graph_nodes = {}
        graph_edges = []

        for edge in relevant_edges:
            src = prod_map.get(edge["source"], {})
            tgt = prod_map.get(edge["target"], {})
            results.append({
                "product": src.get("name", ""),
                "relationship": edge["relation"],
                "related_to": tgt.get("name", ""),
                "confidence": edge["confidence"],
            })
            # Nodes
            for p_data in [src, tgt]:
                if p_data and p_data["id"] not in graph_nodes:
                    graph_nodes[p_data["id"]] = GraphNode(
                        id=p_data["id"], label=p_data["name"],
                        type="Product",
                        properties={"category": p_data.get("category", ""),
                                    "price": p_data.get("price", 0),
                                    "aisle": p_data.get("aisle", "")},
                    )
            graph_edges.append(GraphEdge(
                source=edge["source"], target=edge["target"],
                relation=edge["relation"], confidence=edge["confidence"],
            ))

        return results, list(graph_nodes.values()), graph_edges

    async def _explain_results(self, query: str, results: list[dict]) -> str:
        """Use Gemini to explain results in plain English."""
        if self.gemini.is_live and results:
            import json
            prompt = EXPLAIN_RESULTS_PROMPT.format(
                query=query, results=json.dumps(results[:10], indent=2)
            )
            return self.gemini.generate(prompt, system_instruction=EXPLAIN_RESULTS_SYSTEM)

        # Mock explanation
        if results:
            first = results[0]
            return (
                f"The graph shows that {first.get('product', 'products')} is connected to "
                f"{first.get('related_to', 'other items')} via a {first.get('relationship', 'relationship').replace('_', ' ').lower()} "
                f"relationship (confidence: {first.get('confidence', 0):.0%}). "
                f"Found {len(results)} relevant connections in the product knowledge graph."
            )
        return "No results found for this query."

    def get_all_graph_data(self) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Return full graph for visualization."""
        products = self._graph_data["nodes"]["products"]
        edges = self._graph_data["edges"]

        nodes = [
            GraphNode(id=p["id"], label=p["name"], type="Product",
                      properties={"category": p["category"], "price": p["price"], "aisle": p["aisle"]})
            for p in products
        ]
        # Add category nodes
        for cat in self._graph_data["nodes"]["categories"]:
            nodes.append(GraphNode(id=f"cat_{cat}", label=cat, type="Category", properties={}))

        graph_edges = [
            GraphEdge(source=e["source"], target=e["target"],
                      relation=e["relation"], confidence=e["confidence"])
            for e in edges
        ]
        # Add category edges
        for p in products:
            graph_edges.append(GraphEdge(
                source=p["id"], target=f"cat_{p['category']}",
                relation="IN_CATEGORY", confidence=1.0,
            ))

        return nodes, graph_edges
