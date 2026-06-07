"""FastAPI backend for Knowledge Graph Explorer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.gemini_client import GeminiClient
from demo_instacart_knowledge_graph.models import QueryRequest, QueryResponse, GraphStats, HealthResponse, GraphNode, GraphEdge
from demo_instacart_knowledge_graph.services import GraphExplorerService

logging.basicConfig(level=logging.INFO)

_service: GraphExplorerService | None = None


def get_service() -> GraphExplorerService:
    global _service
    if _service is None:
        _service = GraphExplorerService(GeminiClient())
    return _service


app = FastAPI(
    title="Knowledge Graph Explorer", version="1.0.0",
    description="Natural language queries over a product knowledge graph via NL→Cypher→Neo4j pipeline.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    svc = get_service()
    return HealthResponse(
        graph_status="neo4j" if svc._neo4j_driver else "in_memory_mock",
        demo_mode=settings.effective_demo_mode,
    )


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_graph(request: QueryRequest):
    """Translate natural language to Cypher, execute, and explain results."""
    try:
        result = await get_service().query(request.query)
        return QueryResponse(
            natural_language_query=request.query,
            generated_cypher=result["cypher"],
            explanation=result["explanation"],
            nodes=result["nodes"],
            edges=result["edges"],
            raw_results=result["results"],
            demo_mode=settings.effective_demo_mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/schema", response_model=GraphStats, tags=["Graph"])
async def graph_schema():
    """Get graph schema and statistics."""
    return get_service().get_stats()


@app.get("/api/v1/graph", tags=["Graph"])
async def full_graph():
    """Get the full graph data for visualization."""
    nodes, edges = get_service().get_all_graph_data()
    return {"nodes": [n.model_dump() for n in nodes], "edges": [e.model_dump() for e in edges]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=8004)
