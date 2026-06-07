# demo_recommendation_engine/vector_search.py
"""VectorSearch encapsulates all vector‑store interactions for the recommendation demo.
It currently uses LanceDB (in‑memory) with a deterministic mock fallback when LanceDB is
unavailable or when the demo runs in `DEMO_MODE`.
"""

import logging
from typing import List, Dict

from shared.gemini_client import GeminiClient
from shared.mock_data import generate_product_catalog

logger = logging.getLogger(__name__)


class VectorSearch:
    def __init__(self, gemini: GeminiClient):
        self._gemini = gemini
        self._catalog: List[Dict] = []
        self._db = None
        self._table = None
        self._setup()

    def _setup(self) -> None:
        """Initialise the LanceDB connection if the library is present.
        In demo mode we simply keep the attributes ``None`` – the higher‑level
        methods will fall back to keyword search.
        """
        if self._gemini.is_live:
            try:
                import lancedb
                self._db = lancedb.connect("memory://")
                logger.info("✅ LanceDB vector store initialized (in‑memory)")
            except Exception as e:
                logger.warning(f"⚠️ LanceDB init failed ({e}); using mock fallback")
        else:
            logger.info("🔶 VectorSearch running in DEMO_MODE – no real DB")

    def ingest_catalog(self, products: List[Dict] | None = None) -> int:
        """Populate the vector store with product embeddings.
        Returns the number of ingested products.
        """
        if products is None:
            products = generate_product_catalog()
        self._catalog = products

        if self._db is not None:
            try:
                data = []
                for p in products:
                    text = (
                        f"{p['name']} - {p['description']} "
                        f"Category: {p['category']} Tags: {', '.join(p.get('tags', []))}"
                    )
                    vector = self._gemini.get_embedding(text)
                    data.append(
                        {
                            "vector": vector,
                            "id": p["product_id"],
                            "name": p["name"],
                            "category": p["category"],
                            "price": p["price"],
                            "description": p["description"],
                        }
                    )
                self._table = self._db.create_table("products", data=data, mode="overwrite")
                logger.info(f"✅ Ingested {len(products)} products into LanceDB")
            except Exception as e:
                logger.warning(f"⚠️ Ingestion into LanceDB failed: {e}")
        else:
            logger.debug("VectorSearch: using in‑memory catalog only (no DB)")
        return len(products)

    def get_catalog(self) -> List[Dict]:
        """Return the raw product catalog (may be empty if not ingested)."""
        return self._catalog

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Public entry‑point used by the service façade.
        It tries a semantic (vector) search and falls back to a lightweight
        keyword search when the vector store is unavailable.
        """
        if self._table is not None:
            return self._semantic_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform a cosine‑similarity search using LanceDB.
        The returned dictionaries include a ``similarity_score`` field.
        """
        try:
            query_vec = self._gemini.get_embedding(query)
            results = (
                self._table.search(query_vec)
                .metric("cosine")
                .limit(min(top_k, len(self._catalog)))
                .to_list()
            )
            matched: List[Dict] = []
            for res in results:
                pid = res["id"]
                product = next((p for p in self._catalog if p["product_id"] == pid), None)
                if product:
                    dist = res.get("_distance", 0.0)
                    product_copy = {**product, "similarity_score": round(1 - dist, 3)}
                    matched.append(product_copy)
            return matched
        except Exception as e:
            logger.warning(f"LanceDB semantic search failed: {e}")
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword search used when the vector DB is unavailable.
        It scores products by the proportion of query words present in the
        product's name/description/tags.
        """
        query_lower = query.lower()
        scored: List[tuple[Dict, float]] = []
        for p in self._catalog:
            text = f"{p['name']} {p['description']} {' '.join(p.get('tags', []))}".lower()
            words = query_lower.split()
            score = sum(1 for w in words if w in text) / max(len(words), 1)
            if score > 0:
                scored.append(({**p, "similarity_score": round(score, 3)}, score))
        if not scored:
            # Return a random sample as a last‑ditch fallback
            import random
            sample = random.sample(self._catalog, min(top_k, len(self._catalog)))
            return [{**p, "similarity_score": 0.5} for p in sample]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:top_k]]
