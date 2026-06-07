"""
Customer 360 RAG service.

Pipeline:
  1. Ingest interaction logs → chunk → embed → ChromaDB
  2. Customer query → retrieve relevant interactions via RAG
  3. Context + customer profile → Gemini → persona synthesis
"""
import logging
import weaviate

from shared.gemini_client import GeminiClient
from shared.config import settings
from shared.mock_data import generate_interactions, get_customer_profiles, PRODUCT_NAMES
from demo_customer_support_360.models import PersonaInsight, CustomerProfile
from demo_customer_support_360.prompts import build_persona_prompt, PERSONA_SYSTEM_INSTRUCTION

logger = logging.getLogger(__name__)


class Customer360Service:
    """RAG-powered customer intelligence service."""

    def __init__(self, gemini: GeminiClient):
        self.gemini = gemini
        self._profiles = {p["id"]: p for p in get_customer_profiles()}
        self._interactions: dict[str, list[dict]] = {}
        self._table = None
        self._db = None
        self._client = None
        self._setup_rag_store()
        self._ingest_default_data()

    def _setup_rag_store(self):
        try:
            import weaviate
            self._client = weaviate.Client(
                url=settings.WEAVIATE_URL,
                auth_client_secret=weaviate.AuthApiKey(settings.WEAVIATE_API_KEY) if settings.WEAVIATE_API_KEY else None,
                timeout_config=(5, 15),
            )
            self._setup_weaviate_schema()
            logger.info("✅ Customer 360 Weaviate RAG store initialized")
        except Exception as e:
            logger.warning(f"⚠️ Weaviate RAG store setup failed: {e}")
            # fallback to LanceDB
            try:
                import lancedb
                self._db = lancedb.connect("memory://")
                logger.info("✅ Fallback LanceDB RAG store initialized (in-memory)")
            except Exception as le:
                logger.error(f"❌ Fallback LanceDB initialization failed: {le}")
                self._client = None
                self._db = None
    def _setup_weaviate_schema(self):
        """Create the CustomerInteraction class in Weaviate if it does not exist."""
        class_schema = {
            "class": "CustomerInteraction",
            "properties": [
            {"name": "interaction_id", "dataType": ["string"]},
            {"name": "customer_id", "dataType": ["string"]},
                {"name": "type", "dataType": ["string"]},
                {"name": "summary", "dataType": ["text"]},
                {"name": "date", "dataType": ["date"]},
            ],
            "vectorizer": "text2vec-openai",
        }
        try:
            if not self._client.schema.contains(class_schema["class"]):
                self._client.schema.create_class(class_schema)
        except Exception as e:
            logger.warning(f"⚠️ Weaviate schema setup failed: {e}")
        

    def _ingest_default_data(self):
        """Pre-load interaction data for all demo customers and index them."""
        all_data = []
        for cid in self._profiles:
            interactions = generate_interactions(cid, 12)
            self._interactions[cid] = interactions

            if self._db is not None:
                for i in interactions:
                    doc = f"[{i['type']}] {i['summary']}"
                    vector = self.gemini.get_embedding(doc)
                    all_data.append({
                        "vector": vector,
                        "id": i["interaction_id"],
                        "customer_id": cid,
                        "type": i["type"],
                        "summary": doc,
                        "date": i["date"]
                    })
                    # Also ingest into Weaviate if available
                    if self._client is not None:
                        try:
                            self._client.data_object.create(
                                data_object={
                                    "interaction_id": i["interaction_id"],
                                    "customer_id": cid,
                                    "type": i["type"],
                                    "summary": doc,
                                    "date": i["date"]
                                },
                                class_name="CustomerInteraction",
                                vector=vector,
                            )
                        except Exception as e:
                            logger.warning(f"⚠️ Weaviate ingestion failed for interaction {i['interaction_id']}: {e}")

        if self._db is not None and all_data:
            try:
                self._table = self._db.create_table("customer_interactions", data=all_data, mode="overwrite")
                logger.info(f"✅ Ingested {len(all_data)} interactions into LanceDB")
            except Exception as e:
                logger.warning(f"⚠️ Ingestion into LanceDB failed: {e}")

        logger.info(f"✅ Loaded interactions for {len(self._profiles)} customers")

    def get_profiles(self) -> list[CustomerProfile]:
        return [CustomerProfile(**p) for p in self._profiles.values()]

    def get_interactions(self, customer_id: str) -> list[dict]:
        return self._interactions.get(customer_id, [])

    async def generate_persona(self, customer_id: str) -> tuple[CustomerProfile | None, PersonaInsight, int]:
        """Generate a customer persona using RAG + Gemini."""
        profile_data = self._profiles.get(customer_id)
        if not profile_data:
            return None, PersonaInsight(summary="Customer not found"), 0

        profile = CustomerProfile(**profile_data)

        # RAG: retrieve relevant interactions
        interactions = self._retrieve_interactions(customer_id)
        rag_count = len(interactions)

        # Gemini synthesis
        if self.gemini.is_live:
            persona = await self._synthesize_live(profile_data, interactions)
        else:
            persona = self._synthesize_mock(profile_data, interactions)

        return profile, persona, rag_count

    def _retrieve_interactions(self, customer_id: str, top_k: int = 10) -> list[dict]:
        """Retrieve interactions via RAG or direct lookup."""
        # Prefer Weaviate if client is available
        if self._client is not None:
            try:
                query = f"customer {customer_id} interactions history"
                query_vector = self.gemini.get_embedding(query)
                # Use GraphQL with nearVector and filter by customer_id
                result = self._client.query.get(
                    class_name="CustomerInteraction",
                    properties=["interaction_id", "customer_id", "type", "summary", "date"]
                ).with_near_vector({"vector": query_vector}).with_where({
                    "path": ["customer_id"],
                    "operator": "Equal",
                    "valueString": customer_id
                }).with_limit(top_k).do()
                objs = result.get("data", {}).get("Get", {}).get("CustomerInteraction", [])
                interactions = []
                for obj in objs:
                    interactions.append({
                        "interaction_id": obj.get("interaction_id"),
                        "customer_id": obj.get("customer_id"),
                        "type": obj.get("type"),
                        "summary": obj.get("summary"),
                        "date": obj.get("date")
                    })
                return sorted(interactions, key=lambda x: x["date"], reverse=True)
            except Exception as e:
                logger.warning(f"Weaviate RAG retrieval failed: {e}")
        # Fallback to LanceDB if available
        if self._table is not None:
            try:
                query = f"customer {customer_id} interactions history"
                query_vector = self.gemini.get_embedding(query)
                results = (
                    self._table.search(query_vector)
                    .where(f"customer_id = '{customer_id}'")
                    .metric("cosine")
                    .limit(top_k)
                    .to_list()
                )
                interactions = []
                for res in results:
                    interactions.append({
                        "interaction_id": res["id"],
                        "customer_id": res["customer_id"],
                        "type": res["type"],
                        "summary": res["summary"],
                        "date": res["date"]
                    })
                return sorted(interactions, key=lambda x: x["date"], reverse=True)
            except Exception as e:
                logger.warning(f"LanceDB RAG retrieval failed: {e}")
        # Final fallback to in‑memory interactions
        return self._interactions.get(customer_id, [])[:top_k]


    async def _synthesize_live(self, profile: dict, interactions: list[dict]) -> PersonaInsight:
        prompt = build_persona_prompt(profile, interactions)
        data = self.gemini.generate_json(prompt, system_instruction=PERSONA_SYSTEM_INSTRUCTION)
        if "error" not in data:
            try:
                return PersonaInsight(**data)
            except Exception:
                pass
        return self._synthesize_mock(profile, interactions)

    def _synthesize_mock(self, profile: dict, interactions: list[dict]) -> PersonaInsight:
        """Generate a rich mock persona based on interaction analysis."""
        name = profile.get("name", "Customer")
        segment = profile.get("segment", "Standard")
        ltv = profile.get("ltv", 0)
        tenure = profile.get("tenure_months", 0)

        # Analyze interactions for insights
        types = [i.get("type", "") for i in interactions]
        has_returns = any("return" in t for t in types)
        has_complaints = any("complaint" in i.get("summary", "").lower() or "frustrated" in i.get("summary", "").lower() for i in interactions)
        purchase_count = sum(1 for t in types if t == "purchase")

        churn = "HIGH" if has_complaints and has_returns else "MEDIUM" if has_complaints else "LOW"
        trend = "DECLINING" if has_complaints else "IMPROVING" if purchase_count >= 3 else "STABLE"
        ltv_tier = "HIGH" if ltv > 2000 else "MEDIUM" if ltv > 500 else "LOW"

        return PersonaInsight(
            summary=f"{name} is a {tenure}-month {segment} customer with ${ltv:,.2f} lifetime value. "
                    f"Recent activity shows {len(interactions)} interactions across {len(set(types))} channels. "
                    f"{'Shows signs of frustration requiring attention.' if has_complaints else 'Engagement pattern is healthy.'}",
            segment=segment,
            lifetime_value_tier=ltv_tier,
            preferences=["Mobile app purchases", "Quick resolution support", f"Prefers {segment.lower()} tier benefits"],
            pain_points=["Shipping delays" if has_returns else "None identified",
                         "Response time expectations" if has_complaints else "Standard experience"],
            sentiment_trend=trend,
            churn_risk=churn,
            recommendations=[
                f"{'Proactive outreach with loyalty offer' if churn == 'HIGH' else 'Continue standard engagement'}",
                f"{'Escalate to retention team' if churn == 'HIGH' else 'Offer tier upgrade path'}",
                "Personalize next campaign based on purchase history",
            ],
            key_interactions=[
                {"date": i["date"][:10], "type": i.get("type", ""), "summary": i.get("summary", "")[:100]}
                for i in interactions[:5]
            ],
        )
