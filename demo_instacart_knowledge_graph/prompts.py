"""Prompt templates for NL→Cypher translation."""

NL2CYPHER_SYSTEM = """You are a Cypher query generator for a Neo4j grocery market basket graph.

GRAPH SCHEMA:
Nodes:
  (:Product {id, name, category, price, aisle})
  (:Category {name})
  (:Aisle {name})

Relationships:
  (:Product)-[:FREQUENTLY_BOUGHT_WITH {confidence}]->(:Product)
  (:Product)-[:OFTEN_IN_SAME_RECIPE {confidence}]->(:Product)
  (:Product)-[:PAIRS_WELL_WITH {confidence}]->(:Product)
  (:Product)-[:WEEKEND_SNACK_BASKET {confidence}]->(:Product)
  (:Product)-[:HEALTH_CONSCIOUS_BASKET {confidence}]->(:Product)
  (:Product)-[:IN_CATEGORY]->(:Category)
  (:Product)-[:IN_AISLE]->(:Aisle)

RULES:
- Return ONLY valid Cypher, no explanation
- Always use RETURN to return results
- Use LIMIT 10 by default unless the user specifies otherwise
- Variable names should be descriptive (p, related, cat, aisle, etc.)"""

NL2CYPHER_PROMPT = """Convert this natural language question to a Cypher query:

"{query}"

Return ONLY the Cypher query, nothing else."""

EXPLAIN_RESULTS_SYSTEM = """You are a data analyst explaining product graph query results in plain English.
Be concise (2-3 sentences), mention specific product names, and highlight interesting patterns."""

EXPLAIN_RESULTS_PROMPT = """The user asked: "{query}"
The Cypher query returned these results:
{results}

Explain the findings in plain English."""


MOCK_CYPHER_QUERIES = {
    "frequently bought": "MATCH (p:Product)-[r:FREQUENTLY_BOUGHT_WITH]->(related:Product)\nRETURN p.name AS product, related.name AS bought_with, r.confidence AS confidence\nORDER BY r.confidence DESC\nLIMIT 10",
    "organic": "MATCH (p:Product)-[r]->(related:Product)\nWHERE p.name CONTAINS 'Organic'\nRETURN p.name AS product, type(r) AS relationship, related.name AS related_to, r.confidence\nORDER BY r.confidence DESC\nLIMIT 10",
    "produce": "MATCH (p:Product {aisle: 'Produce'})-[r]->(related:Product)\nRETURN p.name AS product, type(r) AS relationship, related.name AS related_to, r.confidence\nORDER BY r.confidence DESC\nLIMIT 10",
    "recipe": "MATCH (p:Product)-[r:OFTEN_IN_SAME_RECIPE]->(related:Product)\nRETURN p.name AS ingredient_1, related.name AS ingredient_2, r.confidence\nORDER BY r.confidence DESC\nLIMIT 10",
    "expensive": "MATCH (p:Product)\nWHERE p.price > 5.0\nRETURN p.name, p.category, p.aisle, p.price\nORDER BY p.price DESC\nLIMIT 10",
}


def get_mock_cypher(query: str) -> str:
    query_lower = query.lower()
    for key, cypher in MOCK_CYPHER_QUERIES.items():
        if key in query_lower:
            return cypher
    return "MATCH (p:Product)-[r]->(related:Product)\nRETURN p.name AS product, type(r) AS relationship, related.name AS related_to, r.confidence\nORDER BY r.confidence DESC\nLIMIT 10"
