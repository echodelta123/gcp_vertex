"""
Prompt templates for the Sentiment Intelligence Engine.
Structured prompts designed for reliable JSON output from Gemini.
"""

SENTIMENT_SYSTEM_INSTRUCTION = """You are an enterprise-grade sentiment analysis engine built for customer experience teams.
You analyze customer feedback with nuance, detecting not just overall sentiment but specific aspects, 
entities, urgency levels, and actionable insights.

CRITICAL RULES:
- Always return valid JSON matching the exact schema specified
- Be precise with confidence scores (use the full 0.0-1.0 range)
- Detect multiple aspects even in short texts
- Identify urgency based on customer frustration level and business impact
- Extract specific product names, people, organizations, and features as entities
"""

SENTIMENT_ANALYSIS_PROMPT = """Analyze the following customer feedback and return a structured JSON response.

{context_section}

TEXT TO ANALYZE:
\"\"\"{text}\"\"\"

Return ONLY valid JSON with this exact structure (no markdown fences, no extra text):
{{
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL" | "MIXED",
  "confidence": <float 0.0-1.0>,
  "aspects": [
    {{
      "aspect": "<specific fashion/apparel aspect like 'fit/sizing', 'fabric quality', 'style/design', 'durability', 'price'>",
      "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL" | "MIXED",
      "score": <float 0.0-1.0>
    }}
  ],
  "entities": [
    {{
      "name": "<entity name>",
      "type": "PRODUCT" | "PERSON" | "ORG" | "FEATURE" | "ISSUE"
    }}
  ],
  "key_phrases": ["<phrase1>", "<phrase2>"],
  "urgency": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "summary": "<1-sentence executive summary of the feedback>"
}}"""


def build_analysis_prompt(text: str, context: str | None = None) -> str:
    """Build the full analysis prompt with optional context."""
    context_section = ""
    if context:
        context_section = f"BUSINESS CONTEXT: This is a {context}.\n"
    return SENTIMENT_ANALYSIS_PROMPT.format(text=text, context_section=context_section)
