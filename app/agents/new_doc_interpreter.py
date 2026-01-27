from typing import List, Dict
import json


def interpret_new_document(llm, text: str) -> List[Dict]:
    """
    Interpret a policy document and extract PRACTICAL, auditable control candidates.
    """

    prompt = """
You are a senior compliance auditor.

Your task is to analyze the policy document below and identify ONLY statements
that represent potential compliance controls.

For EACH identified statement:
1. Keep the ORIGINAL policy wording (do NOT rewrite into generic requirements)
2. Classify the control maturity:
   - INTENT_ONLY: high-level statement, not directly auditable
   - PARTIAL_CONTROL: relevant but missing key enforcement details
   - AUDIT_READY: clear, specific, and testable
3. Identify what is missing to make it fully auditable.
If rag_decision.include is false, copy missing_elements into rag_decision.blocking_factors.
4. Decide whether this control should be evaluated against formal frameworks (ISO/PDPC)

DECISION RULE (MANDATORY):
- If control_type == "INTENT_ONLY", then rag_decision.include MUST be false
- Only PARTIAL_CONTROL or AUDIT_READY may be included in RAG

IMPORTANT RULES:
- Ignore descriptive or aspirational text
- Prefer fewer, higher-quality controls
- Do NOT invent requirements not present in the policy

Return STRICT JSON only, as a list of objects in this schema:

Return STRICT JSON only, as a list of objects in this schema:

[
  {
    "statement": "...",
    "control_area": "...",
    "control_type": "INTENT_ONLY | PARTIAL_CONTROL | AUDIT_READY",
    "missing_elements": ["..."],
    "rag_decision": {
      "include": true | false,
      "reason": "...",
      "blocking_factors": ["..."]
    },
    "confidence": 0.0 to 1.0
  }
]

POLICY DOCUMENT:
""" + text

    response = llm.invoke(prompt)

    try:
        parsed = json.loads(response.content)
    except Exception:
        # Fail safe: return empty list instead of garbage
        return []

    return parsed