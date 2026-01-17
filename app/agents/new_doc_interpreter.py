# app/agents/new_doc_interpreter.py
from typing import List, Dict

def interpret_new_document(llm, text: str) -> List[Dict]:
    """
    Extract normalized requirements from a new document.
    """
    prompt = f"""
Extract clear, atomic compliance requirements from the document below.
Return each requirement as a bullet point.

DOCUMENT:
{text}
"""
    response = llm.invoke(prompt)

    requirements = []
    for line in response.content.split("\n"):
        line = line.strip("- ").strip()
        if line:
            requirements.append({
                "requirement": line,
                # "confidence": 0.75  # placeholder; can improve later
                "confidence" : 0.7 if "must" in line.lower() else 0.9
            })

    return requirements
