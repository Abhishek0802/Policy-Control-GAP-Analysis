import json
import re
from typing import Dict

def interpret_new_document(llm, text: str) -> Dict:
    prompt = f"""
    You are a Policy Analysis Assistant. 
    IMPORTANT: The text contains markers like , , etc. 
    IGNORE these markers for section numbering. Use the ACTUAL headings (e.g., "1. Purpose", "2. Scope").

    TASK:
    1. Extract Metadata: Title, Owner, Effective Date, and Scope[cite: 1, 2, 3, 5].
    2. List Clauses: Identify every numbered policy section (e.g., 1 through 8)[cite: 6, 8, 10, 13, 16, 19, 22, 24].
    3. Classify Themes: Provide a ONE-WORD theme for each.

    JSON SCHEMA:
    {{
      "metadata": {{
        "title": "...",
        "owner": "...",
        "effective_date": "...",
        "applies_to": "..."
      }},
      "analysis": [
        {{
          "section_reference": "Section X (Use the policy's number, not the source ID)",
          "exact_clause": "The full text of the section",
          "theme": "One Word Theme"
        }}
      ]
    }}

    POLICY DOCUMENT:
    {text}
    """

    response = llm.invoke(prompt)
    content = response.content

    # CLEANER: Removes ```json or ``` blocks if the LLM includes them
    clean_content = re.sub(r'^```json\s*|```\s*$', '', content.strip(), flags=re.MULTILINE)

    try:
        parsed = json.loads(clean_content)
        return parsed
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        # Return a structure that matches your template so it doesn't crash
        return {
            "metadata": {"title": "Error Parsing", "owner": "N/A", "effective_date": "N/A", "applies_to": "N/A"},
            "analysis": []
        }