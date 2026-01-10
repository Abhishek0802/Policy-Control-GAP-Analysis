from app.state import AppState
from app.graph.flow import app
from app.rag.retriever import load_faiss_retriever, retrieve_evidence
from app.config import SCOPE


def run(requirement: str):
    retriever = load_faiss_retriever()
    evidence = retrieve_evidence(retriever, requirement)

    state = AppState(
        requirement=requirement,
        evidence=evidence,
        scope=SCOPE
    )

    out = app.invoke(state)
    return out


if __name__ == "__main__":
    req = "Incident response plan must define roles, SLAs, and testing cadence."
    result = run(req)

    print("\n--- Final state ---")
    print(result.model_dump())

    print("\n--- Audit log ---")
    for item in result.audit_log:
        print(item)
