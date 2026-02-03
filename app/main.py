from app.state import AppState
from app.graph.flow import app
from app.rag.retriever import load_faiss_retriever, retrieve_evidence
from app.config import SCOPE

def run(requirement: str):
    """
    Main RAG flow:
    1. Load FAISS retriever
    2. Retrieve relevant evidence for the user requirement
    3. Create AppState and invoke the LLM workflow
    """
    # Step 1: Load FAISS retriever
    retriever = load_faiss_retriever()

    # Step 2: Retrieve top evidence chunks from FAISS
    evidence = retrieve_evidence(retriever, requirement)

    # Step 3: Package everything into AppState
    state = AppState(
        requirement=requirement,
        evidence=evidence,
        scope=SCOPE
    )

    # Step 4: Pass AppState to the workflow / LLM
    out = app.invoke(state)
    return out


if __name__ == "__main__":
    # Example user query
    req = "Incident response plan must define roles, SLAs, and testing cadence."
    
    # Run the RAG pipeline
    result = run(req)

    # Print final structured state
    print("\n--- Final state ---")
    print(result.model_dump())

    # Print audit log of steps taken
    print("\n--- Audit log ---")
    for item in result.audit_log:
        print(item)
