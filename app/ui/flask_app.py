import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from sentence_transformers import CrossEncoder

from app.config import FAISS_INDEX_PATH, CHAT_MODEL
from app.state import AppState
from app.graph.flow import app as graph_app

from app.rag.vectorstore_indexer import build_faiss_from_folder, build_temp_faiss, get_vector_db

from app.agents.new_doc_interpreter import interpret_new_document

from langchain_openai import ChatOpenAI

import logging
logging.basicConfig(level=logging.INFO)

ALLOWED_EXTENSIONS = {"pdf", "txt"}

flask_app = Flask(__name__)
flask_app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@flask_app.get("/")
def index():
    return render_template("index.html")



@flask_app.post("/analyze")
def analyze():
    uploaded_file = request.files.get("file")   
    filename = secure_filename(uploaded_file.filename)
    
    if not uploaded_file or uploaded_file.filename == "":
        return "No file uploaded", 400
    
    os.makedirs("data/user_input", exist_ok=True)

    uploaded_pdf_path = os.path.join("data/user_input", filename)
    uploaded_file.save(uploaded_pdf_path)
    
    # 1. Build Index (or just load documents for the prompt)
    temp_faiss = build_temp_faiss(uploaded_pdf_path)

    # 2. Extract full text in order
    policy_text = "\n".join([doc.page_content for doc in temp_faiss.docstore._dict.values()])

    # 3. Call your updated prompt function
    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)
    interpreted_data = interpret_new_document(llm, policy_text)

    # 4. Save to config for later retrieval if needed
    flask_app.config["LAST_INTERPRETED"] = interpreted_data

    # 5. Render with specific keys
    return render_template(
        "interpreter_review.html",
        metadata=interpreted_data.get("metadata", {}),
        analysis=interpreted_data.get("analysis", [])
    )

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

@flask_app.post("/review/submit")
def submit_review():
    # Load the Internal Policies Vector DB (FAISS)
    db = get_vector_db()

    # 1. Capture the list of clauses checked by the user
    approved_texts = set(request.form.getlist("approved_clauses"))

    # 2. Retrieve the full interpretation dictionary from config
    interpreted_data = flask_app.config.get("LAST_INTERPRETED", {})
    analysis_items = interpreted_data.get("analysis", [])
     
    results = []

    # 3. Process only the approved clauses through the Graph
    for item in analysis_items:
        current_clause = item.get("exact_clause")
        if current_clause not in approved_texts:
            continue

        # 4. RETRIEVER: Search the knowledge base for the top regulatory requirements    
        initial_results = db.similarity_search(current_clause, k=5)

        # 5. Reranking documents via Cross-Encoder
        pairs = [[current_clause, doc.page_content] for doc in initial_results]
        scores = reranker_model.predict(pairs)
        scored_docs = sorted(zip(scores, initial_results), key=lambda x: x[0], reverse=True)
        top_reranked_docs = [doc for score, doc in scored_docs[:2]]
        dynamic_scope = "\n\n".join([doc.page_content for doc in top_reranked_docs])

        # 5. Prepare the State for Graph-based Analysis
        state = {
            "requirement": current_clause,
            "evidence": dynamic_scope
        }
        
        # 6. Invoke the Graph App (LangChain Graph)
        out = graph_app.invoke(state)

        # Get the LAST entry added to the audit log by the finalized_and_log agent
        # If the log is empty (e.g., dropped), we provide a fallback
        audit_entries = out.get("audit_log", [])
        final_finding = audit_entries[-1] if audit_entries else {}
        
        # Collect the audit findings (e.g., "Missing Verification")
        results.append({
        "theme": item.get("theme"),
        "clause": current_clause,
        "source_ref": out.get("source_ref", "Not Explicitly Stated"),             # Dynamic from Gap Agent
        "status": out.get("gap_status", "Processing Error"),      # Dynamic from Router
        "gap_summary": out.get("gap_summary", "Review complete."), # Dynamic from Gap Agent
        "recommendation": out.get("gap_recommendation", ""),     # Dynamic from Gap Agent
        "risk_rating": out.get("rating", "N/A")                # Dynamic from Risk Agent
        })
        
    return render_template(
        
        "result.html",
        results=results,
        metadata=interpreted_data.get("metadata", {})
    )

def run():
    flask_app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    run()