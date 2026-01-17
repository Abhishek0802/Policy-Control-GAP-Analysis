# app/ui/flask_app.py
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from app.config import FAISS_INDEX_PATH, SCOPE
from app.state import AppState
from app.graph.flow import app as graph_app
from app.rag.retriever import load_faiss_retriever
from app.rag.retriever import retrieve_evidence
from app.rag.ingest import build_faiss_from_folder  
from app.agents.delta_detector import delta_detector

from app.rag.temp_ingest import build_temp_faiss
from app.agents.new_doc_interpreter import interpret_new_document

from langchain_openai import ChatOpenAI
from app.config import CHAT_MODEL

import logging
logging.basicConfig(level=logging.INFO)



UPLOAD_FOLDER = "data/internal_policies"
ALLOWED_EXTENSIONS = {"pdf", "txt"}

flask_app = Flask(__name__)
flask_app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
flask_app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@flask_app.get("/")
def index():
    return render_template("index.html")


@flask_app.post("/upload")
def upload():
    """
    Upload INTERNAL company policies and rebuild persistent FAISS index.
    """
    os.makedirs(flask_app.config["UPLOAD_FOLDER"], exist_ok=True)

    files = request.files.getlist("files")
    saved = 0

    for f in files:
        if f and allowed_file(f.filename):
            name = secure_filename(f.filename)
            path = os.path.join(flask_app.config["UPLOAD_FOLDER"], name)
            f.save(path)
            saved += 1

    # Rebuild index from uploaded docs (simple but works for demo)
    build_faiss_from_folder(folder=UPLOAD_FOLDER, save_path=FAISS_INDEX_PATH)

    return render_template("result.html", mode="upload", saved=saved)


@flask_app.post("/analyze")
def analyze():
    uploaded_file = request.files["file"]
    pdf_path = os.path.join("data/temp", uploaded_file.filename)
    os.makedirs("data/temp", exist_ok=True)
    uploaded_file.save(pdf_path)

    temp_faiss = build_temp_faiss(pdf_path)

    text = "\n".join(
        d.page_content for d in temp_faiss.docstore._dict.values()
    )

    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0.2)
    interpreted = interpret_new_document(llm, text)

    # Delta detection
    delta_filtered = []
    for item in interpreted:
        decision = delta_detector(item["requirement"])
        if decision in ["NOT_COVERED", "STRONGER"]:
            delta_filtered.append(item)

    if not delta_filtered:
        return render_template(
            "result.html",
            message="No new or changed requirements detected.",
            results=[]
        )

    AUTO_THRESHOLD = 0.8
    auto_approved = []
    needs_review = []

    for item in delta_filtered:
        if item["confidence"] >= AUTO_THRESHOLD:
            auto_approved.append(item)
        else:
            needs_review.append(item)

    if not needs_review:
        results = []
        for item in auto_approved:
            req = item["requirement"]
            evidence = retrieve_evidence(req)

            state = AppState(
                requirement=req,
                evidence=evidence,
                scope=SCOPE,
                approved_requirements=[req],
                human_review_done=True
            )

            out = graph_app.invoke(state)
            audit_log = out.get("audit_log", [])
            if audit_log:
                results.append(audit_log[-1])

        return render_template(
            "result.html",
            results=results
        )

    return render_template(
        "review.html",
        interpreted=needs_review,
        auto_approved=auto_approved
    )



@flask_app.post("/review/submit")
def submit_review():
    approved = request.form.getlist("approved_requirements")

    results = []

    for req in approved:
        evidence = retrieve_evidence(req)

        state = AppState(
            requirement=req,
            evidence=evidence,
            scope=SCOPE,
            approved_requirements=[req],
            human_review_done=True
        )

        out = graph_app.invoke(state)
        audit_log = out.get("audit_log", [])
        if audit_log:
            results.append(audit_log[-1])

    return render_template(
        "result.html",
        mode="compare",
        results=results
    )


def run():
    flask_app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    run()
