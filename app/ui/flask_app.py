# app/ui/flask_app.py
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from app.config import FAISS_INDEX_PATH, SCOPE
from app.state import AppState
from app.graph.flow import app as graph_app
from app.rag.retriever import load_faiss_retriever, retrieve_evidence
from app.rag.ingest import build_faiss_from_folder  # you will add this if not present

import logging
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "data/docs"
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
    Upload docs into data/docs and rebuild FAISS index.
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
    """
    Enter requirement -> retrieve evidence -> run LangGraph -> show decision + risk.
    """
    requirement = request.form.get("requirement", "").strip()
    if not requirement:
        return render_template("result.html", error="Requirement cannot be empty.")

    retriever = load_faiss_retriever()
    flask_app.logger.info("DEBUG requirement type=%s val=%s", type(requirement), requirement[:80])
    evidence = retrieve_evidence(str(requirement))

    state = AppState(
        requirement=requirement,
        evidence=evidence,
        scope=SCOPE,
    )

    out = graph_app.invoke(state)

    # Last audit entry (your flow logs to audit_log)
    audit_log = out.get("audit_log", []) if isinstance(out, dict) else getattr(out, "audit_log", [])
    last = audit_log[-1] if audit_log else {}

    return render_template(
        "result.html",
        mode="analyze",
        requirement=requirement,
        evidence=evidence,
        output=last,
    )


def run():
    flask_app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    run()
