import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from app.config import FAISS_INDEX_PATH, SCOPE, CHAT_MODEL
from app.state import AppState
from app.graph.flow import app as graph_app

from app.rag.retriever import retrieve_controls
from app.rag.ingest import build_faiss_from_folder
from app.rag.temp_ingest import build_temp_faiss

from app.agents.new_doc_interpreter import interpret_new_document

from langchain_openai import ChatOpenAI

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
    Upload INTERNAL reference policies and rebuild FAISS index.
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

    build_faiss_from_folder(folder=UPLOAD_FOLDER, save_path=FAISS_INDEX_PATH)

    return render_template("result.html", mode="upload", saved=saved)


@flask_app.post("/analyze")
def analyze():
    uploaded_file = request.files["file"]
    pdf_path = os.path.join("data/temp", uploaded_file.filename)
    os.makedirs("data/temp", exist_ok=True)
    uploaded_file.save(pdf_path)


    temp_faiss = build_temp_faiss(pdf_path)
    policy_text = "\n".join(
    d.page_content for d in temp_faiss.docstore._dict.values()
    )


    llm = ChatOpenAI(model_name=CHAT_MODEL, temperature=0)


    interpreted = interpret_new_document(llm, policy_text)


    # Persist interpreter output for human review
    flask_app.config["LAST_INTERPRETED"] = interpreted


    return render_template(
    "interpreter_review.html",
    clauses=interpreted
    )

@flask_app.post("/review/submit")
def submit_review():
    approved = set(request.form.getlist("approved_clauses"))
    interpreted = flask_app.config.get("LAST_INTERPRETED", [])

    results = []

    for item in interpreted:
        if item["statement"] not in approved:
            continue

        state = AppState(
            requirement=item["statement"],
            evidence="; ".join(item.get("missing_elements", [])),
            scope=SCOPE
        )

        out = graph_app.invoke(state)
        results.extend(out.get("audit_log", []))

    return render_template(
        "result.html",
        results=results
    )

def run():
    flask_app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    run()