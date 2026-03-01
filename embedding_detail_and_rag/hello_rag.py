"""
Flask RAG App — loads pre-built FAISS index + chunks from disk
and serves a search UI.

Run:
    python app.py

Prerequisites:
    1. Run the notebook first to generate rag_artifacts/
    2. pip install flask faiss-cpu numpy ollama
    3. Ollama must be running with the embedding model pulled
"""

import json, os, textwrap
from pathlib import Path

import numpy as np
import faiss
import ollama
from flask import Flask, render_template, request, jsonify

# ── Ollama client (bypass SSL/proxy issues) ──────────────────────────────
ollama_client = ollama.Client(host="http://localhost:11434", trust_env=False)

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "rag_artifacts"

# ── Load artifacts once at startup ───────────────────────────────────────
print("Loading artifacts …")

with open(ARTIFACTS_DIR / "config.json") as f:
    config = json.load(f)

EMBED_MODEL = config["EMBED_MODEL"]
WORDS_PER_CHUNK = config["WORDS_PER_CHUNK"]
TOPK = config["TOPK"]

with open(ARTIFACTS_DIR / "chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

emb = np.load(str(ARTIFACTS_DIR / "embeddings.npy"))
index = faiss.read_index(str(ARTIFACTS_DIR / "faiss_index.bin"))

print(
    f"  Loaded {len(chunks)} chunks, embeddings {emb.shape}, FAISS index {index.ntotal} vectors"
)
print(f"  Embed model: {EMBED_MODEL}")

# ── Lookup map for neighbor expansion ────────────────────────────────────
KEY_TO_IDX = {(c["source_path"], c["chunk_index"]): gi for gi, c in enumerate(chunks)}


# ── Core RAG helpers (same logic as notebook) ────────────────────────────
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def expand_with_neighbors_simple(I, D, chunks, neighbors=1, max_out=8):
    seen = set()
    ordered_idxs = []
    for gi in I[0]:
        c = chunks[int(gi)]
        doc = c["source_path"]
        ci = c["chunk_index"]
        for delta in range(-neighbors, neighbors + 1):
            key = (doc, ci + delta)
            j = KEY_TO_IDX.get(key)
            if j is None:
                continue
            if j not in seen:
                ordered_idxs.append(j)
                seen.add(j)
            if len(ordered_idxs) >= max_out:
                break
        if len(ordered_idxs) >= max_out:
            break

    ordered_idxs.sort(
        key=lambda j: (chunks[j]["source_path"], chunks[j]["chunk_index"])
    )

    merged = []
    span = None
    for gi in ordered_idxs:
        c = chunks[gi]
        if (
            span
            and c["source_path"] == span["source_path"]
            and c["chunk_index"] == span["end_ci"] + 1
        ):
            span["end_ci"] += 1
            span["idxs"].append(gi)
        else:
            if span:
                merged.append(span)
            span = {
                "title": c["title"],
                "source_path": c["source_path"],
                "start_ci": c["chunk_index"],
                "end_ci": c["chunk_index"],
                "idxs": [gi],
            }
    if span:
        merged.append(span)

    contexts = []
    for s in merged:
        text = "\n\n".join(chunks[i]["text"] for i in s["idxs"])
        contexts.append(
            {
                "title": s["title"],
                "source_path": s["source_path"],
                "start_chunk": s["start_ci"],
                "end_chunk": s["end_ci"],
                "text": text,
                "approx_words": sum(len(chunks[i]["text"].split()) for i in s["idxs"]),
                "span_count": len(s["idxs"]),
            }
        )
    return contexts


def search_windowed(query: str, topk: int = TOPK, max_out: int = 8):
    q_vec = np.asarray(
        ollama_client.embeddings(model=EMBED_MODEL, prompt=query)["embedding"],
        dtype="float32",
    )
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    D, I = index.search(q_vec.reshape(1, -1), topk)

    neighbors = 1 if WORDS_PER_CHUNK >= 260 else 2

    top1 = float(D[0][0])
    top2 = float(D[0][1]) if len(D[0]) > 1 else 0.0
    margin = top1 - top2
    init_topk = 1 if (top1 >= 0.35 and margin >= 0.05) else min(3, topk)

    D_init = np.array([D[0][:init_topk]])
    I_init = np.array([I[0][:init_topk]])

    contexts = expand_with_neighbors_simple(
        I_init, D_init, chunks, neighbors=neighbors, max_out=max_out
    )

    hits = []
    for score, idx_ in zip(D[0].tolist(), I[0].tolist()):
        m = chunks[idx_]
        hits.append(
            {
                "score": round(score, 3),
                "id": m["id"],
                "title": m["title"],
                "source_path": m["source_path"],
                "chunk_index": m["chunk_index"],
                "preview": (
                    (m["preview"][:300] + "…")
                    if len(m["preview"]) > 300
                    else m["preview"]
                ),
            }
        )
    return hits, contexts


# ── Flask app ────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html", embed_model=EMBED_MODEL, num_chunks=len(chunks)
    )


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        hits, contexts = search_windowed(query)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            "query": query,
            "hits": hits,
            "contexts": [
                {
                    "title": c["title"],
                    "source": Path(c["source_path"]).name,
                    "span": f"{c['start_chunk']}–{c['end_chunk']}",
                    "approx_words": c["approx_words"],
                    "span_count": c["span_count"],
                    "text": c["text"],
                }
                for c in contexts
            ],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)


# flask --app hello_flask.py run
