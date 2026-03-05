"""
Flask demo app aligned with the classroom notebook:
Gutenberg RAG + Intent Router + deterministic tool orchestration.

Run:
    flask --app hello_rag.py run --port 5001
"""

import json
import os
import re
from pathlib import Path

import chromadb
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request

session = requests.Session()
session.trust_env = False

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma:latest")
TOPK = int(os.getenv("TOPK", "5"))

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "rag_artifacts"
CHROMA_PATH = BASE_DIR / "chroma_db"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "gutenberg_demo")

MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"

manifest = {}
chunks_cache = []
books_list = []

if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
if CHUNKS_PATH.exists():
    chunks_cache = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
if manifest.get("docs"):
    books_list = [d.get("title", "") for d in manifest.get("docs", [])]

CHAT_MODEL = manifest.get("chat_model", CHAT_MODEL)
EMBED_MODEL = manifest.get("embed_model", EMBED_MODEL)
NUM_CHUNKS = int(manifest.get("chunk_count") or len(chunks_cache))

try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
    chroma_available = collection.count() > 0
except Exception:
    collection = None
    chroma_available = False


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s_-]+", "-", s).strip("-")
    return s or "doc"


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def ollama_embed(texts):
    if not texts:
        return []
    r = session.post(
        f"{OLLAMA_URL.rstrip('/')}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    embs = data.get("embeddings")
    if not embs:
        raise RuntimeError(f"Missing 'embeddings' in /api/embed response: {data}")
    return embs


def ollama_chat(messages, temperature=0.2, tools=None):
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    if tools:
        payload["tools"] = tools
    r = session.post(f"{OLLAMA_URL.rstrip('/')}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def retrieve(query: str, k: int = TOPK):
    if not (chroma_available and collection):
        return []

    q = np.asarray(ollama_embed([query])[0], dtype=np.float32)[None, :]
    q = l2_normalize(q)[0].tolist()

    res = collection.query(
        query_embeddings=[q],
        n_results=int(k),
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        hits.append(
            {
                "text": doc or "",
                "doc_id": meta.get("doc_id", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "distance": float(dist) if dist is not None else None,
                "cite": f"[{meta.get('doc_id', 'unknown')}#{meta.get('chunk_index', -1)}]",
            }
        )
    return hits


def tool_list_books():
    books = [
        {
            "title": d.get("title", ""),
            "doc_id": slugify(d.get("title", "")),
            "url": d.get("url", ""),
        }
        for d in (manifest.get("docs") or [])
    ]
    return {"books": books}


def tool_rag_retrieve(query, k=5):
    hits = retrieve(query, k=int(k))
    return {"query": query, "k": int(k), "hits": hits}


def tool_character_context(name, k=5):
    q = f"{name} character description actions relationships"
    return tool_rag_retrieve(q, k=int(k))


def tool_quote_search(query, topn=5):
    query = (query or "").strip()
    topn = int(topn)
    if not query:
        return {"query": query, "matches": []}

    matches = []
    ql = query.lower()

    for chunk in chunks_cache:
        text = chunk.get("text") or ""
        if ql not in text.lower():
            continue
        m = re.search(re.escape(query), text, flags=re.I)
        if not m:
            continue
        a, b = m.start(), m.end()
        start = max(0, a - 200)
        end = min(len(text), b + 200)
        matches.append(
            {
                "doc_id": chunk.get("doc_id"),
                "chunk_index": chunk.get("chunk_index"),
                "title": chunk.get("title"),
                "source": chunk.get("source"),
                "snippet": text[start:end],
                "cite": f"[{chunk.get('doc_id')}#{chunk.get('chunk_index')}]",
            }
        )
        if len(matches) >= topn:
            break

    return {"query": query, "topn": topn, "matches": matches}


def tool_book_stats(title):
    title = (title or "").strip()
    if not title:
        return {"error": "title is required"}

    target = slugify(title)
    docs = manifest.get("docs") or []

    doc = next((d for d in docs if slugify(d.get("title", "")) == target), None)
    if not doc:
        low = title.lower()
        doc = next((d for d in docs if low in (d.get("title", "").lower())), None)

    return {
        "title": title,
        "doc_id": slugify(doc["title"]) if doc else target,
        "in_manifest": bool(doc),
        "manifest_doc": doc,
        "chroma_count": int(collection.count()) if collection else 0,
        "artifact_chunk_count": manifest.get("chunk_count"),
        "books_indexed": manifest.get("books_indexed"),
    }


TOOL_IMPL = {
    "list_books": lambda **kw: tool_list_books(),
    "quote_search": lambda **kw: tool_quote_search(**kw),
    "character_context": lambda **kw: tool_character_context(**kw),
    "rag_retrieve": lambda **kw: tool_rag_retrieve(**kw),
    "book_stats": lambda **kw: tool_book_stats(**kw),
}


pseudo_TOOL_SCHEMAS = [
    {
        "name": "list_books",
        "description": "List available Gutenberg books in the local corpus.",
        "params": {},
    },
    {
        "name": "quote_search",
        "description": "Exact phrase search; returns snippets with citations.",
        "params": {"query": "string", "topn": "int"},
    },
    {
        "name": "character_context",
        "description": "Retrieve top-k semantic chunks about a character.",
        "params": {"name": "string", "k": "int"},
    },
    {
        "name": "rag_retrieve",
        "description": "Semantic retrieval; returns hits with citations.",
        "params": {"query": "string", "k": "int"},
    },
    {
        "name": "book_stats",
        "description": "Book metadata/stats from manifest.",
        "params": {"title": "string"},
    },
]

INTENTS = [
    "quote_search",
    "character_context",
    "compare",
    "rag_qa",
    "creative_scene",
    "stats",
]
TOOLS_INFO = "\n".join(
    [
        f"- {t['name']}: {t['description']} (params: {', '.join(t['params'].keys()) or 'none'})"
        for t in pseudo_TOOL_SCHEMAS
    ]
)

ROUTER_SYSTEM = (
    "You are an intent router for a local Gutenberg RAG system.\n"
    "Return ONLY a single JSON object. No markdown, no code fences, no prose.\n"
    "Use exactly this schema:\n"
    '{"intent": string, "confidence": number, "args": object, "rewrite": string}\n'
    f"Allowed intents: {json.dumps(INTENTS)}.\n\n"
    "Available tools (for planning only):\n"
    f"{TOOLS_INFO}\n\n"
    "Rules:\n"
    "- quote_search: user asks for exact line/quote/phrase.\n"
    "- character_context: user asks 'who is X' or character traits.\n"
    "- stats: metadata/count requests.\n"
    '- compare: compare two characters. Put both names in args as {"left": "...", "right": "...", "k": 5}.\n'
    '- creative_scene: scene/dialogue writing. Put args as {"k": 6} (or another k).\n'
    "- rag_qa: all other questions.\n"
    "Set confidence in [0,1]. Return args as {} if not needed.\n"
    "rewrite should be the cleaned, corrected re-formatted query.\n"
)


def extract_first_json_object(text):
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty router response")

    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Router did not return JSON. Raw: {s[:200]}")


def route_llm(q, temperature=0.0):
    books = ", ".join(books_list)
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": f"Query: {q}\nAvailable books: {books}"},
    ]
    out = ollama_chat(messages, temperature=temperature, tools=None)
    text = out.get("message", {}).get("content", "") or out.get("response", "")
    obj = extract_first_json_object(text)

    if obj.get("intent") not in INTENTS:
        raise ValueError(f"Invalid intent: {obj.get('intent')}")
    if not isinstance(obj.get("args", {}), dict):
        obj["args"] = {}
    obj.setdefault("rewrite", q)
    obj.setdefault("confidence", 0.5)
    return obj


def route_intent(q):
    try:
        return route_llm(q)
    except Exception as e:
        print(f"[router] failed ({type(e).__name__}: {e}). Defaulting to rag_qa.")
        return {"intent": "rag_qa", "confidence": 0.5, "args": {}, "rewrite": q}


def parse_compare_names_fallback(query):
    q = (query or "").strip()
    if not q:
        return None

    m = re.search(r"\b(.+?)\s+(vs\.?|versus)\s+(.+?)\b", q, flags=re.I)
    if m:
        return m.group(1).strip(), m.group(3).strip()

    m = re.search(r"\bcompare\s+(.+?)\s+and\s+(.+?)\b", q, flags=re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None


def run_tool(name, args):
    fn = TOOL_IMPL.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return {"error": f"Bad args for {name}: {str(e)}"}
    except Exception as e:
        return {"error": f"Tool {name} failed: {type(e).__name__}: {str(e)}"}


ANSWER_SYSTEM = (
    "You are a local Gutenberg assistant.\n"
    "You MUST answer using ONLY the TOOL OUTPUTS provided.\n"
    "If the tool outputs do not contain enough information, say you don't know based on the corpus.\n\n"
    "Grounding rules:\n"
    "1) Any direct quote must appear verbatim in tool outputs and include its [doc_id#chunk_index] citation.\n"
    "2) Any factual claim about plot/characters should be supported by citations from tool outputs.\n"
    "3) Creative writing is allowed ONLY if it does not introduce unsupported factual claims; any quotes must still be sourced from tool outputs.\n"
)


def format_tool_outputs(output_from_tool_run, max_chars_per_hit=700):
    lines = []
    for tr in output_from_tool_run:
        name = tr["tool"]
        args = tr.get("args") or {}
        out = tr.get("out") or {}

        lines.append(f"TOOL: {name}")
        lines.append(f"ARGS: {json.dumps(args, ensure_ascii=False)}")

        if "error" in out:
            lines.append(f"ERROR: {out['error']}")
            lines.append("---")
            continue

        if name == "quote_search":
            matches = out.get("matches") or []
            lines.append(f"MATCHES: {len(matches)}")
            for m in matches:
                lines.append(f"{m.get('cite')} {m.get('title')}")
                lines.append((m.get("snippet") or "").strip()[:max_chars_per_hit])
                lines.append("")
            lines.append("---")
            continue

        if name in ("rag_retrieve", "character_context"):
            hits = out.get("hits") or []
            lines.append(f"HITS: {len(hits)}")
            for h in hits:
                lines.append(f"{h.get('cite')} {h.get('title')}")
                lines.append((h.get("text") or "").strip()[:max_chars_per_hit])
                lines.append("")
            lines.append("---")
            continue

        lines.append(json.dumps(out, indent=2, ensure_ascii=False))
        lines.append("---")

    return "\n".join(lines)


def process_answer_from_tools(user_query, output_from_tool_run, temperature=0.2):
    evidence = format_tool_outputs(output_from_tool_run)
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {
            "role": "user",
            "content": f"USER QUERY:\n{user_query}\n\nTOOL OUTPUTS:\n{evidence}",
        },
    ]
    out = ollama_chat(messages, temperature=temperature)
    return out.get("message", {}).get("content", "") or out.get("response", "")


def dispatch(query, trace=True, temperature_router=0.0, temperature_answer=0.2):
    route = route_intent(query)
    intent = route.get("intent", "rag_qa")
    args = route.get("args") or {}
    rewritten_query = route.get("rewrite") or query

    output_from_tool_run = []

    if intent == "quote_search":
        tool_args = {"query": rewritten_query, "topn": int(args.get("topn", 5))}
        out = run_tool("quote_search", tool_args)
        output_from_tool_run.append(
            {"tool": "quote_search", "args": tool_args, "out": out}
        )

    elif intent == "character_context":
        name = args.get("name") or rewritten_query
        tool_args = {"name": name, "k": int(args.get("k", TOPK))}
        out = run_tool("character_context", tool_args)
        output_from_tool_run.append(
            {"tool": "character_context", "args": tool_args, "out": out}
        )

    elif intent == "stats":
        title = args.get("title") or rewritten_query
        tool_args = {"title": title}
        out = run_tool("book_stats", tool_args)
        output_from_tool_run.append(
            {"tool": "book_stats", "args": tool_args, "out": out}
        )

    elif intent == "compare":
        left = args.get("left")
        right = args.get("right")
        k = int(args.get("k", TOPK))

        if not left or not right:
            parsed = parse_compare_names_fallback(rewritten_query)
            if parsed:
                left, right = parsed

        if not left or not right:
            tool_args = {"query": rewritten_query, "k": k}
            out = run_tool("rag_retrieve", tool_args)
            output_from_tool_run.append(
                {"tool": "rag_retrieve", "args": tool_args, "out": out}
            )
        else:
            out_l = run_tool("character_context", {"name": left, "k": k})
            out_r = run_tool("character_context", {"name": right, "k": k})
            output_from_tool_run.append(
                {
                    "tool": "character_context",
                    "args": {"name": left, "k": k},
                    "out": out_l,
                }
            )
            output_from_tool_run.append(
                {
                    "tool": "character_context",
                    "args": {"name": right, "k": k},
                    "out": out_r,
                }
            )

    elif intent == "creative_scene":
        k = int(args.get("k", max(6, TOPK)))
        tool_args = {"query": rewritten_query, "k": k}
        out = run_tool("rag_retrieve", tool_args)
        output_from_tool_run.append(
            {"tool": "rag_retrieve", "args": tool_args, "out": out}
        )

    else:
        k = int(args.get("k", TOPK))
        tool_args = {"query": rewritten_query, "k": k}
        out = run_tool("rag_retrieve", tool_args)
        output_from_tool_run.append(
            {"tool": "rag_retrieve", "args": tool_args, "out": out}
        )

    answer = process_answer_from_tools(
        rewritten_query, output_from_tool_run, temperature=temperature_answer
    )

    return {
        "intent": intent,
        "route": route,
        "answer": answer,
        "tools_used": [
            {"tool": tr["tool"], "args": tr["args"]} for tr in output_from_tool_run
        ],
        "output_from_tool_run": output_from_tool_run if trace else None,
    }


def build_ui_hits_contexts(output_from_tool_run):
    hits = []
    contexts = []

    for tr in output_from_tool_run:
        tool = tr.get("tool")
        out = tr.get("out") or {}

        if tool in ("rag_retrieve", "character_context"):
            for hit in out.get("hits") or []:
                hits.append(hit)
                contexts.append(
                    {
                        "title": hit.get("title", ""),
                        "doc_id": hit.get("doc_id", "unknown"),
                        "chunk_index": hit.get("chunk_index", -1),
                        "cite": hit.get("cite", ""),
                        "text": hit.get("text", ""),
                        "span": str(hit.get("chunk_index", -1)),
                        "approx_words": len((hit.get("text") or "").split()),
                    }
                )

        if tool == "quote_search":
            for match in out.get("matches") or []:
                contexts.append(
                    {
                        "title": match.get("title", ""),
                        "doc_id": match.get("doc_id", "unknown"),
                        "chunk_index": match.get("chunk_index", -1),
                        "cite": match.get("cite", ""),
                        "text": match.get("snippet", ""),
                        "span": str(match.get("chunk_index", -1)),
                        "approx_words": len((match.get("snippet") or "").split()),
                    }
                )

    return hits, contexts


app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        num_chunks=NUM_CHUNKS,
        chat_model=CHAT_MODEL,
        embed_model=EMBED_MODEL,
        books_json=json.dumps(books_list),
    )


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        out = dispatch(query, trace=True)
        output_from_tool_run = out.get("output_from_tool_run") or []
        hits, contexts = build_ui_hits_contexts(output_from_tool_run)

        return jsonify(
            {
                "query": query,
                "intent": out.get("intent"),
                "confidence": float((out.get("route") or {}).get("confidence", 0.5)),
                "tools_used": [
                    t.get("tool", "") for t in (out.get("tools_used") or [])
                ],
                "hits": hits,
                "contexts": contexts,
                "answer": out.get("answer", ""),
            }
        )
    except Exception as e:
        print(f"Error in /ask: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        hits = retrieve(query, k=TOPK)
        contexts = [
            {
                "title": h.get("title", ""),
                "doc_id": h.get("doc_id", "unknown"),
                "chunk_index": h.get("chunk_index", -1),
                "cite": h.get("cite", ""),
                "text": h.get("text", ""),
                "span": str(h.get("chunk_index", -1)),
                "approx_words": len((h.get("text") or "").split()),
            }
            for h in hits
        ]
        return jsonify({"query": query, "hits": hits, "contexts": contexts})
    except Exception as e:
        print(f"Error in /search: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Gutenberg RAG + Intent Router Demo")
    print("=" * 70)
    print(f"Models: chat={CHAT_MODEL}, embed={EMBED_MODEL}")
    print(f"Chunks: {NUM_CHUNKS} | Chroma: {'✓' if chroma_available else '✗'}")
    print("Server: http://localhost:5001")
    print("=" * 70 + "\n")

    app.run(debug=True, port=5001)

