"""Vector Database with real embeddings via Ollama (nomic-embed-text)."""
import os, json, requests
from pathlib import Path

CHROMA_DIR = Path(os.path.expanduser("~/rsa-agentic/vectordb"))
OLLAMA_HOST = "http://localhost:11434"

def get_embedding(text: str) -> list[float]:
    r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={
        "model": "nomic-embed-text", "prompt": text[:1000]
    }, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]

def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a)**0.5
    nb = sum(x*x for x in b)**0.5
    return dot/(na*nb) if na and nb else 0

def get_collection(name="knowledge"):
    path = CHROMA_DIR / f"{name}.json"
    if path.exists(): return json.loads(path.read_text())
    return {"name": name, "docs": [], "vectors": [], "metadatas": []}

def save_collection(col, name="knowledge"):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (CHROMA_DIR / f"{name}.json").write_text(json.dumps(col, default=str))

def add_documents(docs, metadatas=None, collection="knowledge"):
    col = get_collection(collection)
    if metadatas is None: metadatas = [{} for _ in docs]
    for i, doc in enumerate(docs):
        col["docs"].append(doc)
        col["vectors"].append(get_embedding(doc))
        col["metadatas"].append(metadatas[i] if i < len(metadatas) else {})
    save_collection(col, collection)
    return len(docs)

def search(query, n=5, collection="knowledge"):
    col = get_collection(collection)
    if not col["vectors"]: return []
    qv = get_embedding(query)
    scores = [(cosine_similarity(qv, col["vectors"][i]), col["docs"][i], col["metadatas"][i]) for i in range(len(col["vectors"]))]
    scores.sort(key=lambda x: -x[0])
    return scores[:n]

def add_from_knowledge(collection="knowledge"):
    from knowledge import get_facts, get_skills
    facts = get_facts(); skills = get_skills()
    docs = [f"{f['topic']}: {f['content']}" for f in facts]
    docs += [f"{s['name']}: {s['pattern']}" for s in skills]
    return add_documents(docs, [{"type":"fact"}]*len(facts)+[{"type":"skill"}]*len(skills), collection) if docs else 0
