
"""Vector Database adapter using Chroma — proper RAG for RSA Agentic."""
import os, json
from pathlib import Path

CHROMA_DIR = Path(os.path.expanduser("~/rsa-agentic/vectordb"))

def _get_client():
    try:
        import chromadb
        return chromadb.PersistentClient(path=str(CHROMA_DIR))
    except ImportError:
        raise ImportError("Install: pip install chromadb")

def create_collection(name="knowledge"):
    client = _get_client()
    try:
        client.delete_collection(name)
    except: pass
    return client.create_collection(name=name)

def get_collection(name="knowledge"):
    client = _get_client()
    try:
        return client.get_collection(name=name)
    except:
        return create_collection(name)

def add_documents(docs, metadatas=None, ids=None, collection="knowledge"):
    """Add documents to vector DB. Each doc is a string."""
    col = get_collection(collection)
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(docs))]
    if metadatas is None:
        metadatas = [{} for _ in docs]
    col.add(documents=docs, metadatas=metadatas, ids=ids)
    return len(docs)

def search(query, n_results=5, collection="knowledge"):
    """Search for similar documents."""
    col = get_collection(collection)
    try:
        results = col.query(query_texts=[query], n_results=n_results)
        docs = results["documents"][0] if results["documents"] else []
        dists = results["distances"][0] if results["distances"] else []
        return list(zip(docs, dists))
    except Exception as e:
        return [(f"Search error: {e}", 0)]

def add_knowledge_from_jsonl(path, collection="knowledge"):
    """Import knowledge from JSONL file into vector DB."""
    import json
    docs = []
    metas = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            text = ""
            for m in data.get("messages", []):
                text += m.get("content", "") + " "
            if text.strip():
                docs.append(text[:1000])
                metas.append({"domain": data.get("domain",""), "source": path})
    if docs:
        n = add_documents(docs, metadatas=metas, collection=collection)
        return f"Added {n} docs to vector DB"
    return "No documents found"
