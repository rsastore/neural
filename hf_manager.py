"""Model Manager — search, download, use models from HuggingFace."""
import os, subprocess, json, urllib.request, re
from pathlib import Path

MODELS_DIR = Path(os.path.expanduser("~/neural/models_data"))

HF_API = "https://huggingface.co/api/models"

def list_installed():
    """List models available locally (via Ollama or direct GGUF)."""
    models = []
    # Check Ollama
    try:
        r = subprocess.run(["ollama","list"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().split("\n")[1:]:
            if line.strip():
                parts = line.split()
                if parts:
                    models.append({"name": parts[0], "size": parts[2] if len(parts)>2 else "?","backend":"ollama"})
    except: pass
    # Check local GGUF files
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*.gguf"):
            size = f.stat().st_size / 1024 / 1024
            models.append({"name": f.stem, "size": f"{size:.0f}MB", "backend":"gguf"})
    return models

def search_hf(query, limit=10):
    """Search HuggingFace for GGUF models."""
    import urllib.parse
    url = f"{HF_API}?search={urllib.parse.quote(query)}&sort=downloads&direction=-1&limit={limit}"
    # Also filter by GGUF tag
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Neural"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        results = []
        for m in data:
            results.append({
                "id": m.get("modelId", m.get("id","")),
                "downloads": m.get("downloads",0),
                "likes": m.get("likes",0),
                "tags": [t for t in m.get("tags",[]) if t],
            })
        # Filter for GGUF
        gguf_results = [r for r in results if "gguf" in str(r["tags"]).lower() or "gguf" in r["id"].lower()]
        return gguf_results[:limit] or results[:limit]
    except Exception as e:
        return [{"error": str(e)}]

def pull_model(model_id):
    """Download a model and set it up. Returns status messages as list."""
    msgs = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Determine filename from model ID
    # For now, try Ollama pull first
    short = model_id.split("/")[-1].replace("-Instruct-GGUF","").replace("-GGUF","")
    try:
        msgs.append(f"Pulling {short} via Ollama...")
        r = subprocess.run(["ollama","pull",short], capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            msgs.append(f" Done: {short}")
            return msgs
    except: pass
    # Fallback: try direct GGUF download from HuggingFace
    # Find the actual GGUF file URL
    # This is simplified - in real impl would parse HF API for file list
    msgs.append(f"Try: ollama pull {short}")
    msgs.append("Or download GGUF manually from huggingface.co/" + model_id)
    return msgs

def use_model(model_name):
    """Switch Neural to use a specific model."""
    import tomllib
    cfg_path = os.path.expanduser("~/neural/config.toml")
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    if "model" not in cfg:
        cfg["model"] = {}
    cfg["model"]["model_name"] = model_name
    import tomli_w
    try:
        with open(cfg_path, "w") as f:
            tomli_w.dump(cfg, f)
        return f"Switched to {model_name}"
    except:
        return f"Failed to update config. Set model_name={model_name} in config.toml manually."