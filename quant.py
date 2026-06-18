"""Auto-quantization — detect RAM, recommend best model."""
import os, json

# Model sizes (approximate) per quantization level
MODEL_SIZES = {
    "1.5b": {"Q8": 1.8, "Q6": 1.5, "Q5": 1.3, "Q4": 1.1, "Q3": 0.9, "Q2": 0.7},
    "3b":   {"Q8": 3.5, "Q6": 2.8, "Q5": 2.5, "Q4": 2.1, "Q3": 1.7, "Q2": 1.3},
    "7b":   {"Q8": 7.5, "Q6": 6.0, "Q5": 5.3, "Q4": 4.5, "Q3": 3.6, "Q2": 2.7},
}

def get_available_ram() -> float:
    """Get available RAM in GB."""
    try:
        import subprocess
        r = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split("\n"):
            if line.startswith("Mem:"):
                parts = line.split()
                avail = int(parts[6])  # available in bytes
                return avail / 1024 / 1024 / 1024
    except: pass
    return 4.0  # fallback

def recommend_quantization(model_size: str = "1.5b") -> dict:
    """Recommend best quantization based on available RAM."""
    ram = get_available_ram()
    if model_size not in MODEL_SIZES:
        return {"error": f"Unknown model size: {model_size}", "ram_gb": ram}
    
    sizes = MODEL_SIZES[model_size]
    # Leave 1GB for system + agent
    usable_ram = ram - 1.0
    
    best_q = "Q2"
    best_size = 0
    
    for q, size_gb in sorted(sizes.items()):
        if size_gb <= usable_ram and size_gb > best_size:
            best_q = q
            best_size = size_gb
    
    return {
        "recommended": best_q,
        "ram_gb": round(ram, 1),
        "usable_gb": round(usable_ram, 1),
        "model_size_gb": best_size,
        "all_options": {q: s for q, s in sizes.items() if s <= usable_ram},
    }

def suggest_model() -> str:
    """Suggest best model for current hardware."""
    ram = get_available_ram()
    if ram >= 6: return "qwen2.5:3b (Q4)"
    if ram >= 4: return "qwen2.5:1.5b (Q8)"
    if ram >= 3: return "qwen2.5:1.5b (Q4)"
    return "qwen3:0.6b (Q4)"
