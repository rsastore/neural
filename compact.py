"""Smart context pruning — token-aware, relevance-based pruning."""
import re

def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4

def prune_messages(messages: list[dict], max_tokens: int = 4096) -> list[dict]:
    """Remove messages when over token limit, keeping system + latest turns."""
    total = sum(estimate_tokens(m.get("content", "")) for m in messages)
    if total <= max_tokens:
        return messages
    
    # Keep system message
    kept = [m for m in messages if m["role"] == "system"]
    # Keep user/tool messages from the end
    others = [m for m in messages if m["role"] != "system"]
    
    remaining = max_tokens - sum(estimate_tokens(m.get("content","")) for m in kept)
    
    for m in reversed(others):
        t = estimate_tokens(m.get("content", ""))
        if t <= remaining:
            kept.insert(1, m)  # Insert after system, maintaining order
            remaining -= t
    
    return kept

# ── Prompt Templates per Model ──

MODEL_TEMPLATES = {
    "qwen": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "tool": "<|im_start|>tool\n{content}<|im_end|>\n",
        "stop": "<|im_end|>",
    },
    "llama": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>\n",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>\n",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n",
        "tool": "<|start_header_id|>tool<|end_header_id|>\n\n{content}<|eot_id|>\n",
        "stop": "<|eot_id|>",
    },
    "default": {
        "system": "{content}\n",
        "user": "User: {content}\n",
        "assistant": "Assistant: {content}\n",
        "tool": "Tool: {content}\n",
        "stop": "\n",
    },
}

def detect_template(model_name: str) -> dict:
    """Auto-detect prompt template based on model name."""
    name = model_name.lower()
    if "qwen" in name: return MODEL_TEMPLATES["qwen"]
    if "llama" in name or "meta" in name: return MODEL_TEMPLATES["llama"]
    return MODEL_TEMPLATES["default"]

def apply_template(messages: list[dict], model_name: str) -> str:
    """Apply optimal prompt template for the model."""
    tmpl = detect_template(model_name)
    result = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in tmpl:
            result += tmpl[role].format(content=content)
    return result + tmpl["stop"]
def compact_messages(messages: list[dict], provider, max_messages: int = 20, summary_model=None) -> list[dict]:
    """Compact conversation when it grows too long."""
    if len(messages) <= max_messages:
        return messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    recent = messages[-max_messages:] if len(messages) - len(system_msgs) > max_messages else messages[len(system_msgs):]
    to_summarize = messages[len(system_msgs):-max_messages]
    if not to_summarize:
        return messages
    summary_prompt = "Summarize this conversation concisely in 2-3 sentences.\nInclude key facts, decisions, and any ongoing tasks:\n\n"
    for m in to_summarize:
        summary_prompt += f"{m['role'].upper()}: {m['content'][:500]}\n\n"
    try:
        summary = provider.chat([
            {"role": "system", "content": "You are a conversation summarizer."},
            {"role": "user", "content": summary_prompt},
        ])
    except Exception as e:
        summary = f"[Compact failed: {e}]"
    compacted = list(system_msgs)
    compacted.append({"role": "system", "content": f"## Previous conversation summary\n{summary}\n\nContinue the conversation."})
    compacted.extend(recent)
    return compacted
