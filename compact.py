"""Context compaction for long conversations."""

import json
from pathlib import Path
from typing import Optional


def compact_messages(messages: list[dict], provider,
                     max_messages: int = 20,
                     summary_model: Optional[str] = None) -> list[dict]:
    """Compact conversation when it grows too long.
    
    Keeps the system prompt + last N messages, summarizes the rest.
    """
    if len(messages) <= max_messages:
        return messages

    # Keep system + last N messages
    keep = max_messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    recent = messages[-keep:] if len(messages) - len(system_msgs) > keep else messages[len(system_msgs):]

    # Messages to summarize (everything between system and recent)
    to_summarize = messages[len(system_msgs):-keep]
    if not to_summarize:
        return messages

    # Build summary prompt
    summary_prompt = (
        "Summarize this conversation concisely in 2-3 sentences.\n"
        "Include key facts, decisions, and any ongoing tasks:\n\n"
    )
    for m in to_summarize:
        role = m["role"].upper()
        content = m["content"][:500]
        summary_prompt += f"{role}: {content}\n\n"

    try:
        summary = provider.chat([
            {"role": "system", "content": "You are a conversation summarizer."},
            {"role": "user", "content": summary_prompt},
        ])
    except Exception as e:
        summary = f"[Compact failed: {e}]"

    # Build compacted messages
    compacted = list(system_msgs)
    compacted.append({
        "role": "system",
        "content": f"## Previous conversation summary\n{summary}\n\nContinue the conversation."
    })
    compacted.extend(recent)

    return compacted
