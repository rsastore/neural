import json, re, os as os_mod
from pathlib import Path
from typing import Callable

from tools.builtin import get_tool, tool_descriptions, list_tools, BUILTIN_TOOLS
from knowledge import search_knowledge, learn_from_interaction, knowledge_summary
from context import build_context_block, persona_instruction
from grammar import parse_tool_call as grammar_parse
from memory import working_context, episodic_context, preference_context, init_working

# ── System Prompt Builder ─────────────────────────────────────

def build_system_prompt(custom_prompt: str | None = None, persona: str = 'default') -> str:
    tools_desc = tool_descriptions()

    base = f"""You are Neural (RSA Agentic), an autonomous AI agent.

You MUST use JSON to call tools. Example:
```json
{{"tool": "exec_shell", "args": {{"cmd": "df -h"}}}}
```
```json
{{"tool": "read_file", "args": {{"path": "agent.py"}}}}
```
```json
{{"tool": "grep_files", "args": {{"pattern": "TODO", "path": "."}}}}
```
```json
{{"tool": "write_file", "args": {{"path": "file.txt", "content": "hello"}}}}
```

ALL tools available:
{tools_desc}

Rules:
1. Call ONE tool at a time. Read the result before next step.
2. When task is complete, respond in natural language.
3. Never delete/overwrite files without asking.
"""
    try:
        tctx = build_context_block()
        if tctx:
            base += f"\n\n{tctx}"
    except Exception:
        pass
    # Knowledge is injected per-user-message in run_stream()
    # Add persona instruction
    try:
        pi = persona_instruction(persona)
        if pi:
            base += f"\n\n## Mode: {persona}\n{pi}"
    except Exception:
        pass
    if custom_prompt:
        base += f"\n\n## Additional Instructions\n{custom_prompt}"
    return base


# ── Agent Session ─────────────────────────────────────────────

class AgentSession:
    """Maintains conversation state and runs agent loop."""

    def __init__(self, provider, config: dict):
        self.provider = provider
        self.config = config
        self.max_iters = config.get("max_tool_iters", 15)
        self.session_id = os_mod.urandom(4).hex()
        self.persona = "default"
        self._messages: list[dict] = []
        self.tool_callbacks: list[Callable] = []
        self.total_tokens = {"input": 0, "output": 0}
        self._pending_approved = True

    @property
    def messages(self) -> list[dict]:
        return self._messages

    def reset(self):
        self._messages = []
        self.session_id = os_mod.urandom(4).hex()
        self.persona = "default"
        self._pending_approved = True

    def _build_sys_prompt(self) -> str:
        import os as _os
        sys_file = self.config.get("system_prompt_file", "system.md")
        sys_path = Path(_os.path.expanduser("~/rsa-agentic")) / sys_file
        custom = sys_path.read_text() if sys_path.exists() else None
        return build_system_prompt(custom, self.persona)

    def _inject_knowledge(self, user_input: str) -> str:
        try:
            kctx = search_knowledge(user_input)
            if kctx:
                return f"{user_input}\n\n---\n{kctx}"
        except Exception: pass
        return user_input

    def _extract_tool_call(self, text: str) -> dict | None:
        """Parse tool call with grammar-guided JSON + auto-fix for local LLMs."""
        return grammar_parse(text)

    def run(self, user_input: str) -> str:
        """Process user input through agent loop. Returns final answer."""
        if not self._messages:
            init_working()
            self._messages.append({"role": "system", "content": self._build_sys_prompt()})
        enriched = self._inject_knowledge(user_input)
        self._messages.append({"role": "user", "content": enriched})


        for step in range(self.max_iters):
            # Get model response
            raw = self.provider.chat(self._messages)

            # Check for tool call
            call = self._extract_tool_call(raw)

            if call is None:
                # No tool call — this is the final answer
                self._messages.append({"role": "assistant", "content": raw})
                return raw

            # Execute tool
            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}

            tool = get_tool(tool_name)
            if tool is None:
                output = f"Error: Unknown tool '{tool_name}'. Available: {', '.join(list_tools())}"
            else:
                output = tool(**tool_args)

            # Notify callbacks (for TUI to show tool calls)
            for cb in self.tool_callbacks:
                cb(tool_name, tool_args, output)

            # Add to conversation - format depends on provider
            is_openai = type(self.provider).__name__ == "OpenAIProvider"
            if is_openai:
                # Structured format for OpenAI-compatible APIs (DeepSeek, OpenAI, Groq, etc)
                import json as _json
                tool_call_id = f"call_{self.session_id}_{step}"
                self._messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": _json.dumps(tool_args)}
                    }]
                })
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(output)[:3000],
                })
            else:
                # Text format for local models (Ollama)
                self._messages.append({"role": "assistant", "content": raw})
                self._messages.append({
                    "role": "tool",
                    "content": f"[{tool_name}] Result:\n{output[:3000]}",
                })

        self._messages.append({
            "role": "assistant",
            "content": "Max iterations reached. Task may be incomplete.",
        })
        return "Max iterations reached."

    def run_stream(self, user_input: str):
        sys_prompt = self._build_sys_prompt()
        if not self._messages:
            self._messages.append({"role": "system", "content": sys_prompt})
        else:
            self._messages[0] = {"role": "system", "content": sys_prompt}
        
        enriched = self._inject_knowledge(user_input)
        self._messages.append({"role": "user", "content": enriched})
        
        need_approval = self.config.get("need_approval", ["exec_shell"])
        yield {"type": "status", "content": "⏳ Loading model..."}
        for step in range(self.max_iters):
            if step > 0:
                yield {"type": "status", "content": f"↻ Agent step {step+1}/{self.max_iters}..."}
            self._pending_approved = True
            collected = []
            try:
                stream_count = 0
                for token in self.provider.chat_stream(self._messages):
                    stream_count += 1
                    collected.append(token)
                    yield {"type": "token", "content": token}
                if stream_count == 0:
                    # Stream yielded nothing, try non-streaming
                    raw = self.provider.chat(self._messages)
                    if raw:
                        collected = [raw]
                        yield {"type": "token", "content": raw}
            except Exception as e:
                try:
                    with open("/tmp/rsa_stream_error.log", "a") as _f:
                        _f.write(f"{type(e).__name__}: {e}\n")
                except: pass
                yield {"type": "status", "content": f"⚠️ Stream error, retrying..."}
                raw = self.provider.chat(self._messages)
                if raw:
                    collected = [raw]
                    yield {"type": "token", "content": raw}
            raw = "".join(collected)
            # Track tokens
            try:
                t = getattr(self.provider, "last_tokens", {})
                if t:
                    self.total_tokens["input"] += t.get("input", 0)
                    self.total_tokens["output"] += t.get("output", 0)
            except Exception:
                pass
            call = self._extract_tool_call(raw)
            if call:
                yield {"type": "status", "content": f"🛠 Running {call['tool']}()..."}
            if call is None:
                self._messages.append({"role": "assistant", "content": raw})
                # Auto-learn from this interaction
                try:
                    for msg in self._messages:
                        if msg["role"] == "user":
                            learn_from_interaction(msg["content"], raw, [])
                            break
                except Exception:
                    pass
                yield {"type": "final", "content": raw}
                return
            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            yield {"type": "tool_call", "tool": tool_name, "args": tool_args}
            if tool_name in need_approval:
                self._pending_approved = False  # Reset
                yield {"type": "approval_needed", "tool": tool_name, "args": tool_args}
                # TUI sets self._pending_approved before next iteration
                if not self._pending_approved:
                    output = "Cancelled by user"
                    yield {"type": "tool_result", "content": output}
                    self._messages.append({"role": "assistant", "content": raw})
                    is_openai = type(self.provider).__name__ == "OpenAIProvider"
                    if is_openai:
                        import json as _json
                        tc_id = f"call_{self.session_id}_{step}"
                        self._messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": tool_name, "arguments": _json.dumps(tool_args)}}]})
                        self._messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)[:3000]})
                    else:
                        self._messages.append({"role": "tool", "content": f"[{tool_name}] {output}"})
                    yield {"type": "token", "content": "\n[⛔ Cancelled]\n"}
                    continue
            tool = get_tool(tool_name)
            if tool is None:
                output = f"Unknown tool '{tool_name}'"
            else:
                output = tool(**tool_args)
            for cb in self.tool_callbacks:
                cb(tool_name, tool_args, output)
            yield {"type": "tool_result", "content": output[:500]}
            self._messages.append({"role": "assistant", "content": raw})
            is_openai = type(self.provider).__name__ == "OpenAIProvider"
            if is_openai:
                import json as _json
                tc_id = f"call_{self.session_id}_{step}"
                self._messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": tool_name, "arguments": _json.dumps(tool_args)}}]})
                self._messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)[:3000]})
            else:
                self._messages.append({"role": "tool", "content": f"[{tool_name}] Result:\n{output[:3000]}"})
        yield {"type": "final", "content": "Max iterations reached."}


class SubAgent:
    """A lightweight sub-agent for parallel task execution."""

    def __init__(self, provider, config: dict, task: str, parent_messages: list[dict]):
        self.provider = provider
        self.config = config
        self.task = task
        self.parent_messages = parent_messages

    def run(self) -> dict:
        """Execute sub-task and return result."""
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": self.task},
        ]
        try:
            result = ""
            for _ in range(8):
                raw = self.provider.chat(messages)
                call = self._extract_tool_call(raw)
                if not call:
                    result = raw
                    break
                tool = get_tool(call.get("tool", ""))
                if tool:
                    args = call.get("args", {})
                    if not isinstance(args, dict):
                        args = {}
                    output = tool(**args)
                else:
                    output = "Unknown tool"
                messages.append({"role": "assistant", "content": raw})
                is_openai = type(self.provider).__name__ == "OpenAIProvider"
                if is_openai:
                    import json as _json
                    tc_id = f"call_sub_{step}"
                    messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc_id, "type": "function", "function": {"name": tool_name, "arguments": _json.dumps(tool_args)}}]})
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": str(output)[:2000]})
                else:
                    messages.append({"role": "tool", "content": output[:2000]})
            return {"task": self.task, "result": result or "No result"}
        except Exception as e:
            return {"task": self.task, "result": f"Error: {e}"}

    @staticmethod
    def _extract_tool_call(text: str) -> dict | None:
        # Try ```json ... ``` block first
        m = re.search(r'```(?:json)?\s*\n(\{.*?\})\n\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Try brace-counting approach
        idx = text.find('{"tool"')
        if idx == -1:
            idx = text.find('{"tool"')
        if idx == -1:
            return None
        depth = 0
        for i in range(idx, len(text)):
            ch = text[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[idx:i+1])
                        if "tool" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    return None
        return None
