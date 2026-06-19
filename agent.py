import json, re, os as os_mod
from pathlib import Path
from tools.builtin import get_tool, list_tools, tool_descriptions
from models.providers import create_provider

# ── Context Builder ────────────────────────────────────────────────
def build_context_block() -> str:
    try:
        fp = os_mod.path.expanduser("~/rsa-agentic/memory/working.json")
        if os_mod.path.exists(fp):
            data = json.load(open(fp))
            if isinstance(data, dict) and data:
                return "\n## Current Context\n" + "\n".join(f"- {k}: {v}" for k, v in data.items()[:5])
    except: pass
    return ""

def build_system_prompt(custom_prompt: str | None = None, persona: str = 'default') -> str:
    tools_desc = tool_descriptions()
    base = f"""You are Neural (RSA Agentic), an autonomous AI agent.

You have access to tools. To call a tool, respond with JSON:
{json_examples(tools_desc)}

ALL tools:
{tools_desc}

Rules:
1. Call ONE tool at a time. Read result before next step.
2. When done, respond naturally.
3. Never delete/overwrite files without asking.
"""
    try:
        tctx = build_context_block()
        if tctx: base += tctx
    except: pass
    try:
        from knowledge import search_knowledge
        k = search_knowledge("system_prompt")
        if k: base += f"\n{k[:500]}"
    except: pass
    if custom_prompt:
        base += f"\n\n## Custom Instructions\n{custom_prompt}"
    return base

def json_examples(tools_desc: str) -> str:
    return """```json
{"tool": "exec_shell", "args": {"cmd": "df -h"}}
```
```json
{"tool": "read_file", "args": {"path": "file.txt"}}
```
```json
{"tool": "write_file", "args": {"path": "file.txt", "content": "hello"}}
```
```json
{"tool": "grep_files", "args": {"pattern": "TODO", "path": "."}}
```"""

def grammar_parse(text: str) -> dict | None:
    js = extract_json(text)
    if not js: return None
    try:
        d = json.loads(js)
        if 'tool' in d: return d
    except: pass
    try:
        d = json.loads(auto_fix(js))
        if 'tool' in d: return d
    except: pass
    tm = re.search(r'"tool"\s*:\s*"([^"]+)"', js)
    if tm:
        args = {}
        am = re.search(r'"args"\s*:\s*\{(.*?)\}', js, re.DOTALL)
        if am:
            try:
                raw = "{" + am.group(1) + "}"
                args = json.loads(auto_fix(raw))
            except:
                for kv in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', am.group(1)):
                    args[kv[0]] = kv[1]
        return {'tool': tm.group(1), 'args': args}
    return None

def extract_json(text: str) -> str | None:
    # Handle code blocks first
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if m: return m.group(1)
    # Find outermost JSON object with "tool" key by counting braces
    idx = text.find('"tool"')
    if idx >= 0:
        # Search backwards for opening brace
        start = text.rfind('{', 0, idx)
        # Search forwards for matching closing brace
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{': depth += 1
                elif text[i] == '}': depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return None

def auto_fix(s: str) -> str:
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    s = re.sub(r"'", '"', s)
    s = re.sub(r'True|False|None', lambda m: m.group(0).lower(), s)
    return s

# ── Agent Session ─────────────────────────────────────────────────
class AgentSession:
    def __init__(self, provider, config: dict):
        self.provider = provider
        self.config = config
        self.max_iters = config.get("max_tool_iters", 15)
        self.session_id = os_mod.urandom(4).hex()
        self.persona = "default"
        self._messages: list[dict] = []
        self.tool_callbacks: list = []
        self.total_tokens = {"input": 0, "output": 0}
        self._pending_approved = True

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def _build_sys_prompt(self) -> str:
        sys_file = self.config.get("system_prompt_file", "system.md")
        sys_path = Path(os_mod.path.expanduser("~/rsa-agentic")) / sys_file
        custom = sys_path.read_text() if sys_path.exists() else None
        return build_system_prompt(custom, self.persona)

    def _inject_knowledge(self, user_input: str) -> str:
        try:
            from knowledge import search_knowledge
            kctx = search_knowledge(user_input)
            if kctx: return f"{user_input}\n\nContext:\n{kctx[:500]}"
        except: pass
        return user_input

    def run(self, user_input: str) -> str:
        if not self._messages:
            init_working()
            self._messages.append({"role": "system", "content": self._build_sys_prompt()})
        enriched = self._inject_knowledge(user_input)
        self._messages.append({"role": "user", "content": enriched})
        is_openai = type(self.provider).__name__ == "OpenAIProvider"

        for step in range(self.max_iters):
            raw = self.provider.chat(self._messages)
            call = grammar_parse(raw)
            if call is None:
                self._messages.append({"role": "assistant", "content": raw})
                return raw

            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})
            if not isinstance(tool_args, dict): tool_args = {}

            tool = get_tool(tool_name)
            if tool is None:
                output = f"Error: Unknown tool '{tool_name}'. Available: {', '.join(list_tools())}"
            else:
                output = tool(**tool_args)

            for cb in self.tool_callbacks:
                cb(tool_name, tool_args, output)

            if is_openai:
                tc_id = f"call_{self.session_id}_{step}"
                self._messages.append(self.provider.format_assistant_with_tool(tool_name, tool_args, tc_id))
                self._messages.append(self.provider.format_tool_result(tool_name, output, tc_id))
            else:
                self._messages.append({"role": "assistant", "content": raw})
                self._messages.append({"role": "tool", "content": f"[{tool_name}] Result:\n{output[:3000]}"})

        self._messages.append({"role": "assistant", "content": "Max iterations reached."})
        return "Max iterations reached."

    def run_stream(self, user_input: str):
        sys_prompt = self._build_sys_prompt()
        if not self._messages:
            self._messages.append({"role": "system", "content": sys_prompt})
        enriched = self._inject_knowledge(user_input)
        self._messages.append({"role": "user", "content": enriched})
        is_openai = type(self.provider).__name__ == "OpenAIProvider"

        for step in range(self.max_iters):
            raw = ""
            for chunk in self.provider.chat_stream(self._messages):
                raw += chunk
                yield {"type": "token", "content": chunk}
            yield {"type": "status", "content": raw}

            if not raw.strip():
                yield {"type": "final", "content": "No response"}
                return

            call = grammar_parse(raw)
            if call is None:
                self._messages.append({"role": "assistant", "content": raw})
                yield {"type": "final", "content": raw}
                return

            tool_name = call.get("tool", "")
            tool_args = call.get("args", {})
            if not isinstance(tool_args, dict): tool_args = {}

            tool = get_tool(tool_name)
            if tool is None:
                output = f"Error: Unknown tool '{tool_name}'. Available: {', '.join(list_tools())}"
            else:
                output = tool(**tool_args)

            for cb in self.tool_callbacks:
                cb(tool_name, tool_args, output)
            yield {"type": "tool_call", "tool": tool_name, "args": tool_args, "result": output[:500]}

            if is_openai:
                tc_id = f"call_{self.session_id}_{step}"
                self._messages.append(self.provider.format_assistant_with_tool(tool_name, tool_args, tc_id))
                self._messages.append(self.provider.format_tool_result(tool_name, output, tc_id))
            else:
                self._messages.append({"role": "assistant", "content": raw})
                self._messages.append({"role": "tool", "content": f"[{tool_name}] Result:\n{output[:3000]}"})

        yield {"type": "final", "content": "Max iterations reached."}


# ── Sub-Agent (for tools that need sub-agents) ────────────────────
class SubAgent:
    def __init__(self, config: dict):
        from models.providers import create_provider
        self.provider = create_provider(config.get("model", {}))
        self.config = config

    def run(self, task: str, context: str = "") -> dict:
        try:
            from tools.builtin import list_tools, tool_descriptions as td
            messages = [{"role": "system", "content": f"You are a sub-agent. Task: {task}\\nTools:\\n{td()}\\nRespond with JSON tool calls or text."}]
            if context: messages.append({"role": "user", "content": context})
            is_openai = type(self.provider).__name__ == "OpenAIProvider"

            for step in range(5):
                raw = self.provider.chat(messages)
                call = grammar_parse(raw)
                if call is None:
                    messages.append({"role": "assistant", "content": raw})
                    return {"task": task, "result": raw or "No result"}
                tool_name = call.get("tool", "")
                tool_args = call.get("args", {})
                tool = get_tool(tool_name)
                output = tool(**tool_args) if tool else "Unknown tool"
                if is_openai:
                    tc_id = f"sub_call_{step}"
                    messages.append(self.provider.format_assistant_with_tool(tool_name, tool_args, tc_id))
                    messages.append(self.provider.format_tool_result(tool_name, output, tc_id))
                else:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "tool", "content": output[:2000]})
            return {"task": task, "result": "Max iters"}
        except Exception as e:
            return {"task": task, "error": str(e)}


def init_working():
    fp = os_mod.path.expanduser("~/rsa-agentic/memory/working.json")
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    if not os_mod.path.exists(fp):
        json.dump({"node": "root", "os": "linux", "arch": "aarch64", "started": __import__('datetime').datetime.now().isoformat()}, open(fp, "w"))
