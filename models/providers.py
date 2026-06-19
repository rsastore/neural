import json, requests
from models.base import ModelProvider

class OllamaProvider(ModelProvider):
    def __init__(self, config: dict):
        self.host = config.get("host", "http://localhost:11434")
        self.model = config.get("model_name", "qwen2.5:1.5b")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 120)
        self.session = requests.Session()
        self._ensure_running()

    def _ensure_running(self):
        import subprocess, time
        try:
            r = self.session.get(f"{self.host}/api/tags", timeout=2)
            if r.status_code == 200: return
        except: pass
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
        except: pass

    @property
    def name(self) -> str: return f"Ollama/{self.model}"

    def chat(self, messages: list[dict], **kw) -> str:
        body = {"model": self.model, "messages": messages, "stream": False,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        r = self.session.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        if r.status_code != 200:
            return f"Error: Ollama returned {r.status_code}: {r.text}"
        return r.json()["message"]["content"]

    def chat_stream(self, messages: list[dict], **kw):
        body = {"model": self.model, "messages": messages, "stream": True,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        r = self.session.post(f"{self.host}/api/chat", json=body, timeout=self.timeout, stream=True)
        if r.status_code != 200:
            yield f"Error: Ollama returned {r.status_code}: {r.text}"
            return
        for line in r.iter_lines(decode_unicode=True):
            if line:
                try:
                    d = json.loads(line)
                    if d.get("done"):
                        return
                    yield d.get("message", {}).get("content", "")
                except: pass


class OpenAIProvider(ModelProvider):
    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.model = config.get("model", "gpt-4o")
        self.temperature = config.get("temperature", 0.3)

    @property
    def name(self) -> str: return f"OpenAI/{self.model}"

    def chat(self, messages: list[dict], **kw) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return "Error: openai not installed. Run: pip install openai"
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            r = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            r = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
            )
        return r.choices[0].message.content or ""

    def chat_stream(self, messages: list[dict], **kw):
        try:
            from openai import OpenAI
        except ImportError:
            yield "Error: openai not installed. Run: pip install openai"
            return
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            r = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                stream=True, extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception:
            r = client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
                stream=True,
            )
        for chunk in r:
            delta = chunk.choices[0].delta.content or ""
            if delta: yield delta


    def format_assistant_with_tool(self, tool_name: str, tool_args: dict, tool_call_id: str = None) -> dict:
        """Format assistant message with structured tool call."""
        if not tool_call_id:
            import uuid
            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(tool_args)}
            }]
        }

    def format_tool_result(self, tool_name: str, output: str, tool_call_id: str = None) -> dict:
        """Format tool result message."""
        if not tool_call_id:
            tool_call_id = "call_unknown"
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(output)[:30000]
        }

    def build_tools_schema(self, tools_list: list) -> list[dict]:
        """Build OpenAI-compatible tools schema from tool definitions."""
        schemas = []
        for t in tools_list:
            props = {}
            required = []
            for pname, pdesc in t.params.items():
                props[pname] = {"type": "string", "description": pdesc}
                required.append(pname)
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required
                    }
                }
            })
        return schemas


class AnthropicProvider(ModelProvider):
    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self.temperature = config.get("temperature", 0.3)

    @property
    def name(self) -> str: return f"Anthropic/{self.model}"

    def chat(self, messages: list[dict], **kw) -> str:
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": self.model, "messages": messages,
                "max_tokens": 4096, "temperature": self.temperature}
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=60)
            if r.status_code == 200:
                return r.json()["content"][0]["text"]
            return f"Error: {r.status_code} {r.text}"
        except Exception as e:
            return f"Error: {e}"

    def chat_stream(self, messages: list[dict], **kw):
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": self.model, "messages": messages,
                "max_tokens": 4096, "temperature": self.temperature, "stream": True}
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=60, stream=True)
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    d = json.loads(line[6:])
                    if d["type"] == "content_block_delta":
                        yield d["delta"].get("text", "")
        except Exception as e:
            yield f"Error: {e}"


class GoogleProvider(ModelProvider):
    def __init__(self, config: dict):
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gemini-2.0-flash")

    @property
    def name(self) -> str: return f"Google/{self.model}"

    def chat(self, messages: list[dict], **kw) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        parts = [{"text": m["content"]} for m in messages if "content" in m and m["content"]]
        body = {"contents": [{"parts": parts}]}
        try:
            r = requests.post(url, json=body, timeout=30)
            if r.status_code == 200:
                return r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return f"Error: {r.status_code}"
        except Exception as e:
            return f"Error: {e}"

    def chat_stream(self, messages: list[dict], **kw):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?key={self.api_key}&alt=sse"
        parts = [{"text": m["content"]} for m in messages if "content" in m and m["content"]]
        body = {"contents": [{"parts": parts}]}
        try:
            r = requests.post(url, json=body, timeout=30, stream=True)
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    try:
                        d = json.loads(line[6:])
                        text = d.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text: yield text
                    except: pass
        except Exception as e:
            yield f"Error: {e}"


def create_provider(config: dict) -> ModelProvider:
    prov_name = config.get("provider", "ollama")
    prov_cfg = config.get(prov_name, {})
    m = config.get("model_name", prov_cfg.get("model", ""))
    prov_cfg["model_name"] = config.get("model_name", m)
    prov_cfg["model"] = prov_cfg.get("model", config.get("model_name") or "gpt-4o")
    prov_cfg["temperature"] = config.get("temperature", 0.3)
    prov_cfg["max_tokens"] = config.get("max_tokens", 4096)

    if prov_name == "ollama":
        return OllamaProvider(prov_cfg)
    elif prov_name == "openai":
        return OpenAIProvider(prov_cfg)
    elif prov_name == "anthropic":
        return AnthropicProvider(prov_cfg)
    elif prov_name == "google":
        return GoogleProvider(prov_cfg)
    else:
        known_apis = {
            "deepseek": "https://api.deepseek.com",
            "groq": "https://api.groq.com/openai/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "xai": "https://api.x.ai/v1",
            "together": "https://api.together.xyz/v1",
        }
        if prov_name in known_apis and not prov_cfg.get("base_url"):
            prov_cfg["base_url"] = known_apis[prov_name]
        prov_cfg["model"] = prov_cfg.get("model", config.get("model_name") or "gpt-4o")
        if prov_cfg.get("api_key") or prov_cfg.get("base_url"):
            return OpenAIProvider(prov_cfg)
        raise ValueError(f"Unknown provider: {prov_name}. Use /provider add <name> <base_url> <key> to register.")
