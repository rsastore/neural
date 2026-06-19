"""Basic tests for RSA Agentic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Verify all core modules import without error."""
    from agent import AgentSession, build_system_prompt
    from grammar import parse_tool_call
    assert build_system_prompt() is not None
    assert parse_tool_call('{"tool":"test"}') is not None
    print("  ✅ All imports OK")

def test_grammar():
    from grammar import parse_tool_call
    cases = [
        ('{"tool":"exec_shell","args":{"cmd":"ls"}}', "exec_shell"),
        ('{"tool":: "exec_shell"}', "exec_shell"),
    ]
    for text, expected in cases:
        result = parse_tool_call(text)
        assert result is not None and result["tool"] == expected, f"Failed: {text}"
    print("  ✅ Grammar parser OK")

def test_safety():
    from tools.builtin import _exec_shell, DESTRUCTIVE_PATTERNS
    assert len(DESTRUCTIVE_PATTERNS) > 0
    # Verify destructive patterns actually block
    for pattern in DESTRUCTIVE_PATTERNS[:3]:
        assert "blocked" in _exec_shell(pattern).lower() or "block" in _exec_shell(pattern).lower()
    print("  ✅ Safety blocks OK")

def test_providers():
    from models.providers import create_provider
    cfg = {"provider": "ollama", "ollama": {"host": "http://localhost:11434"}, "model_name": "test"}
    p = create_provider(cfg)
    assert p is not None
    assert p.name.startswith("Ollama")
    print("  ✅ Provider factory OK")

if __name__ == "__main__":
    test_imports()
    test_grammar()
    test_safety()
    print("\nAll tests passed!")
