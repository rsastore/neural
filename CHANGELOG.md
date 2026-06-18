# Changelog

## v0.1.0 (2026-06-19)

### Added
- Initial release of RSA Agentic
- Agent loop with tool calling (18 tools)
- Multi-provider: Ollama, OpenAI, Anthropic, Google + auto-compat
- Planner engine: Goal -> Steps -> Execute -> Retry
- RAG + Self-learning with BM25 search
- Hybrid search: BM25 + Embeddings (nomic-embed-text)
- Vector database (Chroma adapter)
- SQLite storage for sessions, knowledge, cost tracking
- Terminal context awareness (CWD, git, OS, history)
- Agent personas (coder, sysadmin, research, default)
- MCP plugin system + custom plugin loader
- Docker sandbox for safe execution
- Model manager (HF search, pull, switch)
- Dataset manager (Nemotron & HF datasets)
- REST API server mode
- GitHub tools (issue, PR, search)
- Git tools (status, diff, commit, log)
- Cost tracking (token usage + $ estimates)
- Scheduled tasks
- Interactive file explorer
- Project auto-detection
- Fine-tuning script generator
- Safety system (rm -rf block, system file block, approval gates)
- Grammar-guided JSON parsing with auto-fix
- Context compaction
- Session persistence

### Supported Platforms
- Linux (primary)
- macOS
- Windows (beta)
- Android/Termux
