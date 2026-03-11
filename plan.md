# Local AI Assistant — Project Plan

## Overview

A fully local, privacy-first AI assistant running on macOS. The LLM runs on-device via Ollama, with a lean HTML/CSS/JS frontend served by a lightweight Python backend. Web search is handled via scraping. The project is designed to be open source and progressively extensible with agentic capabilities.

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Frontend | Vanilla HTML / CSS / JS | Lean, no build step, no framework overhead |
| Backend | Python + FastAPI | Lightweight, async, best AI/agentic ecosystem |
| LLM Runtime | Ollama (local) | Fully local, free, privacy-preserving |
| LLM Model | qwen3.5-custom (Qwen3.5:0.8b base) | Fast on M1, custom system prompt, swappable |
| Web Search | Python scraping (BeautifulSoup + httpx) | Free, no API key, no external dependencies |
| Communication | REST + SSE (Server-Sent Events) | Simple streaming from FastAPI to frontend |

---

## Project Structure

```
assistant/
├── main.py                  # FastAPI app entry point
├── requirements.txt         # Python dependencies
├── config.py                # User-facing config (model name, ollama host, etc.)
│
├── core/
│   ├── llm.py               # Ollama client, streaming responses
│   ├── search.py            # Web scraping logic (DuckDuckGo HTML scrape)
│   └── agent.py             # Decision logic: when to search vs. answer directly
│
└── frontend/
    ├── index.html           # Main UI
    ├── style.css            # Styles
    └── app.js               # Chat logic, SSE handling, render markdown
```

---

## Stage 1 — Simple Chat + Web Search (Current Scope)

### Goals
- Chat with a local Ollama model via a clean browser UI
- AI autonomously decides when to perform a web search
- Search results are injected into context before LLM responds
- Responses stream in real time

### Features
- [ ] Chat UI (message bubbles, input box, send button)
- [ ] Streaming responses via SSE
- [ ] Web search via scraping DuckDuckGo HTML (no API key needed)
- [ ] Agent layer: LLM prompted to emit a `[SEARCH: query]` signal when needed
- [ ] Backend intercepts signal, runs search, re-prompts LLM with results
- [ ] Markdown rendering in frontend (for code blocks, lists, etc.)
- [ ] Model selector (dropdown to switch Ollama models)
- [ ] Clear conversation button

### How Search Works (No API Key Approach)
1. User sends a message
2. Backend sends message to LLM with a system prompt that instructs it to emit `[SEARCH: <query>]` if it needs current/external info
3. Backend detects the signal in the streamed response
4. Backend scrapes DuckDuckGo HTML results for the query
5. Top 3–5 results (title + snippet + URL) are injected back into the conversation context
6. LLM is called again with the search results and produces a final answer
7. Final answer streams to the frontend

---

## Stage 2 — Agentic Capabilities (Planned)

These will be added incrementally as tools the agent can invoke:

- [ ] **File system access** — read/write local files on request
- [ ] **Code execution** — run Python snippets in a sandbox
- [ ] **Browser control** — Playwright for JavaScript-heavy pages (full scraping)
- [ ] **Calendar / reminders** — read/write macOS calendar via AppleScript or `icalBuddy`
- [ ] **Memory** — persist conversation summaries to a local vector store (e.g., ChromaDB)
- [ ] **Multi-tool chaining** — agent can call multiple tools in sequence to complete a task
- [ ] **Plugin system** — drop-in Python tool files the agent can discover and use

---

## Custom Model Setup

The project uses a customized Ollama model with a trimmed system prompt for concise, no-fluff responses.

**Modelfile:**
```
FROM qwen3.5:0.8b
SYSTEM """
You are a no-nonsense assistant. You give short, direct answers.
No fluff, no filler, no unnecessary explanations.
"""
```

**Create the model:**
```bash
cat > Modelfile << 'EOF'
FROM qwen3.5:0.8b
SYSTEM """
You are a no-nonsense assistant. You give short, direct answers.
No fluff, no filler, no unnecessary explanations.
"""
EOF

ollama create qwen3.5-custom -f Modelfile
```

The default model name in `config.py` will be set to `qwen3.5-custom`. Users can swap it out for any Ollama model via the UI.

---

## Running the Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure Ollama is running
brew services start ollama

# 3. Create the custom model (one-time setup)
ollama create qwen3.5-custom -f Modelfile

# 4. Start the assistant
python main.py

# 5. Open in browser
# http://localhost:8000
```

---

## Python Dependencies (requirements.txt)

```
fastapi
uvicorn
httpx
beautifulsoup4
ollama
python-dotenv
```

---

## Design Principles

- **Local first** — no data leaves your machine except for web search queries
- **No API keys** — everything is free and open
- **Swappable models** — any Ollama-compatible model works
- **Minimal dependencies** — no Docker, no databases in Stage 1
- **Progressive enhancement** — Stage 1 is fully functional; Stage 2 tools are opt-in

---

## Open Questions / Decisions Deferred

| Question | Status |
|---|---|
| Markdown rendering library (marked.js vs highlight.js combo) | Decide in Stage 1 build |
| Whether to store chat history to disk between sessions | Decide in Stage 1 build |
| Scraping fallback if DuckDuckGo blocks (rotate user agents, add Brave scrape) | Revisit if needed |
| Vector memory store for Stage 2 (ChromaDB vs sqlite-vec) | Decide in Stage 2 |