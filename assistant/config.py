"""Centralized configuration for the AI agent."""

import os

# ── Ollama ──────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5-custom")

# ── LLM Settings ───────────────────────────────────────
LLM_TIMEOUT = 120.0          # Seconds — max wait for a single LLM call
THINKING_TEMPERATURE = 0.1   # Low temp for tool-decision steps (precision)
ANSWER_TEMPERATURE = 0.3     # Slightly higher for the final answer (fluency)

# ── Agent Loop ─────────────────────────────────────────
MAX_AGENT_STEPS = 20         # Hard safety cap on research iterations

# ── Web Search & Fetching ──────────────────────────────
SEARCH_TIMEOUT = 10.0        # Seconds — timeout for HTTP requests
MAX_SEARCH_RESULTS = 5       # Results to return per DuckDuckGo search
MAX_FETCH_CHARS = 8000       # Max characters extracted from a fetched page