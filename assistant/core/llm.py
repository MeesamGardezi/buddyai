"""LLM client — dual-mode interface to Ollama.

complete()  → Full response as string (for silent tool-decision steps).
stream()    → Async generator of chunks (for the user-facing final answer).
"""

import json
import httpx
from config import OLLAMA_HOST, MODEL_NAME, LLM_TIMEOUT


async def complete(
    messages: list[dict],
    temperature: float = 0.1,
) -> str:
    """Send messages to the LLM and return the full response at once.

    Used during the research phase where we need the complete text
    to reliably parse tool calls — no partial-match guessing.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=LLM_TIMEOUT)
            response.raise_for_status()
            data = response.json() or {}
            message = data.get("message") or {}
            return message.get("content", "")
        except httpx.ConnectError:
            return "[ERROR] Could not connect to Ollama. Is it running?"
        except httpx.TimeoutException:
            return "[ERROR] LLM request timed out."
        except Exception as e:
            return f"[ERROR] {e}"


async def stream(
    messages: list[dict],
    temperature: float = 0.3,
):
    """Stream LLM response chunks as they arrive.

    Used only for the final answer phase — the user sees tokens
    appearing in real-time.

    Yields:
        str: Individual text chunks from the model.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": temperature},
    }

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST", url, json=payload, timeout=LLM_TIMEOUT
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line) or {}
                        message = data.get("message") or {}
                        content = message.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        pass
        except httpx.ConnectError:
            yield "[ERROR] Could not connect to Ollama. Is it running?"
        except httpx.TimeoutException:
            yield "[ERROR] LLM request timed out."
        except Exception as e:
            yield f"[ERROR] {e}"