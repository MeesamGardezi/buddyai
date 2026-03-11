"""Tool registry and dispatcher.

Defines all tools available to the agent, parses tool calls from
LLM output, and dispatches execution.

Tools:
    SEARCH    → Web search via DuckDuckGo
    FETCH     → Fetch and extract content from a URL
    DONE      → Signal that research is complete
"""

import re
from dataclasses import dataclass
from core.search import web_search, fetch_page, format_results_for_llm


# ── Data Classes ────────────────────────────────────────


@dataclass
class ToolCall:
    """A parsed tool invocation from LLM output."""
    name: str
    argument: str


@dataclass
class ToolResult:
    """The outcome of executing a tool."""
    tool_name: str
    argument: str
    output: str
    success: bool
    status_message: str  # Brief message shown to the user during research


# ── Tool Patterns ───────────────────────────────────────
# Order matters — checked top to bottom.

_TOOL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("search", re.compile(r"\[SEARCH:\s*(.*?)\]", re.DOTALL)),
    ("fetch",  re.compile(r"\[FETCH:\s*(.*?)\]", re.DOTALL)),
    ("done",   re.compile(r"\[DONE\]")),
]


# ── Parsing ─────────────────────────────────────────────


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse the first tool call found in LLM output.

    Args:
        text: The raw LLM response text.

    Returns:
        A ToolCall if found, or None if the text contains no tool call.
    """
    for name, pattern in _TOOL_PATTERNS:
        match = pattern.search(text)
        if match:
            argument = match.group(1).strip() if match.lastindex else ""
            # Clean up common LLM artifacts around the argument
            argument = argument.strip("<>\"'` ")
            return ToolCall(name=name, argument=argument)
    return None


# ── Execution ───────────────────────────────────────────


async def _execute_search(query: str) -> ToolResult:
    """Execute a web search."""
    results = await web_search(query)
    formatted = format_results_for_llm(results)

    if not results:
        return ToolResult(
            tool_name="search",
            argument=query,
            output=f"Search for '{query}' returned no results. Try different keywords.",
            success=False,
            status_message=f"Searched for: **{query}** — no results found",
        )

    return ToolResult(
        tool_name="search",
        argument=query,
        output=f"Search results for '{query}':\n\n{formatted}",
        success=True,
        status_message=f"Searched the web for: **{query}**",
    )


async def _execute_fetch(url: str) -> ToolResult:
    """Fetch and extract content from a URL."""
    page = await fetch_page(url)

    if not page.success:
        return ToolResult(
            tool_name="fetch",
            argument=url,
            output=f"Failed to fetch {url}: {page.error}",
            success=False,
            status_message=f"Failed to read: **{url}**",
        )

    title_line = f"Title: {page.title}\n" if page.title else ""
    return ToolResult(
        tool_name="fetch",
        argument=url,
        output=f"Content from {url}:\n{title_line}\n{page.text}",
        success=True,
        status_message=f"Read page: **{url}**",
    )


async def _execute_done() -> ToolResult:
    """The DONE signal — no actual execution needed."""
    return ToolResult(
        tool_name="done",
        argument="",
        output="",
        success=True,
        status_message="Research complete",
    )


# Dispatch table
_EXECUTORS = {
    "search": _execute_search,
    "fetch":  _execute_fetch,
    "done":   _execute_done,
}


async def execute_tool(tool_call: ToolCall) -> ToolResult:
    """Dispatch and execute a tool call.

    Args:
        tool_call: The parsed ToolCall to execute.

    Returns:
        ToolResult with the tool's output and metadata.
    """
    executor = _EXECUTORS.get(tool_call.name)
    if not executor:
        return ToolResult(
            tool_name=tool_call.name,
            argument=tool_call.argument,
            output=f"Unknown tool: {tool_call.name}",
            success=False,
            status_message=f"Unknown tool: {tool_call.name}",
        )

    if tool_call.name == "done":
        return await executor()

    return await executor(tool_call.argument)


# ── Tool Descriptions (for the system prompt) ──────────


def get_tool_descriptions() -> str:
    """Return a formatted string describing all available tools.

    This is injected into the system prompt so the LLM knows
    what tools exist and how to invoke them.
    """
    return (
        "## YOUR TOOLS\n\n"

        "### 1. Web Search\n"
        "Search the web for information on any topic.\n"
        "Syntax: [SEARCH: your query here]\n"
        "Example: [SEARCH: SpaceX Starship launch 2025]\n"
        "Tips:\n"
        "  - Use specific, targeted keywords.\n"
        "  - If results are poor, try rephrasing with different terms.\n"
        "  - You can search multiple times with different queries.\n\n"

        "### 2. Fetch URL\n"
        "Read the full content of a specific webpage.\n"
        "Syntax: [FETCH: https://example.com/page]\n"
        "Example: [FETCH: https://openai.com/blog/gpt-5]\n"
        "Tips:\n"
        "  - Use this to deep-dive into a promising search result.\n"
        "  - Use this when the user provides a specific URL or domain.\n"
        "  - URLs from search results can be fetched for more detail.\n"
        "  - Do NOT add 'www.' to a URL unless the user specifically included it.\n\n"

        "### 3. Done\n"
        "Signal that you have gathered enough information.\n"
        "Syntax: [DONE]\n"
        "Use this when you are confident you can write a thorough answer.\n"
    )
