"""Tool registry and dispatcher.

Defines all tools available to the agent, parses tool calls from
LLM output, and dispatches execution.

Tools:
    SEARCH    → Web search via DuckDuckGo
    FETCH     → Fetch and extract content from a URL
    BROWSER_GOTO → Open URL in browser
    BROWSER_CLICK → Click an element in browser
    BROWSER_TYPE → Type text into an element in browser
    BROWSER_VIEW → Get screenshot of browser
    DONE      → Signal that research is complete
"""

import re
import asyncio
from dataclasses import dataclass
from core.search import web_search, fetch_page, format_results_for_llm
from core.browser import browser_manager


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
    image_base64: str | None = None


# ── Tool Patterns ───────────────────────────────────────
# Order matters — checked top to bottom.
# We include shortened fallbacks (CLICK, TYPE, GOTO, VIEW) because
# the LLM sometimes drops the BROWSER_ prefix.

_TOOL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("browser_goto",  re.compile(r"\[BROWSER_GOTO:\s*(.*?)\]", re.DOTALL)),
    ("browser_click", re.compile(r"\[BROWSER_CLICK:\s*(.*?)\]", re.DOTALL)),
    ("browser_type",  re.compile(r"\[BROWSER_TYPE:\s*(.*?)\]", re.DOTALL)),
    ("browser_view",  re.compile(r"\[BROWSER_VIEW\]", re.DOTALL)),
    # Fallbacks for shortened names
    ("browser_goto",  re.compile(r"\[GOTO:\s*(.*?)\]", re.DOTALL)),
    ("browser_click", re.compile(r"\[CLICK:\s*(.*?)\]", re.DOTALL)),
    ("browser_type",  re.compile(r"\[TYPE:\s*(.*?)\]", re.DOTALL)),
    ("browser_view",  re.compile(r"\[VIEW\]", re.DOTALL)),
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
            argument = argument.strip("<>\"'` ")
            return ToolCall(name=name, argument=argument)
    return None


# ── Execution ───────────────────────────────────────────


def _execute_search(query: str) -> ToolResult:
    """Execute a web search."""
    results = web_search(query)
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


def _execute_fetch(url: str) -> ToolResult:
    """Fetch and extract content from a URL."""
    page = fetch_page(url)

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


def _execute_done() -> ToolResult:
    """The DONE signal — no actual execution needed."""
    return ToolResult(
        tool_name="done",
        argument="",
        output="",
        success=True,
        status_message="Research complete",
    )


async def _execute_browser_goto(url: str) -> ToolResult:
    try:
        page = await browser_manager.get_page()
        await page.goto(url, wait_until="networkidle")
        image = await browser_manager.get_screenshot_base64()
        return ToolResult(
            tool_name="browser_goto",
            argument=url,
            output=f"Navigated to {url}. Attached screenshot.",
            success=True,
            status_message=f"Browsing to: **{url}**",
            image_base64=image
        )
    except Exception as e:
        return ToolResult(
            tool_name="browser_goto",
            argument=url,
            output=f"Error navigating: {e}",
            success=False,
            status_message=f"Failed to load: **{url}**"
        )


async def _execute_browser_click(selector: str) -> ToolResult:
    try:
        page = await browser_manager.get_page()
        await page.click(selector)
        image = await browser_manager.get_screenshot_base64()
        return ToolResult(
            tool_name="browser_click",
            argument=selector,
            output=f"Clicked on '{selector}'. Attached screenshot.",
            success=True,
            status_message=f"Clicked element: **{selector}**",
            image_base64=image
        )
    except Exception as e:
        return ToolResult("browser_click", selector, f"Error: {e}", False, f"Failed clicking {selector}")


async def _execute_browser_type(args: str) -> ToolResult:
    try:
        parts = [p.strip() for p in args.split("|", 1)]
        if len(parts) != 2:
            return ToolResult("browser_type", args, "Error: Argument must be 'selector | text'", False, "Type error")
        selector, text = parts
        page = await browser_manager.get_page()
        await page.fill(selector, text)
        image = await browser_manager.get_screenshot_base64()
        return ToolResult(
            tool_name="browser_type",
            argument=args,
            output=f"Typed text into '{selector}'. Attached screenshot.",
            success=True,
            status_message=f"Typed in: **{selector}**",
            image_base64=image
        )
    except Exception as e:
        return ToolResult("browser_type", args, f"Error: {e}", False, "Type error")


async def _execute_browser_view() -> ToolResult:
    try:
        image = await browser_manager.get_screenshot_base64()
        return ToolResult(
            tool_name="browser_view",
            argument="",
            output="Current browser view:",
            success=True,
            status_message="Viewing screen",
            image_base64=image
        )
    except Exception as e:
        return ToolResult("browser_view", "", f"Error: {e}", False, "Failed to view screen")


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
    if tool_call.name == "browser_goto":
        return await _execute_browser_goto(tool_call.argument)
    elif tool_call.name == "browser_click":
        return await _execute_browser_click(tool_call.argument)
    elif tool_call.name == "browser_type":
        return await _execute_browser_type(tool_call.argument)
    elif tool_call.name == "browser_view":
        return await _execute_browser_view()
    elif tool_call.name == "search":
        return await asyncio.to_thread(_execute_search, tool_call.argument)
    elif tool_call.name == "fetch":
        return await asyncio.to_thread(_execute_fetch, tool_call.argument)
    elif tool_call.name == "done":
        return await asyncio.to_thread(_execute_done)
    
    return ToolResult(
        tool_name=tool_call.name,
        argument=tool_call.argument,
        output=f"Unknown tool: {tool_call.name}",
        success=False,
        status_message=f"Unknown tool: {tool_call.name}",
    )


# ── Tool Descriptions (for the system prompt) ──────────


def get_tool_descriptions() -> str:
    """Return a formatted string describing all available tools.

    This is injected into the system prompt so the LLM knows
    what tools exist and how to invoke them.
    """
    return (
        "## YOUR TOOLS\n\n"

        "### 1. Web Search\n"
        "Search the web for general information when no specific URL/domain is given.\n"
        "Syntax: [SEARCH: your query here]\n"
        "Example: [SEARCH: SpaceX Starship launch 2025]\n\n"

        "### 2. Fetch URL\n"
        "Read the full content of a specific webpage (static fetch). USE THIS FIRST when a domain is mentioned.\n"
        "Syntax: [FETCH: https://example.com]\n\n"

        "### 3. Browser Control (Interactive)\n"
        "Use these to open webpages in a real browser, click buttons, or log into websites.\n"
        "  - [BROWSER_GOTO: https://example.com]\n"
        "  - [BROWSER_CLICK: #login-button]  (CSS selector)\n"
        "  - [BROWSER_TYPE: #email | user@host.com]  (CSS selector | text to type)\n"
        "  - [BROWSER_VIEW]  (Check the screen state without doing anything)\n"
        "Note: Every browser tool action automatically returns a screenshot of the page so you can see what happened.\n\n"

        "### 4. Done\n"
        "Signal that you have gathered enough information.\n"
        "Syntax: [DONE]\n"
        "Use this when you are confident you can write a thorough answer.\n"
    )