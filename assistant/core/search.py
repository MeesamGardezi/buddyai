"""Web search and page fetching utilities.

web_search()             → Search DuckDuckGo, return structured results.
fetch_page()             → Fetch and extract readable text from a URL.
format_results_for_llm() → Format search results into an LLM-friendly string.
"""

from dataclasses import dataclass
import re
import urllib.parse
import httpx
from bs4 import BeautifulSoup
from config import SEARCH_TIMEOUT, MAX_SEARCH_RESULTS, MAX_FETCH_CHARS

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) "
        "Gecko/20100101 Firefox/111.0"
    )
}


# ── Data Classes ────────────────────────────────────────


@dataclass
class SearchResult:
    """A single web search result."""
    index: int
    title: str
    url: str
    snippet: str


@dataclass
class PageContent:
    """Extracted content from a fetched web page."""
    url: str
    title: str
    text: str
    success: bool
    error: str | None = None


# ── Web Search ──────────────────────────────────────────


def _resolve_ddg_redirect(raw_url: str) -> str:
    """Extract the real URL from a DuckDuckGo redirect link."""
    if raw_url.startswith("//duckduckgo.com/l/?"):
        match = re.search(r"uddg=([^&]+)", raw_url)
        if match:
            return urllib.parse.unquote(match.group(1))
    # Handle https:// prefix variant
    if raw_url.startswith("https://duckduckgo.com/l/?"):
        match = re.search(r"uddg=([^&]+)", raw_url)
        if match:
            return urllib.parse.unquote(match.group(1))
    return raw_url


async def web_search(query: str, num_results: int = MAX_SEARCH_RESULTS) -> list[SearchResult]:
    """Scrape DuckDuckGo HTML for search results.

    Args:
        query:       The search query string.
        num_results: Maximum number of results to return.

    Returns:
        List of SearchResult objects. Empty list on failure.
    """
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=_HEADERS, timeout=SEARCH_TIMEOUT)
            response.raise_for_status()
    except Exception as e:
        print(f"[search] Request failed: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[SearchResult] = []

    for idx, item in enumerate(soup.find_all("div", class_="result"), start=1):
        if len(results) >= num_results:
            break

        # Skip ad/sponsored results
        item_classes = item.get("class", [])
        if "result--ad" in item_classes or "result--sep" in item_classes:
            continue

        title_elem = item.find("a", class_="result__a")
        url_elem = item.find("a", class_="result__url")
        snippet_elem = item.find("a", class_="result__snippet")

        # Fallback: try broader selectors if primary ones fail
        if not snippet_elem:
            snippet_elem = item.find("td", class_="result__snippet")
        if not snippet_elem:
            snippet_elem = item.find(class_="result__snippet")

        # Extract title — prefer result__a, fallback to URL text
        if title_elem:
            title = title_elem.get_text(strip=True)
        elif url_elem:
            title = url_elem.get_text(strip=True)
        else:
            title = "Untitled"

        # Extract and resolve URL
        raw_url = ""
        if title_elem:
            raw_url = title_elem.get("href", "")
        if not raw_url and url_elem:
            raw_url = url_elem.get("href", "")
        # If href is empty, try the displayed URL text
        if not raw_url and url_elem:
            displayed = url_elem.get_text(strip=True)
            if displayed and "." in displayed:
                raw_url = displayed if displayed.startswith("http") else f"https://{displayed}"

        resolved_url = _resolve_ddg_redirect(raw_url)

        # Extract snippet text
        snippet = snippet_elem.get_text(strip=True) if snippet_elem else title

        # Only skip if we have neither a URL nor useful content
        if not resolved_url and not snippet:
            continue

        results.append(SearchResult(
            index=len(results) + 1,
            title=title,
            url=resolved_url,
            snippet=snippet,
        ))

    return results


# ── Page Fetching ───────────────────────────────────────


# Tags that typically contain non-content noise
_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "form", "iframe", "noscript",
]

# Class/ID patterns for ad and sidebar containers
_NOISE_PATTERNS = re.compile(
    r"(sidebar|widget|advert|promo|popup|modal|cookie|banner|comment)",
    re.IGNORECASE,
)


async def fetch_page(url: str) -> PageContent:
    """Fetch a URL and extract its readable text content.

    Strips navigation, ads, scripts, and other noise.
    Truncates to MAX_FETCH_CHARS.

    Args:
        url: The webpage URL to fetch.

    Returns:
        PageContent with extracted text or error info.
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, headers=_HEADERS, timeout=SEARCH_TIMEOUT, follow_redirects=True
            )
            response.raise_for_status()
    except httpx.TimeoutException:
        return PageContent(url=url, title="", text="", success=False, error="Request timed out")
    except Exception as e:
        return PageContent(url=url, title="", text="", success=False, error=str(e))

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Remove noise elements by tag
    for tag_name in _NOISE_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Remove noise elements by class/id pattern
    for element in soup.find_all(True):
        classes = " ".join(element.get("class", []))
        elem_id = element.get("id", "")
        if _NOISE_PATTERNS.search(classes) or _NOISE_PATTERNS.search(elem_id):
            element.decompose()

    # Extract and clean text
    text = soup.get_text(separator="\n", strip=True)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Truncate
    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + "\n\n[Content truncated...]"

    return PageContent(url=url, title=title, text=text, success=True)


# ── Formatting ──────────────────────────────────────────


def format_results_for_llm(results: list[SearchResult]) -> str:
    """Format search results into a numbered block for LLM context.

    Example output:
        1. «Title»
           URL: https://example.com
           Snippet: Brief description of the page...

    The numbering lets the LLM refer to specific results
    (e.g., "Fetch result #3 for more details").
    """
    if not results:
        return "No search results found."

    lines: list[str] = []
    for r in results:
        lines.append(f"{r.index}. {r.title}")
        lines.append(f"   URL: {r.url}")
        lines.append(f"   Snippet: {r.snippet}")
        lines.append("")

    return "\n".join(lines).strip()
