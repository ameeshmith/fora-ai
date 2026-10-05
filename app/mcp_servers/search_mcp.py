"""Custom MCP Tool Server for live web search (DuckDuckGo DDGS) and sandboxed HTML extraction."""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import httpx

logger = logging.getLogger("research_engine.mcp.search")

# Initialize official Anthropic MCP server via FastMCP interface
try:
    from mcp.server.fastmcp import FastMCP
    search_mcp = FastMCP("SearchAndScrapeServer")
except ImportError:
    # Graceful fallback shim when mcp library is being installed or tested in minimal environments
    class FastMCPFallback:
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            def decorator(func):
                return func
            return decorator

        def run(self, transport: str = "stdio"):
            logger.info(f"Running mock MCP server {self.name} on transport {transport}")

    search_mcp = FastMCPFallback("SearchAndScrapeServer")


async def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Execute live, zero-cost search via DuckDuckGo (DDGS).
    
    Returns a list of dicts with keys: 'title', 'url', 'snippet'.
    """
    logger.info(f"Executing DuckDuckGo search: '{query}' (limit={max_results})")
    results: List[Dict[str, str]] = []

    def _sync_search():
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
                return raw_results
        except Exception as err:
            logger.warning(f"DuckDuckGo search encountered an issue: {err}")
            return []

    try:
        raw = await asyncio.to_thread(_sync_search)
        for item in raw:
            title = item.get("title", "").strip()
            url = item.get("href") or item.get("link") or item.get("url", "")
            snippet = item.get("body") or item.get("snippet", "")
            if url and (title or snippet):
                results.append({
                    "title": title or url,
                    "url": url,
                    "snippet": snippet.strip(),
                })
    except Exception as exc:
        logger.error(f"Error during DuckDuckGo execution for '{query}': {exc}")

    return results


import ipaddress
import socket
from urllib.parse import urlparse


def is_safe_url(url: str) -> bool:
    """SSRF Protection: Rejects loopback, private, link-local, and cloud metadata addresses."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "metadata.google.internal"):
            return False
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False
            if str(ip) == "169.254.169.254":
                return False
        except ValueError:
            try:
                addr_info = socket.getaddrinfo(hostname, None)
                for item in addr_info:
                    resolved_ip = ipaddress.ip_address(item[4][0])
                    if resolved_ip.is_loopback or resolved_ip.is_private or resolved_ip.is_link_local or resolved_ip.is_reserved:
                        return False
            except Exception:
                pass
        return True
    except Exception:
        return False


async def fetch_article_markdown(url: str, timeout: int = 10) -> Dict[str, Any]:
    """Safely fetch and clean an article webpage with SSRF protection and noise stripping."""
    if not is_safe_url(url):
        logger.warning(f"Blocked SSRF/Unsafe URL request: {url}")
        return {
            "url": url,
            "title": "",
            "content": "[Blocked: URL resolves to private, loopback, or metadata address]",
            "status": "error",
        }

    logger.info(f"Fetching article content from: {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        strict_timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(
            headers=headers,
            timeout=strict_timeout,
            follow_redirects=True,
            verify=False,
        ) as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                logger.warning(f"HTTP {resp.status_code} fetching {url}")
                return {
                    "url": url,
                    "title": "",
                    "content": f"[Error: Server returned HTTP {resp.status_code}]",
                    "status": "error",
                }
            html = resp.text

        soup = BeautifulSoup(html, "html.parser")

        # Strip scripts, styles, forms, and navigation chrome
        for noise in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form"]):
            noise.decompose()

        page_title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Target primary content containers if available
        article = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"content|article|body", re.I)) or soup.body

        if article:
            # Extract paragraphs and headings
            elements = article.find_all(["h1", "h2", "h3", "h4", "p", "li"])
            text_blocks = []
            for el in elements:
                t = el.get_text().strip()
                if len(t) > 25:  # filter noise fragments
                    if el.name in ["h1", "h2", "h3"]:
                        text_blocks.append(f"\n### {t}\n")
                    elif el.name == "li":
                        text_blocks.append(f"- {t}")
                    else:
                        text_blocks.append(t)
            clean_text = "\n\n".join(text_blocks)
        else:
            clean_text = soup.get_text(separator="\n", strip=True)

        # Normalize whitespace and limit length to safe context bounds
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
        max_length = 6000
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "\n...[truncated for length]"

        return {
            "url": url,
            "title": page_title or url,
            "content": clean_text.strip(),
            "status": "success",
        }

    except Exception as exc:
        logger.warning(f"Failed to scrape {url}: {exc}")
        return {
            "url": url,
            "title": "",
            "content": f"[Scrape error: {str(exc)}]",
            "status": "error",
        }


# Register tools on MCP server
@search_mcp.tool()
async def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Runs DuckDuckGo search, extracts clean snippets, titles, and URLs."""
    return await search_duckduckgo(query, max_results)


@search_mcp.tool()
async def fetch_article_content(url: str) -> Dict[str, Any]:
    """Safely scrapes article body text, trims HTML noise, and returns clean markdown text."""
    return await fetch_article_markdown(url)


if __name__ == "__main__":
    # Allows running as a standalone MCP server over stdio
    try:
        search_mcp.run()
    except Exception as e:
        logger.error(f"Error running search MCP server: {e}")
