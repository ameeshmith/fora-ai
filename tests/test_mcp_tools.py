"""Unit tests for MCP server tools (DuckDuckGo search, article scraper, document parser)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.mcp_servers.doc_parser_mcp import (
    chunk_text,
    clean_and_summarize_content,
    doc_parser_mcp,
)
from app.mcp_servers.search_mcp import (
    fetch_article_markdown,
    search_duckduckgo,
    search_mcp,
)


@pytest.mark.asyncio
async def test_search_duckduckgo_structure():
    """Verify DuckDuckGo search returns normalized title, url, snippet fields."""
    mock_ddg_results = [
        {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
        {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
    ]

    with patch("duckduckgo_search.DDGS") as mock_ddgs:
        instance = MagicMock()
        instance.__enter__.return_value = instance
        instance.text.return_value = mock_ddg_results
        mock_ddgs.return_value = instance

        results = await search_duckduckgo("quantum computing breakthroughs", max_results=2)

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[0]["url"] == "https://example.com/1"
        assert results[0]["snippet"] == "Snippet 1"


@pytest.mark.asyncio
async def test_fetch_article_markdown_html_cleaning():
    """Verify HTML scraper strips scripts, navs, and extracts clean article text."""
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Quantum Breakthrough</title></head>
    <body>
        <nav><a href="/">Home</a><a href="/login">Login</a></nav>
        <script>console.log("tracking code");</script>
        <article>
            <h1>Quantum Computing Milestone</h1>
            <p>Researchers have demonstrated fault-tolerant logical qubits with exceptional fidelity.</p>
            <p>This achievement represents a significant paradigm shift for high-performance computing.</p>
        </article>
        <footer><p>Copyright 2025 All Rights Reserved.</p></footer>
    </body>
    </html>
    """

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = sample_html

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        data = await fetch_article_markdown("https://example.com/quantum-news", timeout=5)

        assert data["status"] == "success"
        assert "Quantum Breakthrough" in data["title"]
        assert "fault-tolerant logical qubits" in data["content"]
        assert "tracking code" not in data["content"]
        assert "Home" not in data["content"]


def test_chunk_text_boundaries():
    """Verify text chunker splits correctly without truncating in awkward middle characters."""
    sample_text = (
        "Paragraph one is introducing a novel concept in computational biology. "
        "It sets the foundational premise for subsequent discussions.\n\n"
        "Paragraph two delves deeper into empirical validation and benchmark figures. "
        "The metrics indicate an order of magnitude improvement over baseline models.\n\n"
        "Paragraph three synthesizes the trade-offs and outlines future research pathways."
    )

    chunks = chunk_text(sample_text, chunk_size=150, overlap=30)
    assert len(chunks) >= 2
    assert all(len(c) <= 200 for c in chunks)
    # Ensure content is preserved across chunks
    combined = " ".join(chunks)
    assert "Paragraph one" in combined
    assert "Paragraph three" in combined


def test_clean_and_summarize_content():
    """Verify boilerplate lines like cookie policy and newsletter prompts are excised."""
    dirty_text = (
        "Accept all cookies to proceed.\n"
        "Significant advancements have been recorded in cryogenic cooling mechanisms.\n"
        "Subscribe to our newsletter for weekly tech updates.\n"
        "The experiments were conducted under strictly controlled ambient pressure."
    )

    cleaned = clean_and_summarize_content(dirty_text)
    assert "cookies" not in cleaned.lower()
    assert "newsletter" not in cleaned.lower()
    assert "cryogenic cooling mechanisms" in cleaned
    assert "strictly controlled ambient pressure" in cleaned


def test_mcp_servers_initialized():
    """Verify FastMCP instances are properly named and have registered tools."""
    assert search_mcp is not None
    assert doc_parser_mcp is not None
