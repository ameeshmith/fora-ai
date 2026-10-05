"""Custom MCP Tool Server for text chunking, document parsing, and noise elimination."""

import logging
import re
from typing import List, Optional

logger = logging.getLogger("research_engine.mcp.doc_parser")

try:
    from mcp.server.fastmcp import FastMCP
    doc_parser_mcp = FastMCP("DocParserServer")
except ImportError:
    class FastMCPFallback:
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            def decorator(func):
                return func
            return decorator

        def run(self, transport: str = "stdio"):
            logger.info(f"Running mock MCP server {self.name} on transport {transport}")

    doc_parser_mcp = FastMCPFallback("DocParserServer")


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """Splits raw text into manageable overlapping chunks respecting paragraph/sentence boundaries."""
    if not text or not text.strip():
        return []

    # Clean redundant whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Attempt to split on paragraph or sentence boundary
        split_point = text.rfind("\n\n", start, end)
        if split_point == -1:
            split_point = text.rfind(". ", start, end)
        if split_point == -1 or split_point <= start:
            split_point = end

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start with overlap
        start = max(start + 1, split_point - overlap if split_point < end else end - overlap)

    return chunks


def clean_and_summarize_content(text: str, focus_topic: Optional[str] = None) -> str:
    """Cleans extracted content, removes common web artifacts, and retains top relevant excerpts."""
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_lines = []
    
    # Strip cookie disclaimers, login prompts, and social share artifacts
    boilerplate_patterns = [
        r"cookie policy",
        r"privacy policy",
        r"all rights reserved",
        r"subscribe to our newsletter",
        r"sign up for free",
        r"accept all cookies",
        r"terms of service",
        r"advertisement",
    ]
    compiled_patterns = [re.compile(p, re.I) for p in boilerplate_patterns]

    for line in lines:
        if any(pat.search(line) for pat in compiled_patterns):
            continue
        cleaned_lines.append(line)

    filtered_text = "\n".join(cleaned_lines)

    # If focus topic keywords are present, prioritize paragraphs containing them
    if focus_topic:
        keywords = [k.lower() for k in focus_topic.split() if len(k) > 3]
        paragraphs = filtered_text.split("\n\n")
        relevant = [
            p for p in paragraphs
            if any(k in p.lower() for k in keywords)
        ]
        if relevant:
            return "\n\n".join(relevant[:8])

    return filtered_text[:4000]


@doc_parser_mcp.tool()
async def parse_and_chunk(text: str, chunk_size: int = 1500) -> List[str]:
    """MCP tool: Cleans and chunks extracted document text."""
    return chunk_text(text, chunk_size=chunk_size)


@doc_parser_mcp.tool()
async def filter_document_noise(text: str, topic: str = "") -> str:
    """MCP tool: Filters boilerplate and extracts high-salience text blocks."""
    return clean_and_summarize_content(text, focus_topic=topic)


if __name__ == "__main__":
    try:
        doc_parser_mcp.run()
    except Exception as e:
        logger.error(f"Error running doc parser MCP server: {e}")
