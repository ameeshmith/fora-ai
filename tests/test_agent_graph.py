"""Integration tests for LangGraph state machine, self-correction cycle, and FastAPI endpoints."""

from typing import Any, Dict
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import pytest

from app.agent.graph import build_research_graph, run_research_pipeline, should_continue_research
from app.agent.nodes import critic_node, planner_node, researcher_node, synthesizer_node
from app.agent.state import ResearchState
from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_planner_node_generation():
    """Verify planner generates structured search plan."""
    state: ResearchState = {
        "task_id": "test-planner-1",
        "topic": "Neuromorphic Computing Chips",
        "depth": "quick",
        "iteration_count": 0,
        "queries_searched": [],
        "scraped_sources": [],
    }

    with patch("app.config.Settings.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value.content = '["neuromorphic computing architecture", "spiking neural network hardware"]'
        mock_get_llm.return_value = mock_llm

        res = await planner_node(state)

        assert "plan" in res
        assert len(res["plan"]) >= 1
        assert res["current_step"] == "planner"


@pytest.mark.asyncio
async def test_researcher_node_execution():
    """Verify researcher queries tools and aggregates sources."""
    state: ResearchState = {
        "task_id": "test-res-1",
        "topic": "Solid State Batteries",
        "depth": "quick",
        "plan": ["solid state battery commercialization"],
        "queries_searched": [],
        "scraped_sources": [],
    }

    mock_search = [
        {"title": "Solid State Breakthrough", "url": "https://example.com/battery", "snippet": "New electrolyte discovered."}
    ]
    mock_scrape = {
        "status": "success",
        "title": "Solid State Breakthrough",
        "content": "Full article: A novel solid electrolyte delivers 500 Wh/kg energy density with high thermal stability.",
        "url": "https://example.com/battery",
    }

    with patch("app.agent.nodes.search_duckduckgo", new_callable=AsyncMock) as m_search, \
         patch("app.agent.nodes.fetch_article_markdown", new_callable=AsyncMock) as m_scrape:
        m_search.return_value = mock_search
        m_scrape.return_value = mock_scrape

        res = await researcher_node(state)

        assert len(res["scraped_sources"]) == 1
        assert res["scraped_sources"][0]["url"] == "https://example.com/battery"
        assert res["current_step"] == "researcher"


@pytest.mark.asyncio
async def test_critic_node_self_correction_and_approval():
    """Verify critic flags insufficiency on first loop and passes on valid sources or max iteration."""
    # Scenario A: Insufficient sources triggers reflection loop
    state_poor: ResearchState = {
        "task_id": "crit-1",
        "topic": "Quantum Encryption",
        "scraped_sources": [],
        "iteration_count": 0,
    }
    res_poor = await critic_node(state_poor)
    assert res_poor["critique_passed"] is False
    assert should_continue_research({"critique_passed": False, "iteration_count": 1}) == "planner"

    # Scenario B: Max iterations reached triggers synthesis
    state_max: ResearchState = {
        "task_id": "crit-2",
        "topic": "Quantum Encryption",
        "scraped_sources": [{"title": "QKD", "url": "https://qkd.org", "content": "Long enough text about quantum key distribution."}],
        "iteration_count": 3,
    }
    res_max = await critic_node(state_max)
    assert res_max["critique_passed"] is True
    assert should_continue_research({"critique_passed": True, "iteration_count": 3}) == "synthesizer"


@pytest.mark.asyncio
async def test_synthesizer_node_report_structure():
    """Verify synthesizer produces compliant ResearchReport payload."""
    state: ResearchState = {
        "task_id": "synth-1",
        "topic": "Room Temperature Superconductors",
        "scraped_sources": [
            {
                "title": "Superconductor Survey",
                "url": "https://nature.com/articles/supercon",
                "snippet": "High pressure hydride research continues.",
                "content": "High pressure hydride research continues at multiple global facilities.",
            }
        ],
        "iteration_count": 1,
    }

    with patch("app.config.Settings.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value.content = (
            "# Room Temperature Superconductors\n\n"
            "## Executive Summary\n"
            "Research indicates promising hydrides under mega-bar pressures.\n\n"
            "## Key Findings\n"
            "- Discovered hydride phase at 250K ([Superconductor Survey](https://nature.com/articles/supercon))\n\n"
            "## Citations and References\n"
            "1. [Superconductor Survey](https://nature.com/articles/supercon)\n"
        )
        mock_get_llm.return_value = mock_llm

        res = await synthesizer_node(state)

        assert "final_report" in res
        report = res["final_report"]
        assert report["topic"] == "Room Temperature Superconductors"
        assert len(report["key_findings"]) >= 1
        assert len(report["citations"]) == 1
        assert report["citations"][0]["url"] == "https://nature.com/articles/supercon"
        assert res["current_step"] == "completed"


def test_api_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "research-engine"


def test_api_research_flow():
    """Test full API submission, progress status, and report retrieval."""
    # 1. Submit task
    post_res = client.post(
        "/api/v1/research",
        json={"topic": "AI Agents in Healthcare", "depth": "quick"},
    )
    assert post_res.status_code == 202
    data = post_res.json()
    task_id = data["task_id"]
    assert data["status"] == "processing"

    # 2. Check task status immediately
    status_res = client.get(f"/api/v1/research/{task_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["task_id"] == task_id
    assert status_data["status"] in ["processing", "completed"]

    # 3. Report before completion returns 400
    report_res = client.get(f"/api/v1/research/{task_id}/report")
    assert report_res.status_code in [200, 400]
