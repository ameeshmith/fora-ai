"""LangGraph StateGraph workflow definition with self-correction cycles and SQLite checkpointing."""

import asyncio
import logging
from typing import Any, Dict, Literal, Optional
from langgraph.graph import END, START, StateGraph

from app.agent.nodes import critic_node, planner_node, researcher_node, synthesizer_node, verifier_node
from app.agent.state import ResearchState
from app.config import get_settings

logger = logging.getLogger("research_engine.agent.graph")

# Global async checkpointer instances
_async_checkpointer = None
_async_cm = None
_research_app = None


async def get_checkpointer():
    """Initializes and returns an active AsyncSqliteSaver checkpoint instance."""
    global _async_checkpointer, _async_cm
    if _async_checkpointer is not None:
        return _async_checkpointer

    settings = get_settings()
    db_path = settings.SQLITE_DB_PATH

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        _async_cm = AsyncSqliteSaver.from_conn_string(db_path)
        _async_checkpointer = await _async_cm.__aenter__()
        logger.info(f"Initialized AsyncSqliteSaver with '{db_path}'")
        return _async_checkpointer
    except Exception as exc:
        logger.warning(f"Could not initialize AsyncSqliteSaver ({exc}); falling back to MemorySaver.")
        try:
            from langgraph.checkpoint.memory import MemorySaver
            _async_checkpointer = MemorySaver()
            return _async_checkpointer
        except Exception as inner_exc:
            logger.error(f"Failed to initialize MemorySaver: {inner_exc}")
            return None


async def close_checkpointer():
    """Cleanly closes SQLite connection on shutdown."""
    global _async_checkpointer, _async_cm
    if _async_cm is not None:
        try:
            await _async_cm.__aexit__(None, None, None)
            logger.info("Closed AsyncSqliteSaver connection.")
        except Exception as exc:
            logger.warning(f"Error closing AsyncSqliteSaver: {exc}")
        finally:
            _async_checkpointer = None
            _async_cm = None


def should_continue_research(state: ResearchState) -> Literal["planner", "synthesizer"]:
    """Conditional edge router: Determines whether to iterate or move to synthesis."""
    passed = state.get("critique_passed", False)
    iteration = state.get("iteration_count", 1)
    max_iters = get_settings().MAX_RESEARCH_ITERATIONS

    if passed or iteration >= max_iters:
        logger.info(f"Routing to 'synthesizer' (passed={passed}, iteration={iteration}/{max_iters})")
        return "synthesizer"

    logger.info(f"Routing back to 'planner' for reflection iteration {iteration + 1}")
    return "planner"


async def get_research_app():
    """Constructs and returns compiled LangGraph with active checkpointer."""
    global _research_app
    if _research_app is not None:
        return _research_app

    workflow = StateGraph(ResearchState)

    # 1. Register 5-Stage Multi-Agent Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("verifier", verifier_node)

    # 2. Register Edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")

    # 3. Register Conditional Self-Correction Edge
    workflow.add_conditional_edges(
        "critic",
        should_continue_research,
        {
            "planner": "planner",
            "synthesizer": "synthesizer",
        },
    )

    # 4. Final Claim Verification Gate Edge
    workflow.add_edge("synthesizer", "verifier")
    workflow.add_edge("verifier", END)

    # 5. Attach Checkpointer for persistence
    checkpointer = await get_checkpointer()
    _research_app = workflow.compile(checkpointer=checkpointer)
    return _research_app


# Backward compatibility alias
build_research_graph = get_research_app


async def run_research_pipeline(task_id: str, topic: str, depth: str = "quick") -> Dict[str, Any]:
    """Asynchronous runner invoked by FastAPI background worker to execute graph."""
    logger.info(f"Starting research pipeline execution for task '{task_id}' (topic: '{topic}')")

    initial_state: ResearchState = {
        "task_id": task_id,
        "topic": topic,
        "depth": depth,
        "query_profile": {},
        "plan": [],
        "queries_searched": [],
        "scraped_sources": [],
        "draft_report": "",
        "critique": "",
        "critique_passed": False,
        "contradictions": [],
        "claims": [],
        "overall_uncertainty": "WELL_SUPPORTED",
        "iteration_count": 0,
        "current_step": "queued",
        "final_report": None,
        "verification_passed": False,
        "error": None,
    }

    config = {"configurable": {"thread_id": task_id}}

    try:
        app = await get_research_app()
        # Run graph to completion
        final_state = await app.ainvoke(initial_state, config=config)
        logger.info(f"Research pipeline completed successfully for task '{task_id}'")
        return final_state
    except Exception as exc:
        logger.exception(f"Unhandled error in research pipeline for task '{task_id}': {exc}")
        # Update checkpoint with error state if possible
        try:
            app = await get_research_app()
            error_state: Dict[str, Any] = {
                "task_id": task_id,
                "current_step": "failed",
                "error": str(exc),
            }
            await app.aupdate_state(config, error_state)
        except Exception:
            pass
        raise


async def get_task_state(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the latest state checkpoint for a given task ID."""
    config = {"configurable": {"thread_id": task_id}}
    try:
        app = await get_research_app()
        state_snapshot = await app.aget_state(config)
        if state_snapshot and state_snapshot.values:
            return state_snapshot.values
    except Exception as exc:
        logger.warning(f"Error querying checkpoint for task '{task_id}': {exc}")
    return None
