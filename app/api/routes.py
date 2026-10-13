"""FastAPI REST routes for autonomous research execution, cancellation, and report retrieval."""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from app.agent.graph import get_task_state, run_research_pipeline
from app.schemas.research import (
    AnswerResponse,
    CancelResponse,
    QuestionRequest,
    ResearchReport,
    ResearchRequest,
    ResearchStartResponse,
    TaskStatusResponse,
    TaskSummaryItem,
)

logger = logging.getLogger("research_engine.api.routes")

router = APIRouter(prefix="/api/v1/research", tags=["Research Engine"])

# In-memory registry to provide immediate status tracking while LangGraph starts up
_in_memory_task_registry: Dict[str, Dict[str, Any]] = {}
_active_async_tasks: Dict[str, asyncio.Task] = {}


async def _execute_task_wrapper(
    task_id: str,
    topic: str,
    depth: str,
    time_limit_seconds: int = 120,
):
    """Wraps background task execution to ensure registry tracking, cancellation, and timeouts."""
    _in_memory_task_registry[task_id]["status"] = "processing"
    _in_memory_task_registry[task_id]["current_step"] = "planner"

    try:
        final_state = await asyncio.wait_for(
            run_research_pipeline(task_id=task_id, topic=topic, depth=depth),
            timeout=float(time_limit_seconds),
        )
        _in_memory_task_registry[task_id]["status"] = "completed"
        _in_memory_task_registry[task_id]["current_step"] = "completed"
        _in_memory_task_registry[task_id]["final_report"] = final_state.get("final_report")
    except asyncio.CancelledError:
        logger.info(f"Task {task_id} was cancelled by user directive.")
        _in_memory_task_registry[task_id]["status"] = "cancelled"
        _in_memory_task_registry[task_id]["current_step"] = "cancelled"
        logs = _in_memory_task_registry[task_id].setdefault("logs", [])
        logs.append({
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "node": "security",
            "action": "Task execution terminated by user cancellation directive",
            "status": "warning",
        })
    except asyncio.TimeoutError:
        logger.warning(f"Task {task_id} timed out after {time_limit_seconds}s.")
        _in_memory_task_registry[task_id]["status"] = "failed"
        _in_memory_task_registry[task_id]["current_step"] = "failed"
        _in_memory_task_registry[task_id]["error"] = f"Execution exceeded maximum configured time limit ({time_limit_seconds}s)."
    except Exception as exc:
        logger.error(f"Background execution failed for task {task_id}: {exc}")
        _in_memory_task_registry[task_id]["status"] = "failed"
        _in_memory_task_registry[task_id]["current_step"] = "failed"
        _in_memory_task_registry[task_id]["error"] = str(exc)
    finally:
        _active_async_tasks.pop(task_id, None)


@router.get(
    "",
    response_model=List[TaskSummaryItem],
    summary="List research tasks session history",
    description="Returns recent research missions created during the current session.",
)
async def list_research_tasks() -> List[TaskSummaryItem]:
    summaries: List[TaskSummaryItem] = []
    for tid, item in _in_memory_task_registry.items():
        rep = item.get("final_report") or {}
        qprof = rep.get("query_profile") or {}
        cits = rep.get("citations") or []
        summaries.append(
            TaskSummaryItem(
                task_id=tid,
                topic=item.get("topic", "Untitled Mission"),
                status=item.get("status", "processing"),
                depth=item.get("depth", "quick"),
                query_intent=qprof.get("intent") or item.get("query_intent"),
                created_at=item.get("created_at", datetime.now(timezone.utc).isoformat()),
                sources_count=len(cits),
            )
        )
    summaries.sort(key=lambda x: x.created_at, reverse=True)
    return summaries


@router.post(
    "",
    response_model=ResearchStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new research task",
    description="Submits a topic for multi-agent autonomous investigation and returns a task tracking ID.",
)
@router.post(
    "/start",
    response_model=ResearchStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    include_in_schema=False,
)
async def start_research(payload: ResearchRequest) -> ResearchStartResponse:
    task_id = str(uuid.uuid4())
    logger.info(f"Received research request: '{payload.topic}' (depth: {payload.depth}, max_sources: {payload.max_sources}, assigned_id: {task_id})")

    # Initialize tracking entry
    _in_memory_task_registry[task_id] = {
        "task_id": task_id,
        "topic": payload.topic,
        "depth": payload.depth,
        "max_sources": payload.max_sources,
        "time_limit_seconds": payload.time_limit_seconds,
        "status": "processing",
        "current_step": "planner",
        "iteration": 1,
        "logs": [],
        "error": None,
        "final_report": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Dispatch non-blocking background task with cancellation support
    task = asyncio.create_task(
        _execute_task_wrapper(
            task_id=task_id,
            topic=payload.topic,
            depth=payload.depth,
            time_limit_seconds=payload.time_limit_seconds,
        )
    )
    _active_async_tasks[task_id] = task

    return ResearchStartResponse(
        task_id=task_id,
        status="processing",
        message="Research pipeline initialized. Monitor progress via GET /api/v1/research/{task_id}.",
    )


@router.post(
    "/{task_id}/cancel",
    response_model=CancelResponse,
    summary="Cancel an active research task",
    description="Terminates an in-progress research execution to prevent wasted compute or model calls.",
)
async def cancel_research(task_id: str) -> CancelResponse:
    logger.info(f"Cancellation requested for task '{task_id}'")
    task = _active_async_tasks.get(task_id)

    if task and not task.done():
        task.cancel()
        if task_id in _in_memory_task_registry:
            _in_memory_task_registry[task_id]["status"] = "cancelled"
            _in_memory_task_registry[task_id]["current_step"] = "cancelled"
    elif task_id in _in_memory_task_registry:
        _in_memory_task_registry[task_id]["status"] = "cancelled"
        _in_memory_task_registry[task_id]["current_step"] = "cancelled"
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' not found.",
        )

    return CancelResponse(
        task_id=task_id,
        status="cancelled",
        message="Research task cancelled successfully.",
    )


@router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get research task progress",
    description="Polls current execution status, active node step, and iteration count from persistent checkpoint.",
)
async def get_research_status(task_id: str) -> TaskStatusResponse:
    # 1. Fast in-memory lookup first (zero lock contention, microsecond response)
    mem_state = _in_memory_task_registry.get(task_id)
    checkpoint_state = None

    if not mem_state or mem_state.get("status") not in ("completed", "failed", "cancelled"):
        # Query checkpoint if in-memory state is absent or needs hydration
        checkpoint_state = await get_task_state(task_id)

    if not checkpoint_state and not mem_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID '{task_id}' was not found.",
        )

    # Aggregate state: prefer active checkpoint if available, fallback to memory
    source_state = checkpoint_state or mem_state or {}

    current_step = source_state.get("current_step")
    final_report = source_state.get("final_report")
    error = source_state.get("error")
    iteration = source_state.get("iteration_count", source_state.get("iteration", 1))

    logs = source_state.get("logs", [])
    active_substep = source_state.get("active_substep")
    query_profile = source_state.get("query_profile") or {}
    query_intent = query_profile.get("intent")

    if (mem_state and mem_state.get("status") == "cancelled") or source_state.get("status") == "cancelled" or current_step == "cancelled":
        calc_status = "cancelled"
        current_step = "cancelled"
    elif error or current_step == "failed":
        calc_status = "failed"
    elif final_report is not None or current_step == "completed":
        calc_status = "completed"
    else:
        calc_status = "processing"

    return TaskStatusResponse(
        task_id=task_id,
        status=calc_status,
        current_step=current_step,
        active_substep=active_substep,
        iteration=iteration,
        query_intent=query_intent,
        logs=logs,
        error=error,
        report_available=final_report is not None,
    )


@router.post(
    "/{task_id}/cancel",
    response_model=CancelResponse,
    summary="Cancel a running research task",
    description="Terminates the background agent execution thread and updates state to cancelled.",
)
async def cancel_research_task(task_id: str) -> CancelResponse:
    # 1. Cancel the active asyncio task if running
    running_task = _active_async_tasks.get(task_id)
    if running_task and not running_task.done():
        running_task.cancel()
        logger.info(f"Cancellation signal dispatched to asyncio task for {task_id}")

    # 2. Update memory registry
    if task_id in _in_memory_task_registry:
        _in_memory_task_registry[task_id]["status"] = "cancelled"
        _in_memory_task_registry[task_id]["current_step"] = "cancelled"
        logs = _in_memory_task_registry[task_id].setdefault("logs", [])
        logs.append({
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "node": "security",
            "action": "Mission execution cancelled by investigator directive",
            "status": "warning",
        })
    else:
        _in_memory_task_registry[task_id] = {
            "status": "cancelled",
            "current_step": "cancelled",
            "logs": [{
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "node": "security",
                "action": "Mission execution cancelled by investigator directive",
                "status": "warning",
            }],
        }

    return CancelResponse(
        task_id=task_id,
        status="cancelled",
        message="Research task cancelled successfully.",
    )


@router.get(
    "/{task_id}/report",
    response_model=ResearchReport,
    summary="Retrieve final synthesized research report",
    description="Returns the full executive summary, key findings, and verified citations once synthesis completes.",
)
async def get_research_report(task_id: str) -> ResearchReport:
    checkpoint_state = await get_task_state(task_id)
    mem_state = _in_memory_task_registry.get(task_id)

    if not checkpoint_state and not mem_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' does not exist.",
        )

    source_state = checkpoint_state or mem_state or {}
    final_report_data = source_state.get("final_report")

    if not final_report_data:
        task_status = source_state.get("status") or "processing"
        if task_status == "cancelled":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This research task was cancelled by user directive.",
            )
        if source_state.get("error"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Task failed with error: {source_state.get('error')}",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report is not ready yet. Task is currently '{task_status}'. Please poll status at /api/v1/research/{task_id}.",
        )

    return ResearchReport(**final_report_data)


@router.post(
    "/{task_id}/ask",
    response_model=AnswerResponse,
    summary="Ask follow-up questions with zero hallucination",
    description="Grounds answers strictly in verified research sources using Groq with temperature=0.0.",
)
async def ask_research_question(
    task_id: str,
    payload: QuestionRequest,
) -> AnswerResponse:
    checkpoint_state = await get_task_state(task_id)
    mem_state = _in_memory_task_registry.get(task_id)

    if not checkpoint_state and not mem_state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' does not exist.",
        )

    source_state = checkpoint_state or mem_state or {}
    sources = source_state.get("scraped_sources", [])

    if not sources:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No verified sources are available for this task.",
        )

    # Build prompt with temperature=0.0 and prompt injection protection
    context_blocks = []
    source_urls = []
    for s in sources[:8]:
        url = s.get("url", "")
        if url:
            source_urls.append(url)
            content_preview = (s.get("content") or s.get("snippet", ""))[:600].replace("\n", " ")
            context_blocks.append(
                f"<untrusted_source_content>\n"
                f"Source: {s.get('title', url)} ({url})\n"
                f"Content: {content_preview}\n"
                f"</untrusted_source_content>"
            )

    system_prompt = (
        "You are an Elite Intelligence Verification Agent answering follow-up questions.\n"
        "ZERO-HALLUCINATION RULE: Answer STRICTLY using only facts from the provided sources.\n"
        "If the answer cannot be determined from the sources, state clearly:\n"
        "'Based on the verified research sources, this specific information is not documented.'\n"
        "Never speculate or invent facts beyond the provided text."
    )

    user_prompt = f"Question: {payload.question}\n\nVerified Sources:\n" + "\n\n".join(context_blocks)

    try:
        from app.config import get_settings
        from langchain_core.messages import HumanMessage, SystemMessage
        settings = get_settings()
        llm = settings.get_llm(temperature=0.0)
        res = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        answer_text = res.content if hasattr(res, "content") else str(res)
    except Exception as exc:
        logger.error(f"Grounded Q&A failed for task {task_id}: {exc}")
        answer_text = f"An error occurred while evaluating the query: {exc}"

    return AnswerResponse(
        task_id=task_id,
        question=payload.question,
        answer=answer_text,
        grounded_sources=source_urls[:4],
    )
