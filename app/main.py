"""FastAPI Application entrypoint with CORS, lifespan management, and interactive documentation."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
import sys
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as research_router
from app.config import get_settings
from app.agent.graph import close_checkpointer, get_checkpointer

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("research_engine.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for application startup and clean resource shutdown."""
    logger.info("Initializing Autonomous Multi-Agent Research Engine...")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER} (Default Model: {settings.DEFAULT_MODEL})")
    logger.info(f"Checkpoint Store: {settings.SQLITE_DB_PATH}")

    # Initialize checkpointer
    try:
        checkpointer = await get_checkpointer()
        if checkpointer:
            logger.info("AsyncSqliteSaver checkpointer pre-warmed successfully.")
    except Exception as exc:
        logger.warning(f"Checkpointer initialization warning: {exc}")

    yield

    logger.info("Shutting down Research Engine services gracefully.")
    await close_checkpointer()


app = FastAPI(
    title="FORA AI - Autonomous Multi-Agent Deep Research Engine",
    description=(
        "FORA AI: Production-grade, zero-cost research backend orchestrating LangGraph multi-agent teams "
        "(Planner, Researcher, Critic, Synthesizer) with Model Context Protocol (MCP) live search tools."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled server error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred while processing your request."},
    )


# Setup Static Directory for UI
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Health check endpoint
@app.get("/health", tags=["System"], summary="Health check probe")
async def health_check():
    """Liveness and readiness probe for container orchestrators."""
    return {
        "status": "healthy",
        "service": "FORA AI",
        "llm_provider": settings.LLM_PROVIDER,
        "model": settings.DEFAULT_MODEL,
        "database": settings.SQLITE_DB_PATH,
    }


# Root endpoint: Serves Deep Research Dashboard UI
@app.get("/", tags=["UI"], include_in_schema=False)
async def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return FileResponse(str(index_file), headers=headers)
    return {
        "message": "Welcome to the Autonomous Multi-Agent Research Engine API.",
        "docs": "/docs",
    }


# Mount API Routers
app.include_router(research_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
