"""Application configuration module using Pydantic v2 Settings."""

from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Any, List, Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("research_engine.config")

# Resolve absolute path to .env file
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Central configuration for Autonomous Multi-Agent Research Engine."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM Settings
    LLM_PROVIDER: Literal["groq", "ollama", "mock"] = Field(
        default="groq",
        description="LLM provider: 'groq' for Groq Cloud API, 'ollama' for local Ollama, or 'mock' for testing",
    )
    GROQ_API_KEY: Optional[str] = Field(
        default="",
        description="Groq API key (Required if LLM_PROVIDER=groq)",
    )
    DEFAULT_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default Groq model identifier",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama server",
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.1:8b",
        description="Ollama model tag to query",
    )

    # Agent Engine Controls
    MAX_RESEARCH_ITERATIONS: int = Field(
        default=3,
        description="Maximum reflection and critique loops before final synthesis",
    )
    SEARCH_MAX_RESULTS: int = Field(
        default=5,
        description="Number of search results to pull per query",
    )
    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="HTTP timeout in seconds for scraping and external calls",
    )

    # State & Persistence
    SQLITE_DB_PATH: str = Field(
        default="checkpoints.db",
        description="File path for persistent SQLite LangGraph checkpoints",
    )

    # API Server Settings
    API_PORT: int = Field(default=8000, description="Port to bind FastAPI application")
    API_HOST: str = Field(default="0.0.0.0", description="Host interface for server")
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )
    LOG_LEVEL: str = Field(default="INFO", description="Console logging level")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]

    def get_llm(self, model: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None):
        """Factory method returning a configured LangChain ChatModel based on settings."""
        provider = self.LLM_PROVIDER.lower()

        if provider == "groq":
            api_key = self.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning(
                    "GROQ_API_KEY is not set. If running without a live Groq key, "
                    "configure LLM_PROVIDER=ollama or set GROQ_API_KEY in .env."
                )
            try:
                from langchain_groq import ChatGroq

                kwargs = {
                    "model": model or self.DEFAULT_MODEL,
                    "groq_api_key": api_key or "placeholder_key",
                    "temperature": temperature,
                    "timeout": self.REQUEST_TIMEOUT_SECONDS,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens

                return ChatGroq(**kwargs)
            except Exception as exc:
                logger.error(f"Failed to initialize ChatGroq: {exc}. Falling back to FakeListChatModel if in test.")
                raise

        elif provider == "ollama":
            try:
                from langchain_ollama import ChatOllama

                return ChatOllama(
                    base_url=self.OLLAMA_BASE_URL,
                    model=model or self.OLLAMA_MODEL,
                    temperature=temperature,
                    timeout=float(self.REQUEST_TIMEOUT_SECONDS),
                )
            except Exception as exc:
                logger.error(f"Failed to initialize ChatOllama: {exc}")
                raise

        elif provider == "mock":
            from langchain_community.chat_models import FakeListChatModel

            return FakeListChatModel(
                responses=[
                    '{"queries": ["quantum computing breakthroughs 2025", "quantum error correction milestone"]}',
                    "Findings cover key areas well with verified facts.",
                    "# Quantum Computing\n\n## Executive Summary\nSignificant advances.\n\n## Key Findings\n- Error correction milestone achieved.\n\n## Sources\n- [Source 1](https://example.com)",
                ]
            )

        raise ValueError(f"Unsupported LLM_PROVIDER '{self.LLM_PROVIDER}'. Choose 'groq', 'ollama', or 'mock'.")


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
