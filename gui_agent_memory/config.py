"""
Configuration management for the memory system.

This module handles:
- Loading environment variables and API keys
- Initializing AI service clients
- Managing system-wide configuration
- Fail-fast validation of required settings

Implemented with pydantic-settings for declarative env parsing and validation.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import AnyUrl, Field, SecretStr, ValidationError, field_validator
from pydantic.fields import PrivateAttr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Provide a load_dotenv symbol for tests that patch it, without changing runtime behavior
try:  # pragma: no cover - existence for patching compatibility
    from dotenv import load_dotenv as _real_load_dotenv
except Exception:  # pragma: no cover
    _real_load_dotenv = None


def load_dotenv(*args: Any, **kwargs: Any) -> None:
    """Compatibility wrapper for tests; runtime does not rely on it."""
    if _real_load_dotenv is not None:
        _real_load_dotenv(*args, **kwargs)
    return None


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


class MemoryConfig(BaseSettings):
    """
    Configuration using pydantic-settings, with validation and env aliases.
    """

    # Embedding LLM Service (required)
    embedding_llm_base_url: AnyUrl = Field(alias="EMBEDDING_LLM_BASE_URL")
    embedding_llm_api_key: SecretStr = Field(alias="EMBEDDING_LLM_API_KEY")

    # Reranker LLM Service (required)
    reranker_llm_base_url: AnyUrl = Field(alias="RERANKER_LLM_BASE_URL")
    reranker_llm_api_key: SecretStr = Field(alias="RERANKER_LLM_API_KEY")

    # Experience LLM Configuration (required)
    experience_llm_base_url: AnyUrl = Field(alias="EXPERIENCE_LLM_BASE_URL")
    experience_llm_api_key: SecretStr = Field(alias="EXPERIENCE_LLM_API_KEY")

    # Model Configuration (optional)
    embedding_model: str = Field(default="Qwen3-Embedding-8B", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(default="Qwen3-Reranker-8B", alias="RERANKER_MODEL")
    experience_llm_model: str = Field(default="gpt-4o", alias="EXPERIENCE_LLM_MODEL")

    # ChromaDB Configuration
    chroma_db_path: Path = Field(
        default=Path("./memory_system/data/chroma"), alias="CHROMA_DB_PATH"
    )
    experiential_collection_name: str = Field(default="experiential_memories")
    declarative_collection_name: str = Field(default="declarative_memories")

    # Retrieval Configuration
    default_top_k: int = Field(default=20, alias="DEFAULT_TOP_K")
    default_top_n: int = Field(default=3, alias="DEFAULT_TOP_N")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    # Similarity policy threshold for LLM judge routing
    similarity_threshold_judge: float = Field(
        default=0.80, alias="SIMILARITY_THRESHOLD_JUDGE"
    )

    # Logging Configuration
    failed_learning_log_path: Path = Field(
        default=Path("./memory_system/logs/failed_learning_tasks.jsonl"),
        alias="FAILED_LEARNING_LOG_PATH",
    )
    prompt_log_dir: Path = Field(
        default=Path("./memory_system/logs/prompts"), alias="PROMPT_LOG_DIR"
    )
    operation_log_dir: Path = Field(
        default=Path("./memory_system/logs/operations"), alias="OPERATION_LOG_DIR"
    )
    log_enabled: bool = Field(default=True, alias="LOG_ENABLED")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # Optional: external prompt templates directory; if empty, use packaged prompts
    prompt_templates_dir: Path | None = Field(
        default=None, alias="PROMPT_TEMPLATES_DIR"
    )

    @field_validator("prompt_templates_dir", mode="before")
    @classmethod
    def _normalize_prompt_dir(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # Advanced/infra Configuration
    chroma_anonymized_telemetry: bool = Field(
        default=False, alias="CHROMA_ANONYMIZED_TELEMETRY"
    )
    rerank_candidate_limit: int = Field(default=20, alias="RERANK_CANDIDATE_LIMIT")
    hybrid_topk_multiplier: int = Field(default=4, alias="HYBRID_TOPK_MULTIPLIER")
    http_timeout_seconds: int = Field(default=10, alias="HTTP_TIMEOUT_SECONDS")

    # pydantic-settings configuration
    # Load variables from environment and automatically from a project-root .env file.
    # Priority (high -> low) follows pydantic-settings docs:
    # init kwargs > env vars > .env > secrets > defaults
    # Ref: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        In tests (pytest), or when explicitly disabled, skip reading .env to ensure
        deterministic behavior that relies only on process env and defaults.

        Ref: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
        """
        disable_dotenv_flag = os.getenv("MEMORY_DISABLE_DOTENV", "").lower() in {
            "1",
            "true",
            "yes",
        }
        running_pytest = "PYTEST_CURRENT_TEST" in os.environ

        if disable_dotenv_flag or running_pytest:
            return (init_settings, env_settings, file_secret_settings)
        return (init_settings, env_settings, dotenv_settings, file_secret_settings)

    # Runtime attributes (not part of settings fields)
    _logger: logging.Logger | None = PrivateAttr(default=None)
    _embedding_llm_client: OpenAI | None = PrivateAttr(default=None)
    _reranker_llm_client: OpenAI | None = PrivateAttr(default=None)
    _experience_llm_client: OpenAI | None = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        try:
            super().__init__(**data)
        except ValidationError as e:
            raise ConfigurationError(f"Missing or invalid configuration: {e}") from e

        self._logger = logging.getLogger(__name__)
        # Normalize log level
        if isinstance(self.log_level, str):
            self.log_level = self.log_level.upper()
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize AI service clients."""
        try:
            # Initialize embedding LLM client (OpenAI-compatible)
            self._embedding_llm_client = OpenAI(
                api_key=self.embedding_llm_api_key.get_secret_value(),
                base_url=str(self.embedding_llm_base_url),
            )

            # Initialize reranker LLM client (OpenAI-compatible)
            self._reranker_llm_client = OpenAI(
                api_key=self.reranker_llm_api_key.get_secret_value(),
                base_url=str(self.reranker_llm_base_url),
            )

            # Initialize Experience LLM client (OpenAI-compatible)
            self._experience_llm_client = OpenAI(
                api_key=self.experience_llm_api_key.get_secret_value(),
                base_url=str(self.experience_llm_base_url),
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize AI service clients: {e}"
            ) from e

    def get_embedding_client(self) -> OpenAI:
        """Get the embedding LLM client (OpenAI-compatible)."""
        if self._embedding_llm_client is None:
            raise ConfigurationError("Embedding client is not initialized")
        return self._embedding_llm_client

    def get_reranker_client(self) -> OpenAI:
        """Get the reranker LLM client (OpenAI-compatible)."""
        if self._reranker_llm_client is None:
            raise ConfigurationError("Reranker client is not initialized")
        return self._reranker_llm_client

    def get_experience_llm_client(self) -> OpenAI:
        """Get the experience distillation LLM client."""
        if self._experience_llm_client is None:
            raise ConfigurationError("Experience LLM client is not initialized")
        return self._experience_llm_client

    def validate_configuration(self) -> bool:
        """
        Validate that all required configuration is present and clients are working.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Test embedding client
            if self._embedding_llm_client is None:
                raise ConfigurationError("Embedding client not initialized")
            self._embedding_llm_client.models.list()

            # Test reranker client - try models.list() first, fallback to actual API call
            try:
                if self._reranker_llm_client is None:
                    raise ConfigurationError("Reranker client not initialized")
                self._reranker_llm_client.models.list()
            except Exception:
                # If models.list() fails, test with actual reranker API call
                try:
                    import requests

                    response = requests.post(
                        str(self.reranker_llm_base_url),
                        json={
                            "query": "test",
                            "documents": ["test document"],
                            "model": self.reranker_model,
                        },
                        headers={
                            "X-Failover-Enabled": "true",
                            "Authorization": f"Bearer {self.reranker_llm_api_key.get_secret_value()}",
                            "Content-Type": "application/json",
                        },
                        timeout=self.http_timeout_seconds,
                    )
                    # If we get any response (even an error about the model),
                    # it means the API endpoint is reachable
                    if response.status_code in [200, 400, 422]:
                        # These status codes indicate the API is working
                        pass
                    else:
                        response.raise_for_status()
                except Exception as e:
                    raise ConfigurationError(
                        f"Reranker API validation failed: {e}"
                    ) from e

            # Test experience LLM client
            if self._experience_llm_client is None:
                raise ConfigurationError("Experience client not initialized")
            self._experience_llm_client.models.list()

            return True
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e

    def debug_dump(self) -> dict[str, Any]:
        """Return a sanitized snapshot of current config (secrets masked)."""

        def mask(secret: SecretStr | str) -> str:
            val = (
                secret.get_secret_value()
                if isinstance(secret, SecretStr)
                else str(secret)
            )
            if not val:
                return ""
            return (val[:2] + "***" + val[-2:]) if len(val) > 4 else "***"

        # Diagnose current dotenv loading mode for observability only
        disable_dotenv_flag = os.getenv("MEMORY_DISABLE_DOTENV", "").lower() in {
            "1",
            "true",
            "yes",
        }
        running_pytest = "PYTEST_CURRENT_TEST" in os.environ
        if running_pytest:
            dotenv_mode = "skipped_by_pytest"
        elif disable_dotenv_flag:
            dotenv_mode = "skipped_by_env"
        else:
            dotenv_mode = "enabled"

        return {
            "embedding_llm_base_url": str(self.embedding_llm_base_url),
            "embedding_llm_api_key": mask(self.embedding_llm_api_key),
            "reranker_llm_base_url": str(self.reranker_llm_base_url),
            "reranker_llm_api_key": mask(self.reranker_llm_api_key),
            "experience_llm_base_url": str(self.experience_llm_base_url),
            "experience_llm_api_key": mask(self.experience_llm_api_key),
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "experience_llm_model": self.experience_llm_model,
            "chroma_db_path": str(self.chroma_db_path),
            "experiential_collection_name": self.experiential_collection_name,
            "declarative_collection_name": self.declarative_collection_name,
            "default_top_k": self.default_top_k,
            "default_top_n": self.default_top_n,
            "embedding_dimension": self.embedding_dimension,
            "similarity_threshold_judge": self.similarity_threshold_judge,
            "failed_learning_log_path": str(self.failed_learning_log_path),
            "prompt_log_dir": str(self.prompt_log_dir),
            "operation_log_dir": str(self.operation_log_dir),
            "prompt_templates_dir": str(self.prompt_templates_dir)
            if self.prompt_templates_dir
            else None,
            "log_enabled": self.log_enabled,
            "log_level": self.log_level,
            "chroma_anonymized_telemetry": self.chroma_anonymized_telemetry,
            "rerank_candidate_limit": self.rerank_candidate_limit,
            "hybrid_topk_multiplier": self.hybrid_topk_multiplier,
            "http_timeout_seconds": self.http_timeout_seconds,
            "dotenv_mode": dotenv_mode,
        }


# Global configuration instance
_config: MemoryConfig | None = None
_config_lock = threading.RLock()


def get_config() -> MemoryConfig:
    """
    Get the global configuration instance.

    Returns:
        MemoryConfig instance

    Raises:
        ConfigurationError: If configuration initialization fails
    """
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = MemoryConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance (mainly for testing)."""
    global _config
    with _config_lock:
        _config = None


def set_config(new_config: MemoryConfig | None) -> None:
    """Explicitly inject a configuration instance (for DI or specialized runs)."""
    global _config
    with _config_lock:
        _config = new_config
