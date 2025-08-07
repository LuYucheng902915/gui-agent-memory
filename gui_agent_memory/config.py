"""
Configuration management for the memory system.

This module handles:
- Loading environment variables and API keys
- Initializing AI service clients
- Managing system-wide configuration
- Fail-fast validation of required settings
"""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


class MemoryConfig:
    """
    Central configuration class for the memory system.

    Implements fail-fast strategy - raises ConfigurationError immediately
    if required environment variables are missing.
    """

    def __init__(self) -> None:
        """Initialize configuration by loading environment variables."""
        # Load environment variables from .env file if it exists
        load_dotenv()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Gitee AI Embedding Configuration
        self.gitee_ai_embedding_base_url = self._get_required_env(
            "GITEE_AI_EMBEDDING_BASE_URL"
        )
        self.gitee_ai_embedding_api_key = self._get_required_env(
            "GITEE_AI_EMBEDDING_API_KEY"
        )

        # Gitee AI Reranker Configuration (separate endpoint)
        self.gitee_ai_reranker_base_url = self._get_required_env(
            "GITEE_AI_RERANKER_BASE_URL"
        )
        self.gitee_ai_reranker_api_key = self._get_required_env(
            "GITEE_AI_RERANKER_API_KEY"
        )

        # Experience LLM Configuration
        self.experience_llm_base_url = self._get_required_env("EXPERIENCE_LLM_BASE_URL")
        self.experience_llm_api_key = self._get_required_env("EXPERIENCE_LLM_API_KEY")

        # Model Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
        self.reranker_model = os.getenv("RERANKER_MODEL", "Qwen3-Reranker-8B")
        self.experience_llm_model = os.getenv("EXPERIENCE_LLM_MODEL", "gpt-4o")

        # ChromaDB Configuration
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./memory_system/data/chroma")
        self.experiential_collection_name = "experiential_memories"
        self.declarative_collection_name = "declarative_memories"

        # Retrieval Configuration
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", "20"))
        self.default_top_n = int(os.getenv("DEFAULT_TOP_N", "3"))
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "1024"))

        # Logging Configuration
        self.failed_learning_log_path = os.getenv(
            "FAILED_LEARNING_LOG_PATH",
            "./memory_system/logs/failed_learning_tasks.jsonl",
        )

        # Initialize clients
        self._init_clients()

    def _get_required_env(self, key: str) -> str:
        """
        Get a required environment variable.

        Args:
            key: Environment variable name

        Returns:
            Environment variable value

        Raises:
            ConfigurationError: If the required environment variable is missing
        """
        value = os.getenv(key)
        if not value:
            raise ConfigurationError(
                f"Required environment variable '{key}' is missing. "
                f"Please set it in your .env file or environment."
            )
        return value

    def _init_clients(self) -> None:
        """Initialize AI service clients."""
        try:
            # Initialize Gitee AI client for embeddings
            self.gitee_ai_embedding_client = OpenAI(
                api_key=self.gitee_ai_embedding_api_key,
                base_url=self.gitee_ai_embedding_base_url,
            )

            # Initialize Gitee AI client for reranker (separate endpoint)
            self.gitee_ai_reranker_client = OpenAI(
                api_key=self.gitee_ai_reranker_api_key,
                base_url=self.gitee_ai_reranker_base_url,
            )

            # Initialize Experience LLM client
            self.experience_llm_client = OpenAI(
                api_key=self.experience_llm_api_key,
                base_url=self.experience_llm_base_url,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize AI service clients: {e}"
            ) from e

    def get_embedding_client(self) -> OpenAI:
        """Get the embedding client (Gitee AI)."""
        return self.gitee_ai_embedding_client

    def get_reranker_client(self) -> OpenAI:
        """Get the reranker client (Gitee AI)."""
        return self.gitee_ai_reranker_client

    def get_experience_llm_client(self) -> OpenAI:
        """Get the experience distillation LLM client."""
        return self.experience_llm_client

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
            self.gitee_ai_embedding_client.models.list()

            # Test reranker client - try models.list() first, fallback to actual API call
            try:
                self.gitee_ai_reranker_client.models.list()
            except Exception:
                # If models.list() fails, test with actual reranker API call
                try:
                    import requests

                    response = requests.post(
                        self.gitee_ai_reranker_base_url,
                        json={
                            "query": "test",
                            "documents": ["test document"],
                            "model": self.reranker_model,
                        },
                        headers={
                            "X-Failover-Enabled": "true",
                            "Authorization": f"Bearer {self.gitee_ai_reranker_api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=10,
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
            self.experience_llm_client.models.list()

            return True
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e


# Global configuration instance
_config: MemoryConfig | None = None


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
        _config = MemoryConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance (mainly for testing)."""
    global _config
    _config = None
