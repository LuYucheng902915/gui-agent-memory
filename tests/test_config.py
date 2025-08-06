"""
Unit tests for the configuration module.
"""

import os
from unittest.mock import patch

import pytest

from gui_agent_memory.config import (
    ConfigurationError,
    MemoryConfig,
    get_config,
    reset_config,
)


class TestMemoryConfig:
    """Test cases for MemoryConfig class."""

    def test_config_initialization_success(self, mock_openai_client):
        """Test successful configuration initialization."""
        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            config = MemoryConfig()

            # Test with FAKE test environment values - NOT real API keys!
            assert (
                config.gitee_ai_embedding_base_url
                == "https://test-embedding.example.com/v1"
            )
            assert config.gitee_ai_embedding_api_key == "test-fake-embedding-key-12345"
            assert (
                config.gitee_ai_reranker_base_url
                == "https://test-reranker.example.com/v1/rerank"
            )
            assert config.gitee_ai_reranker_api_key == "test-fake-reranker-key-67890"
            assert config.experience_llm_base_url == "https://test-llm.example.com/v1"
            assert config.experience_llm_api_key == "test-fake-llm-key-abcdef"
            assert config.embedding_model == "Qwen3-Embedding-8B"
            assert config.reranker_model == "Qwen3-Reranker-8B"

    def test_config_missing_required_env_var(self, monkeypatch):
        """Test configuration failure with missing required environment variable."""
        # Mock os.getenv to return None for required variables
        original_getenv = os.getenv

        def mock_getenv(key, default=None):
            if key in [
                "GITEE_AI_EMBEDDING_API_KEY",
                "GITEE_AI_EMBEDDING_BASE_URL",
                "GITEE_AI_RERANKER_API_KEY",
                "GITEE_AI_RERANKER_BASE_URL",
                "EXPERIENCE_LLM_API_KEY",
                "EXPERIENCE_LLM_BASE_URL",
            ]:
                return None
            return original_getenv(key, default)

        monkeypatch.setattr(os, "getenv", mock_getenv)

        # Reset config to ensure clean state
        from gui_agent_memory.config import reset_config

        reset_config()

        with pytest.raises(ConfigurationError) as exc_info:
            MemoryConfig()

        assert "missing" in str(exc_info.value).lower()

    def test_config_client_initialization_failure(self, monkeypatch):
        """Test configuration failure during client initialization."""
        # Set required env vars
        monkeypatch.setenv("GITEE_AI_EMBEDDING_API_KEY", "test-key")
        monkeypatch.setenv("GITEE_AI_EMBEDDING_BASE_URL", "https://ai.gitee.com/v1")
        monkeypatch.setenv("GITEE_AI_RERANKER_API_KEY", "test-key")
        monkeypatch.setenv(
            "GITEE_AI_RERANKER_BASE_URL", "https://ai.gitee.com/v1/rerank"
        )
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://poloai.top/v1")

        # Mock OpenAI to raise an exception
        with patch(
            "gui_agent_memory.config.OpenAI",
            side_effect=Exception("Client init failed"),
        ):
            with pytest.raises(ConfigurationError) as exc_info:
                MemoryConfig()

            assert "Failed to initialize AI service clients" in str(exc_info.value)

    def test_config_default_values(self, mock_openai_client):
        """Test that default configuration values are set correctly."""
        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            config = MemoryConfig()

            assert config.embedding_model == "Qwen3-Embedding-8B"
            assert config.reranker_model == "Qwen3-Reranker-8B"
            assert config.experience_llm_model == "gpt-4o"
            assert config.default_top_k == 20
            assert config.default_top_n == 3
            assert config.embedding_dimension == 1024

    def test_config_custom_env_values(self, monkeypatch, mock_openai_client):
        """Test configuration with custom environment values."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom-embedding-model")
        monkeypatch.setenv("RERANKER_MODEL", "custom-reranker-model")
        monkeypatch.setenv("DEFAULT_TOP_K", "15")
        monkeypatch.setenv("DEFAULT_TOP_N", "5")

        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            config = MemoryConfig()

            assert config.embedding_model == "custom-embedding-model"
            assert config.reranker_model == "custom-reranker-model"
            assert config.default_top_k == 15
            assert config.default_top_n == 5

    def test_get_clients(self, mock_openai_client):
        """Test getting client instances."""
        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            config = MemoryConfig()

            embedding_client = config.get_embedding_client()
            reranker_client = config.get_reranker_client()
            llm_client = config.get_experience_llm_client()

            assert embedding_client is not None
            assert reranker_client is not None
            assert llm_client is not None

    def test_validate_configuration_success(self, mock_openai_client):
        """Test successful configuration validation."""
        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            config = MemoryConfig()

            result = config.validate_configuration()
            assert result is True

    def test_validate_configuration_failure(self, mock_openai_client):
        """Test configuration validation failure."""
        mock_openai_client.models.list.side_effect = Exception("API error")

        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            config = MemoryConfig()

            with pytest.raises(ConfigurationError) as exc_info:
                config.validate_configuration()

            assert "Configuration validation failed" in str(exc_info.value)


class TestConfigModule:
    """Test cases for module-level configuration functions."""

    def test_get_config_singleton(self, mock_openai_client):
        """Test that get_config returns the same instance."""
        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            reset_config()  # Ensure clean state

            config1 = get_config()
            config2 = get_config()

            assert config1 is config2

    def test_reset_config(self, mock_openai_client):
        """Test that reset_config creates a new instance."""
        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            config1 = get_config()
            reset_config()
            config2 = get_config()

            assert config1 is not config2

    def test_get_config_with_missing_env(self, monkeypatch):
        """Test get_config with missing environment variables."""
        reset_config()

        # Mock os.getenv to return None for required variables
        original_getenv = os.getenv

        def mock_getenv(key, default=None):
            if key == "GITEE_AI_EMBEDDING_API_KEY":
                return None
            return original_getenv(key, default)

        monkeypatch.setattr(os, "getenv", mock_getenv)

        with pytest.raises(ConfigurationError):
            get_config()
