"""
Unit tests for the configuration module.
"""

from unittest.mock import ANY, Mock, patch

import pytest
import requests

from gui_agent_memory.config import (
    ConfigurationError,
    MemoryConfig,
    get_config,
    reset_config,
)


class TestMemoryConfig:
    """Test cases for MemoryConfig class."""

    def test_config_initialization_success(self, mock_openai_client, monkeypatch):
        """Test successful configuration initialization."""
        # Set up the test environment variables
        monkeypatch.setenv(
            "EMBEDDING_LLM_BASE_URL", "https://test-embedding.example.com/v1"
        )
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-fake-embedding-key-12345")
        monkeypatch.setenv(
            "RERANKER_LLM_BASE_URL", "https://test-reranker.example.com/v1/rerank"
        )
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-fake-reranker-key-67890")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test-llm.example.com/v1")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-fake-llm-key-abcdef")

        with patch("gui_agent_memory.config.OpenAI", return_value=mock_openai_client):
            config = MemoryConfig()

            # Test with FAKE test environment values - NOT real API keys!
            assert (
                str(config.embedding_llm_base_url)
                == "https://test-embedding.example.com/v1"
            )
            assert (
                config.embedding_llm_api_key.get_secret_value()
                == "test-fake-embedding-key-12345"
            )
            assert (
                str(config.reranker_llm_base_url)
                == "https://test-reranker.example.com/v1/rerank"
            )
            assert (
                config.reranker_llm_api_key.get_secret_value()
                == "test-fake-reranker-key-67890"
            )
            assert (
                str(config.experience_llm_base_url) == "https://test-llm.example.com/v1"
            )
            assert (
                config.experience_llm_api_key.get_secret_value()
                == "test-fake-llm-key-abcdef"
            )
            assert config.embedding_model == "Qwen3-Embedding-8B"
            assert config.reranker_model == "Qwen3-Reranker-8B"

    def test_config_missing_required_env_var(self, monkeypatch):
        """Test configuration failure with missing required environment variable."""
        # Ensure required env vars are absent
        for key in [
            "EMBEDDING_LLM_API_KEY",
            "EMBEDDING_LLM_BASE_URL",
            "RERANKER_LLM_API_KEY",
            "RERANKER_LLM_BASE_URL",
            "EXPERIENCE_LLM_API_KEY",
            "EXPERIENCE_LLM_BASE_URL",
        ]:
            monkeypatch.delenv(key, raising=False)

        from gui_agent_memory.config import reset_config

        reset_config()

        with pytest.raises(ConfigurationError) as exc_info:
            MemoryConfig()

        assert (
            "missing" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )

    def test_config_client_initialization_failure(self, monkeypatch):
        """Test configuration failure during client initialization."""
        # Set required env vars
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_LLM_BASE_URL", "https://ai.gitee.com/v1")
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_LLM_BASE_URL", "https://ai.gitee.com/v1/rerank")
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

    def test_config_default_values(self, monkeypatch, mock_openai_client):
        """Test that default configuration values are set correctly."""
        # Set required env vars for successful initialization
        monkeypatch.setenv(
            "EMBEDDING_LLM_BASE_URL", "https://test-embedding.example.com/v1"
        )
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-fake-embedding-key-12345")
        monkeypatch.setenv(
            "RERANKER_LLM_BASE_URL", "https://test-reranker.example.com/v1/rerank"
        )
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-fake-reranker-key-67890")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test-llm.example.com/v1")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-fake-llm-key-abcdef")
        # Ensure optional envs are unset to use defaults
        for key in [
            "EMBEDDING_MODEL",
            "RERANKER_MODEL",
            "EXPERIENCE_LLM_MODEL",
            "DEFAULT_TOP_K",
            "DEFAULT_TOP_N",
            "EMBEDDING_DIMENSION",
            "CHROMA_DB_PATH",
        ]:
            monkeypatch.delenv(key, raising=False)

        with patch("gui_agent_memory.config.load_dotenv"):
            with patch(
                "gui_agent_memory.config.OpenAI", return_value=mock_openai_client
            ):
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

        # Remove a required env var
        monkeypatch.delenv("EMBEDDING_LLM_API_KEY", raising=False)

        with pytest.raises(ConfigurationError):
            get_config()


class TestConfigCoverage:
    """Tests for uncovered code paths in config.py"""

    def test_validate_configuration_reranker_fallback_success(self, monkeypatch):
        """Test reranker validation fallback to HTTP request (lines 167-194)."""
        # Set up environment variables
        monkeypatch.setenv("EMBEDDING_LLM_BASE_URL", "https://test-embedding.com/v1")
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_LLM_BASE_URL", "https://test-reranker.com/v1")
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test-llm.com/v1")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")

        with patch("gui_agent_memory.config.OpenAI") as mock_openai:
            # Mock embedding and experience LLM clients to succeed
            mock_embedding_client = Mock()
            mock_embedding_client.models.list.return_value = Mock()

            mock_experience_client = Mock()
            mock_experience_client.models.list.return_value = Mock()

            # Mock reranker client to fail on models.list() but succeed on HTTP request
            mock_reranker_client = Mock()
            mock_reranker_client.models.list.side_effect = Exception(
                "models.list failed"
            )

            # Make the mock more flexible to handle multiple calls
            def openai_side_effect(*args, **kwargs):
                # Return the same clients for repeated calls
                if mock_openai.call_count <= 3:
                    # First three calls: embedding, reranker, experience
                    clients = [
                        mock_embedding_client,
                        mock_reranker_client,
                        mock_experience_client,
                    ]
                    return clients[(mock_openai.call_count - 1) % 3]
                else:
                    # Subsequent calls: return one of the existing clients
                    return mock_embedding_client

            mock_openai.side_effect = openai_side_effect

            # Mock successful HTTP response
            mock_response = Mock()
            mock_response.status_code = 200

            with patch("requests.post", return_value=mock_response) as mock_post:
                config = MemoryConfig()
                result = config.validate_configuration()

                assert result is True

                # Verify the HTTP request was made with correct parameters
                mock_post.assert_called_once_with(
                    "https://test-reranker.com/v1",
                    json={
                        "query": "test",
                        "documents": ["test document"],
                        "model": "Qwen3-Reranker-8B",
                    },
                    headers={
                        "X-Failover-Enabled": "true",
                        "Authorization": "Bearer test-key",
                        "Content-Type": "application/json",
                    },
                    timeout=ANY,
                )

    def test_validate_configuration_reranker_fallback_422_status(self, monkeypatch):
        """Test reranker validation with 422 status code (valid API)."""
        # Set up environment variables
        monkeypatch.setenv("EMBEDDING_LLM_BASE_URL", "https://test-embedding.com/v1")
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_LLM_BASE_URL", "https://test-reranker.com/v1")
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test-llm.com/v1")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")

        with patch("gui_agent_memory.config.OpenAI") as mock_openai:
            # Mock embedding and experience LLM clients to succeed
            mock_embedding_client = Mock()
            mock_embedding_client.models.list.return_value = Mock()

            mock_experience_client = Mock()
            mock_experience_client.models.list.return_value = Mock()

            # Mock reranker client to fail on models.list()
            mock_reranker_client = Mock()
            mock_reranker_client.models.list.side_effect = Exception(
                "models.list failed"
            )

            # Make the mock more flexible to handle multiple calls
            def openai_side_effect(*args, **kwargs):
                # Return the same clients for repeated calls
                if mock_openai.call_count <= 3:
                    # First three calls: embedding, reranker, experience
                    clients = [
                        mock_embedding_client,
                        mock_reranker_client,
                        mock_experience_client,
                    ]
                    return clients[(mock_openai.call_count - 1) % 3]
                else:
                    # Subsequent calls: return one of the existing clients
                    return mock_embedding_client

            mock_openai.side_effect = openai_side_effect

            # Mock HTTP response with 422 status (still valid)
            mock_response = Mock()
            mock_response.status_code = 422

            with patch("requests.post", return_value=mock_response):
                config = MemoryConfig()
                result = config.validate_configuration()

                assert result is True

    def test_validate_configuration_reranker_fallback_400_status(self, monkeypatch):
        """Test reranker validation with 400 status code (valid API)."""
        # Set up environment variables
        monkeypatch.setenv("EMBEDDING_LLM_BASE_URL", "https://test-embedding.com/v1")
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_LLM_BASE_URL", "https://test-reranker.com/v1")
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test-llm.com/v1")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")

        with patch("gui_agent_memory.config.OpenAI") as mock_openai:
            # Mock embedding and experience LLM clients to succeed
            mock_embedding_client = Mock()
            mock_embedding_client.models.list.return_value = Mock()

            mock_experience_client = Mock()
            mock_experience_client.models.list.return_value = Mock()

            # Mock reranker client to fail on models.list()
            mock_reranker_client = Mock()
            mock_reranker_client.models.list.side_effect = Exception(
                "models.list failed"
            )

            mock_openai.side_effect = [
                mock_embedding_client,
                mock_reranker_client,
                mock_experience_client,
            ]

            # Mock HTTP response with 400 status (still valid)
            mock_response = Mock()
            mock_response.status_code = 400

            with patch("requests.post", return_value=mock_response):
                config = MemoryConfig()
                result = config.validate_configuration()

                assert result is True

    def test_validate_configuration_reranker_fallback_http_error(self, monkeypatch):
        """Test reranker validation when HTTP request fails."""
        # Set up environment variables
        monkeypatch.setenv("EMBEDDING_LLM_BASE_URL", "https://test-embedding.com/v1")
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_LLM_BASE_URL", "https://test-reranker.com/v1")
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test-llm.com/v1")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")

        with patch("gui_agent_memory.config.OpenAI") as mock_openai:
            # Mock embedding client to succeed
            mock_embedding_client = Mock()
            mock_embedding_client.models.list.return_value = Mock()

            # Mock reranker client to fail on models.list()
            mock_reranker_client = Mock()
            mock_reranker_client.models.list.side_effect = Exception(
                "models.list failed"
            )

            # Mock experience client (won't be reached due to error)
            mock_experience_client = Mock()

            mock_openai.side_effect = [
                mock_embedding_client,
                mock_reranker_client,
                mock_experience_client,
            ]

            # Mock HTTP request to raise an exception
            with patch(
                "requests.post", side_effect=requests.RequestException("HTTP error")
            ):
                config = MemoryConfig()

                with pytest.raises(ConfigurationError) as exc_info:
                    config.validate_configuration()

                assert "Reranker API validation failed" in str(exc_info.value)

    def test_validate_configuration_reranker_fallback_bad_status(self, monkeypatch):
        """Test reranker validation with bad HTTP status code."""
        # Set up environment variables
        monkeypatch.setenv("EMBEDDING_LLM_BASE_URL", "https://test-embedding.com/v1")
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_LLM_BASE_URL", "https://test-reranker.com/v1")
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test-llm.com/v1")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")

        with patch("gui_agent_memory.config.OpenAI") as mock_openai:
            # Mock embedding client to succeed
            mock_embedding_client = Mock()
            mock_embedding_client.models.list.return_value = Mock()

            # Mock reranker client to fail on models.list()
            mock_reranker_client = Mock()
            mock_reranker_client.models.list.side_effect = Exception(
                "models.list failed"
            )

            # Mock experience client (won't be reached due to error)
            mock_experience_client = Mock()

            mock_openai.side_effect = [
                mock_embedding_client,
                mock_reranker_client,
                mock_experience_client,
            ]

            # Mock HTTP response with bad status code
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.HTTPError(
                "500 Server Error"
            )

            with patch("requests.post", return_value=mock_response):
                config = MemoryConfig()

                with pytest.raises(ConfigurationError) as exc_info:
                    config.validate_configuration()

                assert "Reranker API validation failed" in str(exc_info.value)
