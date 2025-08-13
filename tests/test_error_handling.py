"""
Comprehensive tests for error handling and exception scenarios.

Tests all custom exception classes and error conditions across modules.
"""

from unittest.mock import Mock, mock_open, patch

import pytest

from gui_agent_memory.config import ConfigurationError
from gui_agent_memory.ingestion import IngestionError, MemoryIngestion
from gui_agent_memory.main import MemorySystem, MemorySystemError
from gui_agent_memory.retriever import MemoryRetriever, RetrievalError
from gui_agent_memory.storage import MemoryStorage, StorageError


class TestCustomExceptions:
    """Test all custom exception classes."""

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError is properly defined."""
        error = ConfigurationError("Test config error")
        assert isinstance(error, Exception)
        assert str(error) == "Test config error"

    def test_storage_error_inheritance(self):
        """Test StorageError is properly defined."""
        error = StorageError("Test storage error")
        assert isinstance(error, Exception)
        assert str(error) == "Test storage error"

    def test_ingestion_error_inheritance(self):
        """Test IngestionError is properly defined."""
        error = IngestionError("Test ingestion error")
        assert isinstance(error, Exception)
        assert str(error) == "Test ingestion error"

    def test_retrieval_error_inheritance(self):
        """Test RetrievalError is properly defined."""
        error = RetrievalError("Test retrieval error")
        assert isinstance(error, Exception)
        assert str(error) == "Test retrieval error"

    def test_memory_system_error_inheritance(self):
        """Test MemorySystemError is properly defined."""
        error = MemorySystemError("Test system error")
        assert isinstance(error, Exception)
        assert str(error) == "Test system error"


class TestStorageErrorHandling:
    """Test error handling in storage layer."""

    @patch("gui_agent_memory.storage.chromadb")
    def test_chromadb_connection_failure(self, mock_chromadb):
        """Test ChromaDB connection failure handling."""
        mock_chromadb.PersistentClient.side_effect = Exception("Connection failed")

        with pytest.raises(StorageError) as exc_info:
            MemoryStorage()

        assert "Failed to initialize ChromaDB client" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_collection_creation_failure(self, mock_chromadb):
        """Test collection creation failure handling."""
        mock_client = Mock()
        mock_client.get_or_create_collection.side_effect = Exception(
            "Collection creation failed"
        )
        mock_chromadb.PersistentClient.return_value = mock_client

        with pytest.raises(StorageError) as exc_info:
            MemoryStorage()

        assert "Failed to initialize collections" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_experiences_chromadb_error(
        self, mock_chromadb, sample_experience_record
    ):
        """Test error handling when ChromaDB add fails."""
        mock_collection = Mock()
        mock_collection.add.side_effect = Exception("ChromaDB add failed")

        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3]]

        with pytest.raises(StorageError) as exc_info:
            storage.add_experiences([sample_experience_record], embeddings)

        assert "Failed to add experiences" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_experiences_chromadb_error(self, mock_chromadb):
        """Test error handling when ChromaDB query fails."""
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("ChromaDB query failed")

        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        query_embedding = [0.1, 0.2, 0.3]

        with pytest.raises(StorageError) as exc_info:
            storage.query_experiences(query_embeddings=[query_embedding], n_results=5)

        assert "Failed to query experiences" in str(exc_info.value)


class TestIngestionErrorHandling:
    """Test error handling in ingestion layer."""

    @patch("gui_agent_memory.ingestion.get_config")
    @patch("gui_agent_memory.ingestion.MemoryStorage")
    def test_embedding_generation_failure(self, mock_storage_class, mock_get_config):
        """Test error handling when embedding generation fails."""
        mock_config = Mock()
        mock_config.get_embedding_client.return_value.embeddings.create.side_effect = (
            Exception("API error")
        )
        mock_config.logs_base_dir = "./test_data/test_logs"
        mock_config.prompt_templates_dir = ""
        mock_get_config.return_value = mock_config

        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        ingestion = MemoryIngestion()

        with pytest.raises(IngestionError) as exc_info:
            ingestion._generate_embedding("test text")

        assert "Failed to generate embedding" in str(exc_info.value)

    @patch("gui_agent_memory.ingestion.get_config")
    @patch("gui_agent_memory.ingestion.MemoryStorage")
    def test_llm_experience_distillation_failure(
        self, mock_storage_class, mock_get_config
    ):
        """Test error handling when LLM experience distillation fails."""
        mock_config = Mock()
        mock_config.get_experience_llm_client.return_value.chat.completions.create.side_effect = Exception(
            "LLM API error"
        )
        mock_config.logs_base_dir = "./test_data/test_logs"
        mock_config.prompt_templates_dir = ""
        mock_get_config.return_value = mock_config

        mock_storage = Mock()
        mock_storage.experience_exists.return_value = (
            False  # Ensure it doesn't skip due to existing experience
        )
        mock_storage_class.return_value = mock_storage

        # Mock prompt loading
        with patch("builtins.open", mock_open(read_data="Mock prompt template")):
            ingestion = MemoryIngestion()

        with pytest.raises(IngestionError) as exc_info:
            ingestion.learn_from_task(
                raw_history=[{"action": "click", "target": "button"}],
                task_description="Test task",
                is_successful=True,
                source_task_id="test_123",
            )

        assert "Failed to learn from task" in str(exc_info.value)

    @patch("gui_agent_memory.ingestion.get_config")
    @patch("gui_agent_memory.ingestion.MemoryStorage")
    def test_invalid_json_response_from_llm(self, mock_storage_class, mock_get_config):
        """Test error handling when LLM returns invalid JSON."""
        mock_config = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Invalid JSON response"
        mock_config.get_experience_llm_client.return_value.chat.completions.create.return_value.choices = [
            mock_choice
        ]
        mock_config.logs_base_dir = "./test_data/test_logs"
        mock_config.prompt_templates_dir = ""
        mock_get_config.return_value = mock_config

        mock_storage = Mock()
        mock_storage.experience_exists.return_value = (
            False  # Ensure it doesn't skip due to existing experience
        )
        mock_storage_class.return_value = mock_storage

        # Mock prompt loading
        with patch("builtins.open", mock_open(read_data="Mock prompt template")):
            ingestion = MemoryIngestion()

        with pytest.raises(IngestionError) as exc_info:
            ingestion.learn_from_task(
                raw_history=[{"action": "click", "target": "button"}],
                task_description="Test task",
                is_successful=True,
                source_task_id="test_123",
            )

        assert "Failed to learn from task" in str(exc_info.value)


class TestRetrievalErrorHandling:
    """Test error handling in retrieval layer."""

    @patch("gui_agent_memory.retriever.get_config")
    @patch("gui_agent_memory.retriever.MemoryStorage")
    def test_embedding_generation_failure(self, mock_storage_class, mock_get_config):
        """Test error handling when query embedding generation fails."""
        mock_config = Mock()
        mock_config.get_embedding_client.return_value.embeddings.create.side_effect = (
            Exception("Embedding API error")
        )
        mock_get_config.return_value = mock_config

        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage

        retriever = MemoryRetriever()

        with pytest.raises(RetrievalError) as exc_info:
            retriever._generate_query_embedding("test query")

        assert "Failed to generate query embedding" in str(exc_info.value)

    @patch("gui_agent_memory.retriever.get_config")
    @patch("gui_agent_memory.retriever.MemoryStorage")
    def test_storage_query_failure(self, mock_storage_class, mock_get_config):
        """Test error handling when storage query fails."""
        mock_config = Mock()
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )
        mock_get_config.return_value = mock_config

        mock_storage = Mock()
        mock_storage.query_experiences.side_effect = Exception("Storage query failed")
        mock_storage_class.return_value = mock_storage

        retriever = MemoryRetriever()

        with pytest.raises(RetrievalError) as exc_info:
            retriever.retrieve_memories("test query")

        assert "Memory retrieval failed" in str(
            exc_info.value
        ) or "Failed to retrieve memories" in str(exc_info.value)


class TestMemorySystemErrorHandling:
    """Test error handling in main MemorySystem class."""

    @patch("gui_agent_memory.main.get_config")
    def test_system_initialization_config_failure(self, mock_get_config):
        """Test MemorySystem initialization failure due to config error."""
        mock_get_config.side_effect = ConfigurationError("Config failed")

        with pytest.raises(ConfigurationError):
            MemorySystem()

    @patch("gui_agent_memory.main.get_config")
    @patch("gui_agent_memory.main.MemoryStorage")
    def test_system_initialization_storage_failure(
        self, mock_storage_class, mock_get_config
    ):
        """Test MemorySystem initialization failure due to storage error."""
        mock_get_config.return_value = Mock()
        mock_storage_class.side_effect = Exception("Storage init failed")

        with pytest.raises(MemorySystemError) as exc_info:
            MemorySystem()

        assert "Failed to initialize memory system" in str(exc_info.value)

    def test_retrieve_memories_empty_query(self):
        """Test error handling for empty query."""
        with (
            patch("gui_agent_memory.main.get_config"),
            patch("gui_agent_memory.main.MemoryStorage"),
            patch("gui_agent_memory.main.MemoryIngestion"),
            patch("gui_agent_memory.main.MemoryRetriever"),
        ):
            system = MemorySystem()

            with pytest.raises(MemorySystemError) as exc_info:
                system.retrieve_memories("")

            assert "Query cannot be empty" in str(exc_info.value)

    def test_retrieve_memories_whitespace_only_query(self):
        """Test error handling for whitespace-only query."""
        with (
            patch("gui_agent_memory.main.get_config"),
            patch("gui_agent_memory.main.MemoryStorage"),
            patch("gui_agent_memory.main.MemoryIngestion"),
            patch("gui_agent_memory.main.MemoryRetriever"),
        ):
            system = MemorySystem()

            with pytest.raises(MemorySystemError) as exc_info:
                system.retrieve_memories("   \n\t  ")

            assert "Query cannot be empty" in str(exc_info.value)


class TestEnvironmentErrorHandling:
    """Test error handling related to environment and configuration."""

    def test_missing_env_file_handling(self, monkeypatch):
        """Test graceful handling when .env file is missing."""
        # This should not raise an error as dotenv handles missing files gracefully
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_LLM_BASE_URL", "https://test.com")
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_LLM_BASE_URL", "https://test.com")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test.com")

        from gui_agent_memory.config import MemoryConfig

        with patch("gui_agent_memory.config.OpenAI"):
            config = MemoryConfig()
            assert config is not None

    def test_invalid_numeric_env_values(self, monkeypatch):
        """Test handling of invalid numeric environment values (pydantic)."""
        monkeypatch.setenv("DEFAULT_TOP_K", "not_a_number")
        monkeypatch.setenv("EMBEDDING_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_LLM_BASE_URL", "https://test.com")
        monkeypatch.setenv("RERANKER_LLM_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_LLM_BASE_URL", "https://test.com")
        monkeypatch.setenv("EXPERIENCE_LLM_API_KEY", "test-key")
        monkeypatch.setenv("EXPERIENCE_LLM_BASE_URL", "https://test.com")

        from gui_agent_memory.config import ConfigurationError, MemoryConfig

        with (
            patch("gui_agent_memory.config.OpenAI"),
            pytest.raises(ConfigurationError),
        ):
            # pydantic validation should raise ConfigurationError wrapping ValidationError
            MemoryConfig()
