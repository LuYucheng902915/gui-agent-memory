"""
Test suite for the retrieval layer.
Tests hybrid search and reranking functionality.
"""

from unittest.mock import Mock, patch

import pytest

from gui_agent_memory.retriever import MemoryRetriever


class TestMemoryRetriever:
    """Test cases for MemoryRetriever class."""

    @pytest.fixture
    def mock_storage(self):
        """Mock storage for testing."""
        storage = Mock()
        return storage

    @pytest.fixture
    def retriever(self, mock_config, mock_storage):
        """Create MemoryRetriever instance with mocked dependencies."""
        with (
            patch("gui_agent_memory.retriever.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.retriever.MemoryStorage", return_value=mock_storage
            ),
        ):
            return MemoryRetriever()

    @pytest.fixture
    def sample_experiences(self):
        """Sample experience records for testing."""
        return [
            {
                "id": "exp_1",
                "metadata": {
                    "task_description": "Login to application",
                    "keywords": ["login", "authentication"],
                    "is_successful": True,
                    "source_task_id": "task_1",
                },
                "distance": 0.2,
            },
            {
                "id": "exp_2",
                "metadata": {
                    "task_description": "Navigate to settings",
                    "keywords": ["settings", "navigation"],
                    "is_successful": True,
                    "source_task_id": "task_2",
                },
                "distance": 0.3,
            },
        ]

    @pytest.mark.unit
    def test_init(self, mock_config, mock_storage):
        """Test MemoryRetriever initialization."""
        with (
            patch("gui_agent_memory.retriever.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.retriever.MemoryStorage", return_value=mock_storage
            ),
        ):
            retriever = MemoryRetriever()
            assert retriever.storage == mock_storage
            assert retriever.config == mock_config

    @pytest.mark.unit
    def test_generate_query_embedding(self, retriever, mock_config):
        """Test query embedding generation."""
        # Arrange
        query = "How to login to the application"
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = mock_response

        # Act
        result = retriever._generate_query_embedding(query)

        # Assert
        assert result == [0.1, 0.2, 0.3]
        mock_config.embedding_client.embeddings.create.assert_called_once()

    @pytest.mark.unit
    def test_extract_keywords(self, retriever):
        """Test keyword extraction from query."""
        # Arrange
        query = "How to login to the application"

        # Act
        result = retriever._extract_keywords(query)

        # Assert
        assert isinstance(result, list)
        assert "login" in result
        assert "application" in result
        assert "how" in result
        # Stop words should be filtered out
        assert "to" not in result
        assert "the" not in result

    @pytest.mark.unit
    def test_retrieve_memories(self, retriever, mock_config, mock_storage):
        """Test memory retrieval."""
        # Arrange
        query = "How to login to application"
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_response
        )

        # Mock storage query results
        mock_storage.query_experiences.return_value = {
            "ids": [["exp_1"]],
            "documents": [["Test experience"]],
            "metadatas": [
                [
                    {
                        "keywords": ["test"],
                        "is_successful": True,
                        "source_task_id": "test_1",
                        "preconditions": "None",
                        "action_flow": "[]",
                    }
                ]
            ],
            "distances": [[0.1]],
        }
        mock_storage.query_facts.return_value = {
            "ids": [["fact_1"]],
            "documents": [["Test fact"]],
            "metadatas": [[{"keywords": ["test"], "source": "manual"}]],
            "distances": [[0.1]],
        }

        # Mock reranker response
        mock_rerank_response = Mock()
        mock_rerank_response.choices = [Mock()]
        mock_rerank_response.choices[0].message.content = "[0]"
        mock_config.get_reranker_client.return_value.chat.completions.create.return_value = (
            mock_rerank_response
        )

        # Act
        result = retriever.retrieve_memories(query, top_n=1)

        # Assert
        assert result is not None
        assert hasattr(result, "experiences")
        assert hasattr(result, "facts")
