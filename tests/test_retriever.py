"""
Test suite for the retrieval layer.
Tests hybrid search and reranking functionality.
"""

import json
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

    def test_keyword_extraction_chinese_text(self, retriever):
        """Test keyword extraction with Chinese text."""
        query = "如何登录应用程序"

        keywords = retriever._extract_query_keywords(query)

        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should contain meaningful Chinese words
        assert any(len(word) > 1 for word in keywords)

    def test_keyword_extraction_mixed_languages(self, retriever):
        """Test keyword extraction with mixed languages."""
        query = "How to 登录 the application"

        keywords = retriever._extract_query_keywords(query)

        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should handle both English and Chinese
        assert "application" in keywords or "登录" in keywords

    def test_keyword_extraction_special_characters(self, retriever):
        """Test keyword extraction with special characters."""
        query = "How to login? User@domain.com with password!"

        keywords = retriever._extract_query_keywords(query)

        assert isinstance(keywords, list)
        assert "login" in keywords
        # Should filter out common stop words
        assert "to" not in keywords
        assert "the" not in keywords
        assert "with" not in keywords

    @patch("gui_agent_memory.retriever.requests.post")
    def test_rerank_results_success(self, mock_post, retriever, mock_config):
        """Test successful reranking of results."""
        # Setup
        candidates = [
            {"content": "Login to app", "metadata": {"score": 0.8}},
            {"content": "Navigate settings", "metadata": {"score": 0.6}},
        ]
        query = "How to login"

        # Mock successful reranker response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.3},
            ]
        }
        mock_post.return_value = mock_response

        # Mock config
        mock_config.get_reranker_config.return_value = {
            "base_url": "https://test-reranker.com",
            "api_key": "test-key",
            "model": "test-model",
        }

        # Execute
        result = retriever._rerank_results(query, candidates)

        # Verify
        assert len(result) == 2
        assert result[0]["content"] == "Login to app"  # Higher relevance score
        assert result[1]["content"] == "Navigate settings"

    @patch("gui_agent_memory.retriever.requests.post")
    def test_rerank_results_api_failure(self, mock_post, retriever, mock_config):
        """Test reranking fallback when API fails."""
        candidates = [
            {"content": "Login to app", "metadata": {"score": 0.8}},
            {"content": "Navigate settings", "metadata": {"score": 0.6}},
        ]
        query = "How to login"

        # Mock API failure
        import requests
        mock_post.side_effect = requests.RequestException("API error")
        mock_config.get_reranker_config.return_value = {
            "base_url": "https://test-reranker.com",
            "api_key": "test-key",
            "model": "test-model",
        }

        # Execute - should not raise exception, should return original order
        result = retriever._rerank_results(query, candidates)

        # Verify fallback behavior
        assert len(result) == 2
        assert result == candidates  # Original order preserved

    @patch("gui_agent_memory.retriever.requests.post")
    def test_rerank_results_malformed_response(self, mock_post, retriever, mock_config):
        """Test reranking with malformed API response."""
        candidates = [{"content": "Login to app", "metadata": {"score": 0.8}}]
        query = "How to login"

        # Mock malformed response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "format"}
        mock_post.return_value = mock_response

        mock_config.get_reranker_config.return_value = {
            "base_url": "https://test-reranker.com",
            "api_key": "test-key",
            "model": "test-model",
        }

        # Execute - should fallback gracefully
        result = retriever._rerank_results(query, candidates)

        # Verify fallback
        assert result == candidates

    def test_retrieve_memories_comprehensive(
        self, retriever, mock_config, mock_storage
    ):
        """Test comprehensive memory retrieval workflow."""
        query = "How to login to application"

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage responses
        mock_storage.query_experiences.return_value = {
            "ids": [["exp_1"]],
            "documents": [["Login process"]],
            "metadatas": [
                [
                    {
                        "task_description": "User login",
                        "keywords": ["login", "user"],
                        "is_successful": True,
                        "source_task_id": "task_1",
                        "preconditions": "App open",
                        "action_flow": json.dumps(
                            [
                                {
                                    "thought": "Click login",
                                    "action": "click",
                                    "target_element_description": "login button",
                                }
                            ]
                        ),
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        mock_storage.query_facts.return_value = {
            "ids": [["fact_1"]],
            "documents": [["Login requires credentials"]],
            "metadatas": [
                [
                    {
                        "content": "Login requires valid credentials",
                        "keywords": ["login", "credentials"],
                        "source": "manual",
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        # Mock reranking (disable for simplicity) - fixed parameter order
        with patch.object(retriever, "_rerank_results", side_effect=lambda q, x, top_n=10: x):
            result = retriever.retrieve_memories(query, top_n=1)

        # Verify comprehensive result
        assert result is not None
        assert hasattr(result, "experiences")
        assert hasattr(result, "facts")
        assert hasattr(result, "query")
        assert result.query == query

    def test_retrieve_memories_no_results(self, retriever, mock_config, mock_storage):
        """Test retrieval when no results are found."""
        query = "Nonexistent functionality"

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock empty storage responses
        mock_storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = retriever.retrieve_memories(query)

        # Should return empty but valid result
        assert result is not None
        assert len(result.experiences) == 0
        assert len(result.facts) == 0
        assert result.total_results == 0

    def test_retrieve_memories_empty_query(self, retriever, mock_config, mock_storage):
        """Test retrieval with empty query."""
        query = ""

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock empty storage responses
        mock_storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # Should handle empty query gracefully or raise appropriate error
        try:
            result = retriever.retrieve_memories(query)
            # If successful, result should be valid
            assert result is not None
        except ValueError:
            # Acceptable to raise ValueError for empty query
            pass

    def test_retrieve_memories_zero_top_n(self, retriever, mock_config, mock_storage):
        """Test retrieval with top_n=0."""
        query = "Test query"

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock empty storage responses
        mock_storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = retriever.retrieve_memories(query, top_n=0)

        # Should return empty results
        assert result is not None
        assert len(result.experiences) == 0
        assert len(result.facts) == 0

    def test_retrieve_memories_negative_top_n(self, retriever, mock_config, mock_storage):
        """Test retrieval with negative top_n."""
        query = "Test query"

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock empty storage responses
        mock_storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # Should handle negative values gracefully (likely convert to default)
        # Implementation may vary, but should not crash
        try:
            result = retriever.retrieve_memories(query, top_n=-5)
            # If it doesn't raise an exception, verify result is reasonable
            assert result is not None
        except ValueError:
            # Acceptable to raise ValueError for negative values
            pass
