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
        return Mock()

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
        mock_config.get_reranker_client.return_value.chat.completions.create.return_value = mock_rerank_response

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
        mock_config.reranker_llm_base_url = "https://test-reranker.com"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

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
        mock_config.reranker_llm_base_url = "https://test-reranker.com"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

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

        mock_config.reranker_llm_base_url = "https://test-reranker.com"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

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
        with patch.object(
            retriever,
            "_rerank_results",
            side_effect=lambda _q, x, top_n=10: x,
        ):
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

    def test_retrieve_memories_negative_top_n(
        self, retriever, mock_config, mock_storage
    ):
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


class TestMemoryRetrieverAdvanced:
    """Advanced test cases for MemoryRetriever error handling and edge cases."""

    @pytest.fixture
    def mock_storage(self):
        """Mock storage for testing."""
        storage = Mock()
        # Default empty responses
        storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
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

    def test_keyword_filter_experiences_empty_keywords(self, retriever):
        """Test keyword filtering with empty keywords list."""
        result = retriever._keyword_filter_experiences([])

        assert result == {}

    def test_keyword_filter_facts_empty_keywords(self, retriever):
        """Test keyword filtering with empty keywords list."""
        result = retriever._keyword_filter_facts([])

        assert result == {}

    def test_vector_search_experiences_storage_error(self, retriever, mock_config):
        """Test vector search when storage operation fails."""
        from gui_agent_memory.retriever import RetrievalError
        from gui_agent_memory.storage import StorageError

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to raise an error
        retriever.storage.query_experiences.side_effect = StorageError("Storage failed")

        with pytest.raises(RetrievalError) as exc_info:
            retriever._vector_search_experiences([0.1, 0.2, 0.3], top_k=10)

        assert "Failed to perform vector search on experiences" in str(exc_info.value)

    def test_vector_search_facts_storage_error(self, retriever, mock_config):
        """Test vector search when storage operation fails."""
        from gui_agent_memory.retriever import RetrievalError
        from gui_agent_memory.storage import StorageError

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to raise an error
        retriever.storage.query_facts.side_effect = StorageError("Storage failed")

        with pytest.raises(RetrievalError) as exc_info:
            retriever._vector_search_facts([0.1, 0.2, 0.3], top_k=10)

        assert "Failed to perform vector search on facts" in str(exc_info.value)

    def test_vector_search_experiences_malformed_results(self, retriever):
        """Test vector search with malformed storage results."""
        from gui_agent_memory.retriever import RetrievalError

        # Mock storage to return malformed results
        retriever.storage.query_experiences.return_value = {
            "ids": [["exp1", "exp2"]],
            "documents": [["doc1"]],  # Mismatched lengths
            "metadatas": [["meta1", "meta2"]],
            "distances": [[0.1, 0.2]],
        }

        with pytest.raises(RetrievalError) as exc_info:
            retriever._vector_search_experiences([0.1, 0.2, 0.3], top_k=10)

        assert "Malformed storage results: inconsistent array lengths" in str(
            exc_info.value
        )

    def test_vector_search_facts_malformed_results(self, retriever):
        """Test vector search with malformed storage results."""
        from gui_agent_memory.retriever import RetrievalError

        # Mock storage to return malformed results
        retriever.storage.query_facts.return_value = {
            "ids": [["fact1", "fact2"]],
            "documents": [["doc1"]],  # Mismatched lengths
            "metadatas": [["meta1", "meta2"]],
            "distances": [[0.1, 0.2]],
        }

        with pytest.raises(RetrievalError) as exc_info:
            retriever._vector_search_facts([0.1, 0.2, 0.3], top_k=10)

        assert "Malformed storage results: inconsistent array lengths" in str(
            exc_info.value
        )

    def test_rerank_results_empty_candidates(self, retriever, mock_config):
        """Test reranking with empty candidate list."""
        result = retriever._rerank_results("test query", [])

        assert result == []

    def test_rerank_results_single_candidate(self, retriever, mock_config):
        """Test reranking with single candidate."""
        candidates = [{"id": "test1", "content": "test content", "type": "experience"}]

        result = retriever._rerank_results("test query", candidates)

        # Should return the single candidate as-is
        assert len(result) == 1
        assert result[0] == candidates[0]

    def test_rerank_results_reranker_api_error(self, retriever, mock_config):
        """Test reranking when reranker API fails."""
        candidates = [
            {"id": "test1", "content": "content 1", "type": "experience"},
            {"id": "test2", "content": "content 2", "type": "fact"},
        ]

        # Mock reranker client to raise an exception
        mock_client = Mock()
        mock_client.post.side_effect = Exception("Reranker API Error")
        mock_config.get_reranker_client.return_value = mock_client

        # Should fall back to original order when reranking fails
        result = retriever._rerank_results("test query", candidates)

        assert len(result) == 2
        assert result == candidates  # Should return in original order

    def test_rerank_results_malformed_response(self, retriever, mock_config):
        """Test reranking with malformed API response."""
        candidates = [
            {"id": "test1", "content": "content 1", "type": "experience"},
            {"id": "test2", "content": "content 2", "type": "fact"},
        ]

        # Mock reranker response with malformed data
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"index": 0},  # Missing 'relevance_score'
                {"relevance_score": 0.8},  # Missing 'index'
            ]
        }
        mock_response.raise_for_status.return_value = None

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_config.get_reranker_client.return_value = mock_client

        # Should fall back to original order when reranking response is malformed
        result = retriever._rerank_results("test query", candidates)

        assert len(result) == 2
        assert result == candidates  # Should return in original order

    def test_rerank_results_http_error(self, retriever, mock_config):
        """Test reranking when HTTP error occurs - should fallback gracefully."""
        candidates = [
            {"id": "test1", "content": "content 1", "type": "experience"},
            {"id": "test2", "content": "content 2", "type": "fact"},
        ]

        # Mock config for reranker
        mock_config.reranker_llm_base_url = "https://test-reranker.com"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

        # Mock requests.post to raise an HTTP error
        with patch("gui_agent_memory.retriever.requests.post") as mock_post:
            import requests

            mock_post.side_effect = requests.RequestException("HTTP error")

            # Should fall back to original results, not raise an error
            result = retriever._rerank_results("test query", candidates)

            # Should return original candidates
            assert len(result) == 2
            assert result == candidates

    def test_merge_search_results_experiences_only(self, retriever):
        """Test merging search results with only experiences."""
        experience_results = [
            {"id": "exp1", "score": 0.9, "type": "experience"},
            {"id": "exp2", "score": 0.8, "type": "experience"},
        ]
        fact_results: list[dict] = []

        result = retriever._merge_search_results(
            experience_results, fact_results, limit=10
        )

        assert len(result) == 2
        assert all(item["type"] == "experience" for item in result)

    def test_merge_search_results_facts_only(self, retriever):
        """Test merging search results with only facts."""
        experience_results: list[dict] = []
        fact_results = [
            {"id": "fact1", "score": 0.9, "type": "fact"},
            {"id": "fact2", "score": 0.8, "type": "fact"},
        ]

        result = retriever._merge_search_results(
            experience_results, fact_results, limit=10
        )

        assert len(result) == 2
        assert all(item["type"] == "fact" for item in result)

    def test_merge_search_results_with_limit(self, retriever):
        """Test merging search results with limit applied."""
        experience_results = [
            {"id": "exp1", "score": 0.9, "type": "experience"},
            {"id": "exp2", "score": 0.8, "type": "experience"},
            {"id": "exp3", "score": 0.7, "type": "experience"},
        ]
        fact_results = [
            {"id": "fact1", "score": 0.85, "type": "fact"},
            {"id": "fact2", "score": 0.75, "type": "fact"},
        ]

        result = retriever._merge_search_results(
            experience_results, fact_results, limit=3
        )

        # Should be limited to 3 results
        assert len(result) == 3
        # Should be sorted by score (highest first)
        scores = [item["score"] for item in result]
        assert scores == sorted(scores, reverse=True)

    def test_convert_to_memory_objects_experiences(self, retriever):
        """Test converting search results to ExperienceRecord objects."""
        from gui_agent_memory.models import ExperienceRecord

        search_results = [
            {
                "id": "exp1",
                "document": "Test task",  # This is the task_description
                "metadata": {
                    "keywords": "test,task",
                    "action_flow": '[{"thought": "test", "action": "click", "target_element_description": "button"}]',
                    "preconditions": "Test preconditions",
                    "is_successful": True,
                    "usage_count": 0,
                    "last_used_at": "2024-01-01T00:00:00Z",
                    "source_task_id": "task_123",
                },
                "type": "experience",
            }
        ]

        experiences, facts = retriever._convert_to_memory_objects(search_results)

        assert len(experiences) == 1
        assert len(facts) == 0
        assert isinstance(experiences[0], ExperienceRecord)
        assert experiences[0].task_description == "Test task"
        assert experiences[0].keywords == ["test", "task"]

    def test_convert_to_memory_objects_facts(self, retriever):
        """Test converting search results to FactRecord objects."""
        from gui_agent_memory.models import FactRecord

        search_results = [
            {
                "id": "fact1",
                "document": "Test fact content",  # This is the content
                "metadata": {
                    "keywords": "test,fact",
                    "source": "test_source",
                    "usage_count": 0,
                    "last_used_at": "2024-01-01T00:00:00Z",
                },
                "type": "fact",
            }
        ]

        experiences, facts = retriever._convert_to_memory_objects(search_results)

        assert len(experiences) == 0
        assert len(facts) == 1
        assert isinstance(facts[0], FactRecord)
        assert facts[0].content == "Test fact content"
        assert facts[0].keywords == ["test", "fact"]

    def test_convert_to_memory_objects_invalid_action_flow(self, retriever):
        """Test converting search results with invalid action_flow JSON."""
        from gui_agent_memory.retriever import RetrievalError

        search_results = [
            {
                "id": "exp1",
                "document": "Test task",
                "metadata": {
                    "keywords": "test,task",
                    "action_flow": "invalid json",  # Invalid JSON
                    "preconditions": "Test preconditions",
                    "is_successful": True,
                    "usage_count": 0,
                    "last_used_at": "2024-01-01T00:00:00Z",
                    "source_task_id": "task_123",
                },
                "type": "experience",
            }
        ]

        with pytest.raises(RetrievalError) as exc_info:
            retriever._convert_to_memory_objects(search_results)

        assert "Failed to parse action_flow JSON" in str(exc_info.value)

    def test_update_usage_stats_storage_error(self, retriever):
        """Test usage stats update when storage operation fails."""
        from gui_agent_memory.storage import StorageError

        memories = ["exp1", "fact1"]

        # Mock storage to raise an error
        retriever.storage.update_usage_stats.side_effect = StorageError(
            "Storage failed"
        )

        # Should handle error gracefully and log it
        with patch("gui_agent_memory.retriever.logger") as mock_logger:
            retriever._update_usage_stats(memories)

            # Should log the error but not raise exception
            mock_logger.error.assert_called()

    def test_get_similar_experiences_embedding_error(self, retriever, mock_config):
        """Test get_similar_experiences when embedding generation fails."""
        from gui_agent_memory.retriever import RetrievalError

        # Mock embedding client to raise an exception
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Embedding API Error")
        mock_config.get_embedding_client.return_value = mock_client

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_similar_experiences("test task", top_n=5)

        assert "Failed to generate query embedding" in str(exc_info.value)

    def test_get_related_facts_embedding_error(self, retriever, mock_config):
        """Test get_related_facts when embedding generation fails."""
        from gui_agent_memory.retriever import RetrievalError

        # Mock embedding client to raise an exception
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Embedding API Error")
        mock_config.get_embedding_client.return_value = mock_client

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_related_facts("test topic", top_n=5)

        assert "Failed to generate query embedding" in str(exc_info.value)

    def test_get_similar_experiences_storage_error(self, retriever, mock_config):
        """Test get_similar_experiences when storage query fails."""
        from gui_agent_memory.retriever import RetrievalError
        from gui_agent_memory.storage import StorageError

        # Mock successful embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to raise an error
        retriever.storage.query_experiences.side_effect = StorageError("Storage failed")

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_similar_experiences("test task", top_n=5)

        assert "Failed to perform vector search on experiences" in str(exc_info.value)

    def test_get_related_facts_storage_error(self, retriever, mock_config):
        """Test get_related_facts when storage query fails."""
        from gui_agent_memory.retriever import RetrievalError
        from gui_agent_memory.storage import StorageError

        # Mock successful embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to raise an error
        retriever.storage.query_facts.side_effect = StorageError("Storage failed")

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_related_facts("test topic", top_n=5)

        assert "Failed to perform vector search on facts" in str(exc_info.value)

    def test_get_similar_experiences_conversion_error(self, retriever, mock_config):
        """Test get_similar_experiences when object conversion fails."""
        from gui_agent_memory.retriever import RetrievalError

        # Mock successful embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to return malformed experience data
        retriever.storage.query_experiences.return_value = {
            "ids": [["exp1"]],
            "documents": [["doc1"]],
            "metadatas": [
                [
                    {
                        "task_description": "Test task",
                        "keywords": "test",
                        "action_flow": "invalid json",  # This will cause conversion to fail
                        "preconditions": "Test",
                        "is_successful": True,
                        "usage_count": 0,
                        "last_used_at": "2024-01-01T00:00:00Z",
                        "source_task_id": "task_123",
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_similar_experiences("test task", top_n=5)

        assert "Failed to convert experiences" in str(exc_info.value)

    def test_get_related_facts_conversion_error(self, retriever, mock_config):
        """Test get_related_facts when object conversion fails."""
        from gui_agent_memory.retriever import RetrievalError

        # Mock successful embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to return fact data that will cause conversion issues
        # We'll simulate this by patching the conversion method
        retriever.storage.query_facts.return_value = {
            "ids": [["fact1"]],
            "documents": [["doc1"]],
            "metadatas": [
                [
                    {
                        "content": "Test fact",
                        "keywords": "test",
                        "source": "test",
                        "usage_count": 0,
                        "last_used_at": "2024-01-01T00:00:00Z",
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        # Mock the conversion method to raise an exception
        with patch.object(
            retriever,
            "_convert_to_memory_objects",
            side_effect=Exception("Conversion failed"),
        ):
            with pytest.raises(RetrievalError) as exc_info:
                retriever.get_related_facts("test topic", top_n=5)

            assert "Failed to convert facts" in str(exc_info.value)
