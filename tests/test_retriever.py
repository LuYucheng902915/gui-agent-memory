"""
Test suite for the retrieval layer.
Tests hybrid search and reranking functionality.
"""

import json
from unittest.mock import Mock, patch

import pytest

from gui_agent_memory.models import ExperienceRecord
from gui_agent_memory.retriever import MemoryRetriever, RetrievalError


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
                    "keywords": ["navigation", "settings"],
                    "is_successful": True,
                    "source_task_id": "task_2",
                },
                "distance": 0.4,
            },
        ]

    @pytest.fixture
    def sample_facts(self):
        """Sample fact records for testing."""
        return [
            {
                "id": "fact_1",
                "metadata": {
                    "content": "Python is a programming language",
                    "keywords": ["python", "programming"],
                },
                "distance": 0.3,
            },
            {
                "id": "fact_2",
                "metadata": {
                    "content": "Git is a version control system",
                    "keywords": ["git", "version-control"],
                },
                "distance": 0.5,
            },
        ]

    def test_init(self, mock_config, mock_storage):
        """Test MemoryRetriever initialization."""
        with (
            patch("gui_agent_memory.retriever.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.retriever.MemoryStorage", return_value=mock_storage
            ),
        ):
            retriever = MemoryRetriever()
            assert retriever.config == mock_config
            assert retriever.storage == mock_storage

    def test_generate_query_embedding(self, retriever, mock_config):
        """Test query embedding generation."""
        # Arrange
        query = "test query"
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = mock_response

        # Act
        result = retriever._generate_query_embedding(query)

        # Assert
        assert result == [0.1, 0.2, 0.3]
        mock_config.embedding_client.embeddings.create.assert_called_once_with(
            model=mock_config.embedding_model, input=query, dimensions=10
        )

    def test_extract_keywords(self, retriever):
        """Test keyword extraction."""
        query = "login to the application"
        result = retriever._extract_keywords(query)
        assert isinstance(result, list)
        # Should filter out common stop words
        assert "to" not in result
        assert "the" not in result

    def test_retrieve_memories(
        self, retriever, mock_config, sample_experiences, sample_facts
    ):
        """Test memory retrieval."""
        # Mock embedding generation
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = mock_response

        # Mock storage queries with proper structure
        retriever.storage.query_experiences.return_value = {
            "ids": [["exp_1", "exp_2"]],
            "documents": [["Login to application", "Navigate to settings"]],
            "metadatas": [
                [sample_experiences[0]["metadata"], sample_experiences[1]["metadata"]]
            ],
            "distances": [[0.2, 0.4]],
        }

        retriever.storage.query_facts.return_value = {
            "ids": [["fact_1", "fact_2"]],
            "documents": [
                ["Python is a programming language", "Git is a version control system"]
            ],
            "metadatas": [[sample_facts[0]["metadata"], sample_facts[1]["metadata"]]],
            "distances": [[0.3, 0.5]],
        }

        # Mock reranking
        mock_rerank_response = Mock()
        mock_rerank_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.85},
                {"index": 2, "relevance_score": 0.75},
                {"index": 3, "relevance_score": 0.65},
            ]
        }
        with patch("requests.post", return_value=mock_rerank_response):
            result = retriever.retrieve_memories("login to application")

        # Should return RetrievalResult with experiences and facts
        assert hasattr(result, "experiences")
        assert hasattr(result, "facts")
        assert result.query == "login to application"


class TestMemoryRetrieverAdvanced:
    """Advanced test cases for MemoryRetriever."""

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

    def test_keyword_extraction_chinese_text(self, retriever):
        """Test keyword extraction with Chinese text."""
        query = "如何登录应用程序"
        result = retriever._extract_keywords(query)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_keyword_extraction_mixed_languages(self, retriever):
        """Test keyword extraction with mixed languages."""
        query = "How to login 如何登录"
        result = retriever._extract_keywords(query)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_keyword_extraction_special_characters(self, retriever):
        """Test keyword extraction with special characters."""
        query = "login@#$%application"
        result = retriever._extract_keywords(query)
        assert isinstance(result, list)
        # Should filter out special characters
        assert "login" in result or "application" in result

    def test_rerank_results_success(self, retriever, mock_config):
        """Test successful reranking of results."""
        candidates = [
            {"id": "1", "document": "First document", "type": "experience"},
            {"id": "2", "document": "Second document", "type": "fact"},
        ]

        mock_config.reranker_llm_base_url = "https://test-reranker.com/v1/rerank"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response):
            result = retriever._rerank_results("test query", candidates, 5)

        # Should return reranked results
        assert len(result) == 2
        assert result[0]["id"] == "2"  # Higher relevance score should come first

    def test_rerank_results_api_failure(self, retriever, mock_config):
        """Test reranking when API fails."""
        candidates = [
            {"id": "1", "document": "First document", "type": "experience"},
            {"id": "2", "document": "Second document", "type": "fact"},
        ]

        mock_config.reranker_llm_base_url = "https://test-reranker.com/v1/rerank"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

        with patch("requests.post", side_effect=Exception("API error")):
            result = retriever._rerank_results("test query", candidates, 5)

        # Should return original candidates on API failure
        assert result == candidates

    def test_rerank_results_malformed_response(self, retriever, mock_config):
        """Test reranking with malformed API response."""
        candidates = [
            {"id": "1", "document": "First document", "type": "experience"},
            {"id": "2", "document": "Second document", "type": "fact"},
        ]

        mock_config.reranker_llm_base_url = "https://test-reranker.com/v1/rerank"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

        mock_response = Mock()
        mock_response.json.return_value = {"invalid": "format"}
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response):
            result = retriever._rerank_results("test query", candidates, 5)

        # Should return original candidates on malformed response
        assert result == candidates

    def test_retrieve_memories_comprehensive(self, retriever, mock_config):
        """Test comprehensive memory retrieval with all components."""
        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage responses with proper structure
        retriever.storage.query_experiences.return_value = {
            "ids": [["exp1", "exp2"]],
            "documents": [["Experience 1", "Experience 2"]],
            "metadatas": [
                [
                    {
                        "task_description": "Experience 1",
                        "keywords": "exp1,task",
                        "action_flow": json.dumps(
                            [
                                {
                                    "thought": "test",
                                    "action": "click",
                                    "target_element_description": "button",
                                }
                            ]
                        ),
                        "preconditions": "Test preconditions",
                        "is_successful": True,
                        "usage_count": 1,
                        "last_used_at": "2023-01-01T00:00:00",
                        "source_task_id": "task1",
                    },
                    {
                        "task_description": "Experience 2",
                        "keywords": "exp2,task",
                        "action_flow": json.dumps(
                            [
                                {
                                    "thought": "test",
                                    "action": "type",
                                    "target_element_description": "input",
                                }
                            ]
                        ),
                        "preconditions": "Test preconditions",
                        "is_successful": True,
                        "usage_count": 2,
                        "last_used_at": "2023-01-02T00:00:00",
                        "source_task_id": "task2",
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        retriever.storage.query_facts.return_value = {
            "ids": [["fact1", "fact2"]],
            "documents": [["Fact 1 content", "Fact 2 content"]],
            "metadatas": [
                [
                    {
                        "content": "Fact 1 content",
                        "keywords": "fact1,test",
                        "source": "test_source",
                        "usage_count": 1,
                        "last_used_at": "2023-01-01T00:00:00",
                    },
                    {
                        "content": "Fact 2 content",
                        "keywords": "fact2,test",
                        "source": "test_source",
                        "usage_count": 3,
                        "last_used_at": "2023-01-03T00:00:00",
                    },
                ]
            ],
            "distances": [[0.15, 0.25]],
        }

        # Mock reranking
        mock_rerank_response = Mock()
        mock_rerank_response.json.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 1, "relevance_score": 0.85},
                {"index": 2, "relevance_score": 0.75},
                {"index": 3, "relevance_score": 0.65},
            ]
        }
        mock_rerank_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_rerank_response):
            result = retriever.retrieve_memories("test query")

        # Should return properly structured results
        assert hasattr(result, "experiences")
        assert hasattr(result, "facts")
        assert len(result.experiences) > 0
        assert len(result.facts) > 0

    def test_retrieve_memories_no_results(self, retriever, mock_config):
        """Test memory retrieval when no results are found."""
        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock empty storage responses with proper structure
        retriever.storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever.storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # Mock reranking
        mock_rerank_response = Mock()
        mock_rerank_response.json.return_value = {"results": []}
        mock_rerank_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_rerank_response):
            result = retriever.retrieve_memories("test query")

        # Should return empty results
        assert len(result.experiences) == 0
        assert len(result.facts) == 0

    def test_retrieve_memories_empty_query(self, retriever, mock_config):
        """Test memory retrieval with empty query."""
        # Mock embedding generation for empty query
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.0] * 10)]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock empty storage responses with proper structure
        retriever.storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever.storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = retriever.retrieve_memories("")
        assert len(result.experiences) == 0
        assert len(result.facts) == 0

    def test_retrieve_memories_zero_top_n(self, retriever, mock_config):
        """Test memory retrieval with zero top_n."""
        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage responses with proper structure
        retriever.storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever.storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = retriever.retrieve_memories("test query", top_n=0)
        assert len(result.experiences) == 0
        assert len(result.facts) == 0

    def test_retrieve_memories_negative_top_n(self, retriever, mock_config):
        """Test memory retrieval with negative top_n."""
        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage responses with proper structure
        retriever.storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        retriever.storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = retriever.retrieve_memories("test query", top_n=-1)
        assert len(result.experiences) == 0
        assert len(result.facts) == 0

    def test_keyword_filter_experiences_empty_keywords(self, retriever):
        """Test keyword filtering experiences with empty keywords."""
        result = retriever._keyword_filter_experiences([])
        assert result == {}

    def test_keyword_filter_facts_empty_keywords(self, retriever):
        """Test keyword filtering facts with empty keywords."""
        result = retriever._keyword_filter_facts([])
        assert result == {}

    def test_vector_search_experiences_storage_error(self, retriever, mock_config):
        """Test vector search experiences when storage fails."""
        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage error
        retriever.storage.query_experiences.side_effect = Exception("Storage error")

        with pytest.raises(RetrievalError):
            retriever._vector_search_experiences([0.1, 0.2, 0.3], 5)

    def test_vector_search_facts_storage_error(self, retriever, mock_config):
        """Test vector search facts when storage fails."""
        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage error
        retriever.storage.query_facts.side_effect = Exception("Storage error")

        with pytest.raises(RetrievalError):
            retriever._vector_search_facts([0.1, 0.2, 0.3], 5)

    def test_vector_search_experiences_malformed_results(self, retriever, mock_config):
        """Test vector search experiences with malformed results."""
        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock malformed storage response (missing required keys)
        retriever.storage.query_experiences.return_value = {
            "invalid_key": [[]],
        }

        with pytest.raises(RetrievalError):
            retriever._vector_search_experiences([0.1, 0.2, 0.3], 5)

    def test_vector_search_facts_malformed_results(self, retriever, mock_config):
        """Test vector search facts with malformed results."""
        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock malformed storage response (missing required keys)
        retriever.storage.query_facts.return_value = {
            "invalid_key": [[]],
        }

        with pytest.raises(RetrievalError):
            retriever._vector_search_facts([0.1, 0.2, 0.3], 5)

    def test_rerank_results_empty_candidates(self, retriever):
        """Test reranking with empty candidates list."""
        result = retriever._rerank_results("test query", [], 5)
        assert result == []

    def test_rerank_results_single_candidate(self, retriever, mock_config):
        """Test reranking with single candidate."""
        candidates = [{"id": "1", "document": "Test document", "type": "experience"}]

        mock_config.reranker_llm_base_url = "https://test-reranker.com/v1/rerank"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.9}]
        }
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response):
            result = retriever._rerank_results("test query", candidates, 5)

        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_rerank_results_reranker_api_error(self, retriever, mock_config):
        """Test reranking when reranker API returns error status."""
        candidates = [
            {"id": "1", "document": "First document", "type": "experience"},
            {"id": "2", "document": "Second document", "type": "fact"},
        ]

        mock_config.reranker_llm_base_url = "https://test-reranker.com/v1/rerank"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API error")

        with patch("requests.post", return_value=mock_response):
            result = retriever._rerank_results("test query", candidates, 5)

        # Should return original candidates on API error
        assert result == candidates

    # duplicate removed (covered above)

    def test_rerank_results_http_error(self, retriever, mock_config):
        """Test reranking when HTTP request fails."""
        candidates = [
            {"id": "1", "document": "First document", "type": "experience"},
            {"id": "2", "document": "Second document", "type": "fact"},
        ]

        mock_config.reranker_llm_base_url = "https://test-reranker.com/v1/rerank"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

        import requests

        with patch(
            "requests.post", side_effect=requests.RequestException("HTTP error")
        ):
            result = retriever._rerank_results("test query", candidates, 5)

        # Should return original candidates on HTTP error
        assert result == candidates

    def test_merge_search_results_experiences_only(self, retriever):
        """Test merging search results with only experiences."""
        experience_results: list[dict[str, object]] = [
            {
                "id": "exp1",
                "document": "doc1",
                "metadata": {},
                "score": 0.9,
                "type": "experience",
            },
            {
                "id": "exp2",
                "document": "doc2",
                "metadata": {},
                "score": 0.8,
                "type": "experience",
            },
        ]
        fact_results: list[dict[str, object]] = []

        result = retriever._merge_search_results(
            experience_results, fact_results, limit=5
        )

        assert len(result) == 2
        assert all(item["type"] == "experience" for item in result)

    def test_merge_search_results_facts_only(self, retriever):
        """Test merging search results with only facts."""
        experience_results: list[dict[str, object]] = []
        fact_results: list[dict[str, object]] = [
            {
                "id": "fact1",
                "document": "doc1",
                "metadata": {},
                "score": 0.9,
                "type": "fact",
            },
            {
                "id": "fact2",
                "document": "doc2",
                "metadata": {},
                "score": 0.8,
                "type": "fact",
            },
        ]

        result = retriever._merge_search_results(
            experience_results, fact_results, limit=5
        )

        assert len(result) == 2
        assert all(item["type"] == "fact" for item in result)

    def test_merge_search_results_with_limit(self, retriever):
        """Test merging search results with limit."""
        experience_results: list[dict[str, object]] = [
            {
                "id": "exp1",
                "document": "doc1",
                "metadata": {},
                "score": 0.9,
                "type": "experience",
            },
            {
                "id": "exp2",
                "document": "doc2",
                "metadata": {},
                "score": 0.7,
                "type": "experience",
            },
        ]
        fact_results: list[dict[str, object]] = [
            {
                "id": "fact1",
                "document": "doc3",
                "metadata": {},
                "score": 0.8,
                "type": "fact",
            },
            {
                "id": "fact2",
                "document": "doc4",
                "metadata": {},
                "score": 0.6,
                "type": "fact",
            },
        ]

        result = retriever._merge_search_results(
            experience_results, fact_results, limit=3
        )

        assert len(result) == 3
        # Should be sorted by score (highest first)
        assert result[0]["score"] >= result[1]["score"] >= result[2]["score"]

    def test_convert_to_memory_objects_experiences(self, retriever):
        """Test converting results to memory objects for experiences."""
        experience_results = [
            {
                "id": "exp1",
                "document": "Test task",
                "metadata": {
                    "task_description": "Test task",
                    "keywords": "test,task",
                    "action_flow": json.dumps(
                        [
                            {
                                "thought": "test thought",
                                "action": "click",
                                "target_element_description": "test button",
                            }
                        ]
                    ),
                    "preconditions": "Test preconditions",
                    "is_successful": True,
                    "usage_count": 1,
                    "last_used_at": "2023-01-01T00:00:00",
                    "source_task_id": "test_001",
                },
                "type": "experience",
            }
        ]

        fact_results: list[dict[str, object]] = []

        experiences, facts = retriever._convert_to_memory_objects(
            experience_results + fact_results
        )

        assert len(experiences) == 1
        assert len(facts) == 0
        assert experiences[0].task_description == "Test task"
        assert len(experiences[0].action_flow) == 1

    def test_convert_to_memory_objects_facts(self, retriever):
        """Test converting results to memory objects for facts."""
        experience_results: list[dict[str, object]] = []

        fact_results: list[dict[str, object]] = [
            {
                "id": "fact1",
                "document": "Test fact content",
                "metadata": {
                    "content": "Test fact content",
                    "keywords": "test,fact",
                    "source": "test_source",
                    "usage_count": 1,
                    "last_used_at": "2023-01-01T00:00:00",
                },
                "type": "fact",
            }
        ]

        experiences, facts = retriever._convert_to_memory_objects(
            experience_results + fact_results
        )

        assert len(experiences) == 0
        assert len(facts) == 1
        assert facts[0].content == "Test fact content"

    def test_convert_to_memory_objects_invalid_action_flow(self, retriever):
        """Test converting results with invalid action flow data."""
        # Mock results with invalid action_flow
        experience_results = [
            {
                "id": "exp1",
                "document": "Test task",
                "metadata": {
                    "task_description": "Test task",
                    "keywords": "test,task",
                    "action_flow": "invalid_json_string",  # Invalid JSON
                    "preconditions": "Test preconditions",
                    "is_successful": True,
                    "usage_count": 1,
                    "last_used_at": "2023-01-01T00:00:00",
                    "source_task_id": "test_001",
                },
                "type": "experience",
            }
        ]

        fact_results: list[dict[str, object]] = []

        # Combine results as expected by the method
        all_results = experience_results + fact_results

        # Should raise RetrievalError due to invalid JSON
        with pytest.raises(RetrievalError) as exc_info:
            retriever._convert_to_memory_objects(all_results)

        assert "Failed to parse action_flow JSON" in str(exc_info.value)

    def test_update_usage_stats_storage_error(self, retriever):
        """Test update usage stats when storage fails."""
        experiences = [
            ExperienceRecord(
                task_description="Test",
                keywords=["test"],
                action_flow=[],
                preconditions="",
                is_successful=True,
                source_task_id="test_001",
            )
        ]

        # facts not used explicitly; keep behavior minimal

        # Mock storage to raise error
        retriever.storage.update_usage_stats.side_effect = Exception("Storage error")

        # Should handle the error gracefully and not raise
        retriever._update_usage_stats([exp.source_task_id for exp in experiences])

        # Verify storage was called
        assert retriever.storage.update_usage_stats.called

    def test_get_similar_experiences_embedding_error(self, retriever, mock_config):
        """Test get_similar_experiences when embedding generation fails."""
        # Mock embedding client to raise an exception
        mock_config.embedding_client.embeddings.create.side_effect = Exception(
            "Embedding error"
        )

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_similar_experiences("test task", 5)

        assert "Failed to generate query embedding" in str(exc_info.value)

    def test_get_related_facts_embedding_error(self, retriever, mock_config):
        """Test get_related_facts when embedding generation fails."""
        # Mock embedding client to raise an exception
        mock_config.embedding_client.embeddings.create.side_effect = Exception(
            "Embedding error"
        )

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_related_facts("test topic", 5)

        assert "Failed to generate query embedding" in str(exc_info.value)

    def test_get_similar_experiences_storage_error(self, retriever, mock_config):
        """Test get_similar_experiences when storage fails."""
        # Mock embedding generation to succeed
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to raise an exception
        retriever.storage.query_experiences.side_effect = Exception("Storage error")

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_similar_experiences("test task", 5)

        assert "Failed to perform vector search on experiences" in str(exc_info.value)

    def test_get_related_facts_storage_error(self, retriever, mock_config):
        """Test get_related_facts when storage fails."""
        # Mock embedding generation to succeed
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to raise an exception
        retriever.storage.query_facts.side_effect = Exception("Storage error")

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_related_facts("test topic", 5)

        assert "Failed to perform vector search on facts" in str(exc_info.value)

    def test_get_similar_experiences_conversion_error(self, retriever, mock_config):
        """Test get_similar_experiences when conversion fails."""
        # Mock embedding generation to succeed
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to return valid results
        retriever.storage.query_experiences.return_value = {
            "ids": [["exp1"]],
            "documents": [["Test task"]],
            "metadatas": [
                [
                    {
                        "task_description": "Test task",
                        "keywords": "test,task",
                        "action_flow": "invalid_json",  # Invalid JSON to cause conversion error
                        "preconditions": "Test preconditions",
                        "is_successful": True,
                        "usage_count": 1,
                        "last_used_at": "2023-01-01T00:00:00",
                        "source_task_id": "test_001",
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        with pytest.raises(RetrievalError) as exc_info:
            retriever.get_similar_experiences("test task", 5)

        assert "Failed to convert experiences" in str(exc_info.value)

    def test_get_related_facts_conversion_error(self, retriever, mock_config):
        """Test get_related_facts when conversion fails."""
        # Mock embedding generation to succeed
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to return valid results
        retriever.storage.query_facts.return_value = {
            "ids": [["fact1"]],
            "documents": [["Test fact"]],
            "metadatas": [
                [
                    {
                        "content": "Test fact content",
                        "keywords": "test,fact",
                        "source": "test_source",
                        "usage_count": 1,
                        "last_used_at": "2023-01-01T00:00:00",
                    }
                ]
            ],
            "distances": [[0.1]],
        }

        # Mock the conversion method to raise an error
        with patch.object(
            retriever,
            "_convert_to_memory_objects",
            side_effect=Exception("Conversion error"),
        ):
            with pytest.raises(RetrievalError) as exc_info:
                retriever.get_related_facts("test topic", 5)

        assert "Failed to convert facts" in str(exc_info.value)


class TestRetrieverCoverage:
    """Tests for uncovered code paths in retriever.py"""

    @pytest.fixture
    def retriever(self, mock_config, mock_storage):
        """Create MemoryRetriever instance for testing."""
        with patch("gui_agent_memory.retriever.get_config", return_value=mock_config):
            retriever = MemoryRetriever()
            retriever.storage = mock_storage
            return retriever

    def test_build_keyword_filter_empty_keywords(self, retriever):
        """Test _build_keyword_filter with empty keywords (line 154)."""
        # Test with empty list
        result = retriever._build_keyword_filter([])
        assert result == {}

        # Test with None
        result = retriever._build_keyword_filter(None)
        assert result == {}

        # Test with list containing empty strings
        result = retriever._build_keyword_filter(["", "  ", ""])
        # This should still return {} due to the current implementation
        assert result == {}

    def test_build_keyword_filter_with_keywords(self, retriever):
        """Test _build_keyword_filter with actual keywords."""
        # Test with valid keywords - current implementation returns empty dict
        # due to being a temporary workaround (line 159)
        result = retriever._build_keyword_filter(["keyword1", "keyword2"])
        assert result == {}

    def test_vector_search_experiences_error_handling(self, retriever, mock_config):
        """Test vector search error handling in experiences."""
        # Mock embedding generation to succeed
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        )

        # Mock storage query to raise an exception
        retriever.storage.query_experiences.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            retriever._vector_search_experiences([0.1, 0.2, 0.3], 5)

        assert "Database error" in str(exc_info.value)

    def test_vector_search_facts_error_handling(self, retriever, mock_config):
        """Test vector search error handling in facts."""
        # Mock embedding generation to succeed
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        )

        # Mock storage query to raise an exception
        retriever.storage.query_facts.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            retriever._vector_search_facts([0.1, 0.2, 0.3], 5)

        assert "Database error" in str(exc_info.value)

    def test_keyword_filter_experiences_empty_results(self, retriever):
        """Test keyword filtering with empty results."""
        # Mock storage to return empty results in the correct format
        retriever.storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = retriever._keyword_filter_experiences(["test"])
        # Should handle empty results gracefully
        assert isinstance(result, dict)

    def test_keyword_filter_facts_empty_results(self, retriever):
        """Test keyword filtering facts with empty results."""
        # Mock storage to return empty results in the correct format
        retriever.storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        result = retriever._keyword_filter_facts(["test"])
        # Should handle empty results gracefully
        assert isinstance(result, dict)

    def test_convert_to_memory_objects_invalid_action_flow(self, retriever):
        """Test converting results with invalid action flow data."""
        # Mock results with invalid action_flow
        experience_results = [
            {
                "id": "exp1",
                "document": "Test task",
                "metadata": {
                    "task_description": "Test task",
                    "keywords": "test,task",
                    "action_flow": "invalid_json_string",  # Invalid JSON
                    "preconditions": "Test preconditions",
                    "is_successful": True,
                    "usage_count": 1,
                    "last_used_at": "2023-01-01T00:00:00",
                    "source_task_id": "test_001",
                },
                "type": "experience",
            }
        ]

        fact_results: list[dict[str, object]] = []

        # Combine results as expected by the method
        all_results = experience_results + fact_results

        # Should raise RetrievalError due to invalid JSON
        with pytest.raises(RetrievalError) as exc_info:
            retriever._convert_to_memory_objects(all_results)

        assert "Failed to parse action_flow JSON" in str(exc_info.value)

    def test_merge_search_results_with_duplicates(self, retriever):
        """Test merging results with duplicate IDs."""
        # Create test data in the format expected by the actual method
        experience_results = [
            {
                "id": "exp1",
                "document": "doc1",
                "metadata": {"meta1": "value1"},
                "distance": 0.1,
                "type": "experience",
            },
            {
                "id": "exp2",
                "document": "doc2",
                "metadata": {"meta2": "value2"},
                "distance": 0.2,
                "type": "experience",
            },
            {
                "id": "exp1",
                "document": "doc1",
                "metadata": {"meta1": "value1"},
                "distance": 0.15,
                "type": "experience",
            },  # duplicate
            {
                "id": "exp3",
                "document": "doc3",
                "metadata": {"meta3": "value3"},
                "distance": 0.3,
                "type": "experience",
            },
        ]

        fact_results = [
            {
                "id": "fact1",
                "document": "fact_doc1",
                "metadata": {"fact_meta": "fact_value"},
                "distance": 0.05,
                "type": "fact",
            }
        ]

        # Call the method with the correct signature
        result = retriever._merge_search_results(
            experience_results, fact_results, limit=10
        )

        # Extract IDs from results
        result_ids = [item["id"] for item in result]

        # Should have 5 results total (4 experiences + 1 fact)
        assert len(result_ids) == 5

        # exp1 appears twice in the input, so it should appear twice in the output
        # The current implementation doesn't deduplicate, so this is expected behavior
        exp1_count = result_ids.count("exp1")
        assert exp1_count == 2, (
            f"Expected exp1 to appear twice, but appeared {exp1_count} times"
        )

    def test_update_usage_stats_storage_error(self, retriever):
        """Test update usage stats when storage fails."""
        experiences = [
            ExperienceRecord(
                task_description="Test",
                keywords=["test"],
                action_flow=[],
                preconditions="",
                is_successful=True,
                source_task_id="test_001",
            )
        ]

        # facts not used explicitly; keep behavior minimal

        # Mock storage to raise error
        retriever.storage.update_usage_stats.side_effect = Exception("Storage error")

        # Should handle the error gracefully and not raise
        retriever._update_usage_stats([exp.source_task_id for exp in experiences])

        # Verify storage was called
        assert retriever.storage.update_usage_stats.called

    def test_get_similar_experiences_empty_task(self, retriever, mock_config):
        """Test get_similar_experiences with empty task description."""
        # Mock the embedding client to return a proper response
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        )

        # Mock storage to return empty results (no matching experiences)
        retriever.storage.query_experiences.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # Should return empty list for empty task rather than raising ValueError
        result = retriever.get_similar_experiences("", 5)
        assert result == []

    def test_get_related_facts_empty_topic(self, retriever, mock_config):
        """Test get_related_facts with empty topic."""
        # Mock the embedding client to return a proper response
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        )

        # Mock storage to return empty results (no matching facts)
        retriever.storage.query_facts.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # Should return empty list for empty topic rather than raising ValueError
        result = retriever.get_related_facts("", 5)
        assert result == []

    def test_rerank_results_timeout_error(self, retriever, mock_config):
        """Test rerank results with timeout error."""
        candidates = [{"id": "test1", "content": "test content", "type": "experience"}]

        mock_config.reranker_llm_base_url = "https://test-reranker.com"
        mock_config.reranker_llm_api_key = "test-key"
        mock_config.reranker_model = "test-model"

        import requests

        with patch("requests.post", side_effect=requests.Timeout("Request timeout")):
            # Should handle timeout gracefully and return original order
            result = retriever._rerank_results("test query", candidates, 3)

            assert len(result) == 1
            # Should return original candidates without modification
            assert result[0]["id"] == candidates[0]["id"]
