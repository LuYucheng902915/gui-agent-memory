"""
Test configuration and utilities for the memory system.

Provides common test fixtures, mocks, and utility functions.
"""

# Fix SQLite version compatibility for ChromaDB before any other imports
try:
    import sys

    import pysqlite3 as sqlite3

    sys.modules["sqlite3"] = sqlite3
except ImportError:
    pass

import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

# Set up FAKE environment variables for all tests - DO NOT USE REAL API KEYS!
os.environ.setdefault(
    "GITEE_AI_EMBEDDING_BASE_URL", "https://test-embedding.example.com/v1"
)
os.environ.setdefault("GITEE_AI_EMBEDDING_API_KEY", "test-fake-embedding-key-12345")
os.environ.setdefault(
    "GITEE_AI_RERANKER_BASE_URL", "https://test-reranker.example.com/v1/rerank"
)
os.environ.setdefault("GITEE_AI_RERANKER_API_KEY", "test-fake-reranker-key-67890")
os.environ.setdefault("EXPERIENCE_LLM_BASE_URL", "https://test-llm.example.com/v1")
os.environ.setdefault("EXPERIENCE_LLM_API_KEY", "test-fake-llm-key-abcdef")


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.gitee_ai_embedding_base_url = "https://test-ai.example.com/v1"
    config.gitee_ai_embedding_api_key = "test_gitee_key"
    config.gitee_ai_reranker_base_url = "https://ai.gitee.com/v1/rerank"
    config.gitee_ai_reranker_api_key = "test_gitee_key"
    config.experience_llm_base_url = "https://test-llm.example.com/v1"
    config.experience_llm_api_key = "test_llm_key"
    config.embedding_model = "test-embedding-model"
    config.reranker_model = "test-reranker-model"
    config.experience_llm_model = "test-llm-model"
    config.chroma_db_path = "./test_data/chroma"
    config.experiential_collection_name = "test_experiential_memories"
    config.declarative_collection_name = "test_declarative_memories"
    config.default_top_k = 20
    config.default_top_n = 3
    config.embedding_dimension = 10
    config.failed_learning_log_path = "./test_logs/failed_learning.jsonl"

    # Mock clients - set up as both direct attributes and return values for compatibility
    config.gitee_ai_client = Mock()
    config.experience_llm_client = Mock()
    config.embedding_client = (
        config.gitee_ai_client
    )  # Add direct reference for test compatibility
    config.get_embedding_client = Mock(return_value=config.gitee_ai_client)
    config.get_reranker_config = Mock(
        return_value={
            "base_url": "https://ai.gitee.com/v1/rerank",
            "api_key": "test_gitee_key",
            "model": "test-reranker-model",
        }
    )
    config.get_experience_llm_client = Mock(return_value=config.experience_llm_client)
    config.validate_configuration = Mock(return_value=True)

    return config


@pytest.fixture
def api_responses():
    """Load mock API responses from JSON file."""
    responses_path = Path(__file__).parent / "mocks" / "api_responses.json"
    return json.loads(responses_path.read_text())


@pytest.fixture
def test_fixtures():
    """Load test fixtures from JSON file."""
    fixtures_path = Path(__file__).parent / "mocks" / "test_fixtures.json"
    return json.loads(fixtures_path.read_text())


@pytest.fixture
def mock_openai_client(api_responses):
    """Mock OpenAI client with pre-configured responses."""
    client = Mock()

    # Mock embeddings
    embedding_response = Mock()
    embedding_response.data = [Mock()]
    embedding_response.data[0].embedding = api_responses["embedding_response"]["data"][
        0
    ]["embedding"]
    client.embeddings.create.return_value = embedding_response

    # Mock chat completions
    chat_response = Mock()
    chat_response.choices = [Mock()]
    chat_response.choices[0].message.content = api_responses[
        "experience_distillation_response"
    ]["choices"][0]["message"]["content"]
    client.chat.completions.create.return_value = chat_response

    # Mock models list
    models_response = Mock()
    models_response.data = api_responses["models_list_response"]["data"]
    client.models.list.return_value = models_response

    return client


@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB collection for testing."""
    collection = Mock()
    collection.count.return_value = 0
    collection.add.return_value = None
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    return collection


@pytest.fixture
def mock_chroma_collection(mock_chromadb_collection):
    """Alias for mock_chromadb_collection to match test_storage.py expectations."""
    return mock_chromadb_collection


@pytest.fixture
def mock_storage(mock_chromadb_collection):
    """Mock storage layer for testing."""
    storage = Mock()
    storage.experiential_collection = mock_chromadb_collection
    storage.declarative_collection = mock_chromadb_collection
    storage.get_collection.return_value = mock_chromadb_collection
    storage.add_experiences.return_value = ["test_id_1"]
    storage.add_facts.return_value = ["test_id_2"]
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
    storage.experience_exists.return_value = False
    storage.get_collection_stats.return_value = {
        "experiential_memories": 0,
        "declarative_memories": 0,
        "total": 0,
    }
    storage.clear_collections.return_value = None
    return storage


@pytest.fixture
def sample_experience_record():
    """Sample ExperienceRecord for testing."""
    from gui_agent_memory.models import ActionStep, ExperienceRecord

    return ExperienceRecord(
        task_description="Log into Gmail using Chrome browser",
        keywords=["gmail", "login", "chrome", "browser"],
        action_flow=[
            ActionStep(
                thought="Navigate to Gmail login page",
                action="navigate",
                target_element_description="Chrome address bar",
            ),
            ActionStep(
                thought="Enter email address",
                action="type",
                target_element_description="Email input field",
            ),
            ActionStep(
                thought="Click sign in button",
                action="click",
                target_element_description="Sign in button",
            ),
        ],
        preconditions="Chrome browser is open",
        is_successful=True,
        source_task_id="test_task_001",
    )


@pytest.fixture
def sample_fact_record():
    """Sample FactRecord for testing."""
    from gui_agent_memory.models import FactRecord

    return FactRecord(
        content="Chrome browser stores passwords in the password manager",
        keywords=["chrome", "password", "security", "browser"],
        source="documentation",
    )


@pytest.fixture
def sample_retrieval_result():
    """Sample RetrievalResult for testing."""
    from gui_agent_memory.models import RetrievalResult

    return RetrievalResult(
        experiences=[], facts=[], query="Test query", total_results=0
    )


@pytest.fixture
def sample_raw_history():
    """Sample raw history data for testing."""
    return [
        {
            "thought": "I need to click the login button",
            "action": "click",
            "target": "login_button",
            "result": "success",
        },
        {
            "thought": "Now I'll enter the username",
            "action": "type",
            "target": "username_field",
            "text": "user@example.com",
            "result": "success",
        },
        {
            "thought": "I need to enter the password",
            "action": "type",
            "target": "password_field",
            "text": "password123",
            "result": "success",
        },
        {
            "thought": "I should click the submit button",
            "action": "click",
            "target": "submit_button",
            "result": "success",
        },
        {
            "thought": "Wait for the page to load",
            "action": "wait",
            "target": "dashboard",
            "result": "success",
        },
    ]


@pytest.fixture
def sample_learning_request():
    """Sample LearningRequest for testing."""
    from gui_agent_memory.models import LearningRequest

    return LearningRequest(
        raw_history=[
            {"action": "click", "target": "button", "thought": "Need to click"}
        ],
        is_successful=True,
        source_task_id="test_123",
        app_name="TestApp",
    )


def create_mock_experience_record(test_fixtures):
    """Create a mock ExperienceRecord from fixtures."""
    from gui_agent_memory.models import ActionStep, ExperienceRecord

    fixture = test_fixtures["sample_experience_record"]
    action_steps = [ActionStep(**step) for step in fixture["action_flow"]]

    return ExperienceRecord(
        task_description=fixture["task_description"],
        keywords=fixture["keywords"],
        action_flow=action_steps,
        preconditions=fixture["preconditions"],
        is_successful=fixture["is_successful"],
        usage_count=fixture["usage_count"],
        source_task_id=fixture["source_task_id"],
    )


def create_mock_fact_record(test_fixtures):
    """Create a mock FactRecord from fixtures."""
    from gui_agent_memory.models import FactRecord

    fixture = test_fixtures["sample_fact_record"]

    return FactRecord(
        content=fixture["content"],
        keywords=fixture["keywords"],
        source=fixture["source"],
        usage_count=fixture["usage_count"],
    )


class TestEnvironment:
    """Test environment setup utilities."""

    @staticmethod
    def setup_test_env():
        """Setup FAKE test environment variables - DO NOT USE REAL API KEYS!"""
        os.environ["GITEE_AI_EMBEDDING_BASE_URL"] = (
            "https://test-embedding.example.com/v1"
        )
        os.environ["GITEE_AI_EMBEDDING_API_KEY"] = "test-fake-embedding-key-12345"
        os.environ["GITEE_AI_RERANKER_BASE_URL"] = (
            "https://test-reranker.example.com/v1/rerank"
        )
        os.environ["GITEE_AI_RERANKER_API_KEY"] = "test-fake-reranker-key-67890"
        os.environ["EXPERIENCE_LLM_BASE_URL"] = "https://test-llm.example.com/v1"
        os.environ["EXPERIENCE_LLM_API_KEY"] = "test-fake-llm-key-abcdef"
        os.environ["CHROMA_DB_PATH"] = "./test_data/chroma"

    @staticmethod
    def cleanup_test_env():
        """Clean up test environment."""
        test_vars = [
            "GITEE_AI_EMBEDDING_BASE_URL",
            "GITEE_AI_EMBEDDING_API_KEY",
            "GITEE_AI_RERANKER_BASE_URL",
            "GITEE_AI_RERANKER_API_KEY",
            "EXPERIENCE_LLM_BASE_URL",
            "EXPERIENCE_LLM_API_KEY",
            "CHROMA_DB_PATH",
        ]
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]
