"""
Test suite for the main API interface.
Tests high-level memory system operations.
"""

from unittest.mock import Mock, patch

import pytest

from gui_agent_memory.main import MemorySystem


class TestMemorySystem:
    """Test cases for MemorySystem class."""

    @pytest.fixture
    def mock_storage(self):
        """Mock storage for testing."""
        storage = Mock()
        return storage

    @pytest.fixture
    def mock_ingestion(self):
        """Mock ingestion for testing."""
        ingestion = Mock()
        return ingestion

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever for testing."""
        retriever = Mock()
        return retriever

    @pytest.fixture
    def memory_system(self, mock_storage, mock_ingestion, mock_retriever, mock_config):
        """Create MemorySystem instance with mocked dependencies."""
        with (
            patch("gui_agent_memory.main.MemoryStorage", return_value=mock_storage),
            patch("gui_agent_memory.main.MemoryIngestion", return_value=mock_ingestion),
            patch("gui_agent_memory.main.MemoryRetriever", return_value=mock_retriever),
            patch("gui_agent_memory.main.get_config", return_value=mock_config),
        ):
            return MemorySystem()

    @pytest.fixture
    def sample_retrieval_results(self):
        """Sample retrieval results for testing."""
        return {
            "experiences": [
                {
                    "id": "exp_1",
                    "metadata": {
                        "task_description": "Login to application",
                        "keywords": ["login", "authentication"],
                        "is_successful": True,
                        "action_flow": [
                            {
                                "thought": "Click login button",
                                "action": "click",
                                "target_element_description": "login button",
                            }
                        ],
                    },
                }
            ],
            "facts": [
                {
                    "id": "fact_1",
                    "metadata": {
                        "content": "Login requires valid credentials",
                        "keywords": ["login", "credentials"],
                        "source": "manual",
                    },
                }
            ],
        }

    @pytest.mark.unit
    def test_init(self, mock_storage, mock_ingestion, mock_retriever, mock_config):
        """Test MemorySystem initialization."""
        with (
            patch("gui_agent_memory.main.MemoryStorage", return_value=mock_storage),
            patch("gui_agent_memory.main.MemoryIngestion", return_value=mock_ingestion),
            patch("gui_agent_memory.main.MemoryRetriever", return_value=mock_retriever),
            patch("gui_agent_memory.main.get_config", return_value=mock_config),
        ):

            memory_system = MemorySystem()

            assert memory_system.storage == mock_storage
            assert memory_system.ingestion == mock_ingestion
            assert memory_system.retriever == mock_retriever

    @pytest.mark.unit
    def test_retrieve_memories_success(
        self, memory_system, mock_retriever, sample_retrieval_results
    ):
        """Test successful memory retrieval."""
        # Arrange
        query = "How to login to application"
        mock_retriever.retrieve_memories.return_value = sample_retrieval_results

        # Act
        result = memory_system.retrieve_memories(query, top_n=3)

        # Assert
        assert "experiences" in result
        assert "facts" in result
        mock_retriever.retrieve_memories.assert_called_once_with(query, 3)

    @pytest.mark.unit
    def test_learn_from_task_success(self, memory_system, mock_ingestion):
        """Test successful learning from task."""
        # Arrange
        sample_raw_history = [{"action": "click", "target": "button"}]
        mock_ingestion.learn_from_task.return_value = "task_123"

        # Act
        result = memory_system.learn_from_task(
            raw_history=sample_raw_history,
            is_successful=True,
            source_task_id="task_123",
            task_description="Login to application",
            app_name="TestApp",
        )

        # Assert
        assert result == "task_123"
        mock_ingestion.learn_from_task.assert_called_once()

    @pytest.mark.unit
    def test_add_fact_success(self, memory_system, mock_ingestion):
        """Test successful fact addition."""
        # Arrange
        mock_ingestion.add_fact.return_value = "fact_123"

        # Act
        result = memory_system.add_fact(
            content="Python is a programming language",
            keywords=["python", "programming"],
            source="manual",
        )

        # Assert
        assert result == "fact_123"
        mock_ingestion.add_fact.assert_called_once()
