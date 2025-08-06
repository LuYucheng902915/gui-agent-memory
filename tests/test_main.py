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


class TestMemorySystemIntegration:
    """Test comprehensive MemorySystem workflows."""

    @pytest.fixture
    def memory_system(self, mock_config):
        """Create MemorySystem with fully mocked dependencies."""
        with (
            patch("gui_agent_memory.main.get_config", return_value=mock_config),
            patch("gui_agent_memory.main.MemoryStorage") as mock_storage_class,
            patch("gui_agent_memory.main.MemoryIngestion") as mock_ingestion_class,
            patch("gui_agent_memory.main.MemoryRetriever") as mock_retriever_class,
        ):
            mock_storage = Mock()
            mock_ingestion = Mock()
            mock_retriever = Mock()

            mock_storage_class.return_value = mock_storage
            mock_ingestion_class.return_value = mock_ingestion
            mock_retriever_class.return_value = mock_retriever

            system = MemorySystem()

            # Attach mocks for easy access in tests
            system._mock_storage = mock_storage
            system._mock_ingestion = mock_ingestion
            system._mock_retriever = mock_retriever

            yield system

    def test_full_learning_and_retrieval_workflow(self, memory_system):
        """Test complete workflow: learn from task -> retrieve memories."""
        from gui_agent_memory.models import (
            ExperienceRecord,
            FactRecord,
            RetrievalResult,
        )

        # Step 1: Learn from a task
        raw_history = [
            {"thought": "Click login", "action": "click", "target": "login_btn"},
            {
                "thought": "Enter username",
                "action": "type",
                "target": "username",
                "text": "user@test.com",
            },
        ]

        memory_system._mock_ingestion.learn_from_task.return_value = (
            "Successfully learned from task_001"
        )

        learn_result = memory_system.learn_from_task(
            raw_history=raw_history,
            is_successful=True,
            source_task_id="task_001",
            task_description="Login workflow",
            app_name="TestApp",
        )

        # Verify learning
        assert "Successfully learned" in learn_result
        memory_system._mock_ingestion.learn_from_task.assert_called_once()

        # Step 2: Retrieve related memories
        mock_retrieval_result = Mock(spec=RetrievalResult)
        mock_retrieval_result.experiences = [Mock(spec=ExperienceRecord)]
        mock_retrieval_result.facts = [Mock(spec=FactRecord)]
        mock_retrieval_result.total_results = 2

        memory_system._mock_retriever.retrieve_memories.return_value = (
            mock_retrieval_result
        )

        retrieval_result = memory_system.retrieve_memories(
            "How to login to application"
        )

        # Verify retrieval
        assert retrieval_result == mock_retrieval_result
        memory_system._mock_retriever.retrieve_memories.assert_called_once_with(
            "How to login to application", 3
        )

    def test_add_fact_and_retrieve_workflow(self, memory_system):
        """Test workflow: add fact -> retrieve related information."""
        from gui_agent_memory.models import (
            FactRecord,
            RetrievalResult,
        )

        # Step 1: Add a fact
        memory_system._mock_ingestion.add_fact.return_value = (
            "Successfully added fact with ID: fact_001"
        )

        fact_result = memory_system.add_fact(
            "OAuth 2.0 is an authorization framework",
            ["oauth", "authorization", "security"],
            "documentation",
        )

        # Verify fact addition
        assert "Successfully added fact" in fact_result
        memory_system._mock_ingestion.add_fact.assert_called_once_with(
            "OAuth 2.0 is an authorization framework",
            ["oauth", "authorization", "security"],
            "documentation",
        )

        # Step 2: Retrieve related information
        mock_retrieval_result = Mock(spec=RetrievalResult)
        mock_retrieval_result.experiences = []
        mock_retrieval_result.facts = [Mock(spec=FactRecord)]
        mock_retrieval_result.total_results = 1

        memory_system._mock_retriever.retrieve_memories.return_value = (
            mock_retrieval_result
        )

        retrieval_result = memory_system.retrieve_memories("What is OAuth 2.0?")

        # Verify retrieval found the related fact
        assert len(retrieval_result.facts) == 1
        memory_system._mock_retriever.retrieve_memories.assert_called_once_with(
            "What is OAuth 2.0?", 3
        )


class TestMemorySystemEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def memory_system(self, mock_config):
        """Create MemorySystem with mocked dependencies."""
        with (
            patch("gui_agent_memory.main.get_config", return_value=mock_config),
            patch("gui_agent_memory.main.MemoryStorage") as mock_storage_class,
            patch("gui_agent_memory.main.MemoryIngestion") as mock_ingestion_class,
            patch("gui_agent_memory.main.MemoryRetriever") as mock_retriever_class,
        ):
            mock_storage = Mock()
            mock_ingestion = Mock()
            mock_retriever = Mock()

            mock_storage_class.return_value = mock_storage
            mock_ingestion_class.return_value = mock_ingestion
            mock_retriever_class.return_value = mock_retriever

            system = MemorySystem()
            system._mock_storage = mock_storage
            system._mock_ingestion = mock_ingestion
            system._mock_retriever = mock_retriever

            yield system

    def test_very_long_query(self, memory_system):
        """Test retrieval with very long query."""
        from gui_agent_memory.models import RetrievalResult

        long_query = "How to " + "very " * 1000 + "long query"
        mock_result = Mock(spec=RetrievalResult)
        memory_system._mock_retriever.retrieve_memories.return_value = mock_result

        result = memory_system.retrieve_memories(long_query)

        assert result == mock_result
        memory_system._mock_retriever.retrieve_memories.assert_called_once_with(
            long_query, 3
        )

    def test_unicode_query(self, memory_system):
        """Test retrieval with Unicode query."""
        from gui_agent_memory.models import RetrievalResult

        unicode_query = "如何使用这个应用程序 with emojis 🚀📱"
        mock_result = Mock(spec=RetrievalResult)
        memory_system._mock_retriever.retrieve_memories.return_value = mock_result

        result = memory_system.retrieve_memories(unicode_query)

        assert result == mock_result

    def test_extreme_top_n_values(self, memory_system):
        """Test retrieval with extreme top_n values."""
        from gui_agent_memory.models import (
            RetrievalResult,
        )

        query = "test query"
        mock_result = Mock(spec=RetrievalResult)
        memory_system._mock_retriever.retrieve_memories.return_value = mock_result

        extreme_values = [1, 100, 1000, 10000]
        for top_n in extreme_values:
            result = memory_system.retrieve_memories(query, top_n=top_n)
            assert result == mock_result
            # Verify the extreme value was passed through
            memory_system._mock_retriever.retrieve_memories.assert_called_with(
                query, top_n
            )
