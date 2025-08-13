"""
Test suite for the main API interface.
Tests high-level memory system operations.
"""

from unittest.mock import Mock, patch

import pytest

from gui_agent_memory.main import MemorySystem, MemorySystemError, create_memory_system


class TestMemorySystem:
    """Test cases for MemorySystem class."""

    @pytest.fixture
    def mock_storage(self):
        """Mock storage for testing."""
        return Mock()

    @pytest.fixture
    def mock_ingestion(self):
        """Mock ingestion for testing."""
        return Mock()

    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever for testing."""
        return Mock()

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

    def test_zero_top_n(self, memory_system):
        """Test retrieval with zero top_n value."""
        from gui_agent_memory.models import RetrievalResult

        query = "test query"
        mock_result = Mock(spec=RetrievalResult)
        memory_system._mock_retriever.retrieve_memories.return_value = mock_result

        result = memory_system.retrieve_memories(query, top_n=0)
        assert result == mock_result
        memory_system._mock_retriever.retrieve_memories.assert_called_with(query, 0)


class TestMemorySystemErrorHandling:
    """Test error handling in MemorySystem."""

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

    def test_retrieve_memories_retrieval_error(self, memory_system):
        """Test retrieve_memories when RetrievalError is raised."""
        from gui_agent_memory.retriever import RetrievalError

        memory_system._mock_retriever.retrieve_memories.side_effect = RetrievalError(
            "Test retrieval error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.retrieve_memories("test query")

        assert "Memory retrieval failed" in str(exc_info.value)

    def test_learn_from_task_empty_raw_history(self, memory_system):
        """Test learn_from_task with empty raw_history."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.learn_from_task(
                raw_history=[],
                is_successful=True,
                source_task_id="task_123",
            )

        assert "Raw history cannot be empty" in str(exc_info.value)

    def test_learn_from_task_empty_source_task_id(self, memory_system):
        """Test learn_from_task with empty source_task_id."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.learn_from_task(
                raw_history=[{"action": "click"}],
                is_successful=True,
                source_task_id="",
            )

        assert "Source task ID cannot be empty" in str(exc_info.value)

    def test_learn_from_task_whitespace_source_task_id(self, memory_system):
        """Test learn_from_task with whitespace-only source_task_id."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.learn_from_task(
                raw_history=[{"action": "click"}],
                is_successful=True,
                source_task_id="   ",
            )

        assert "Source task ID cannot be empty" in str(exc_info.value)

    def test_learn_from_task_ingestion_error(self, memory_system):
        """Test learn_from_task when IngestionError is raised."""
        from gui_agent_memory.ingestion import IngestionError

        memory_system._mock_ingestion.learn_from_task.side_effect = IngestionError(
            "Test ingestion error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.learn_from_task(
                raw_history=[{"action": "click"}],
                is_successful=True,
                source_task_id="task_123",
            )

        assert "Learning from task failed" in str(exc_info.value)

    def test_add_fact_empty_content(self, memory_system):
        """Test add_fact with empty content."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.add_fact(
                content="",
                keywords=["test"],
            )

        assert "Fact content cannot be empty" in str(exc_info.value)

    def test_add_fact_whitespace_content(self, memory_system):
        """Test add_fact with whitespace-only content."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.add_fact(
                content="   ",
                keywords=["test"],
            )

        assert "Fact content cannot be empty" in str(exc_info.value)

    def test_add_fact_empty_keywords(self, memory_system):
        """Test add_fact with empty keywords list."""
        memory_system._mock_ingestion.add_fact.return_value = "fact_123"

        result = memory_system.add_fact(
            content="Test fact",
            keywords=[],
        )

        assert result == "fact_123"
        memory_system._mock_ingestion.add_fact.assert_called_once_with(
            "Test fact", [], "manual"
        )

    def test_add_fact_ingestion_error(self, memory_system):
        """Test add_fact when IngestionError is raised."""
        from gui_agent_memory.ingestion import IngestionError

        memory_system._mock_ingestion.add_fact.side_effect = IngestionError(
            "Test ingestion error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.add_fact(
                content="Test fact",
                keywords=["test"],
            )

        assert "Adding fact failed" in str(exc_info.value)


class TestMemorySystemAdditionalMethods:
    """Test additional MemorySystem methods not covered in basic tests."""

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

    def test_add_experience_success(self, memory_system):
        """Test successful add_experience."""
        from gui_agent_memory.models import ActionStep, ExperienceRecord

        experience = ExperienceRecord(
            task_description="Test task",
            keywords=["test"],
            action_flow=[
                ActionStep(
                    thought="Test thought",
                    action="click",
                    target_element_description="button",
                )
            ],
            preconditions="Test preconditions",
            is_successful=True,
            source_task_id="task_123",
        )

        memory_system._mock_ingestion.add_experience.return_value = "exp_123"

        result = memory_system.add_experience(experience)

        assert result == "exp_123"
        # 新签名包含 op，使用 kwargs 断言核心参数
        assert memory_system._mock_ingestion.add_experience.call_count == 1
        args, kwargs = memory_system._mock_ingestion.add_experience.call_args
        assert args[0] == experience
        assert "op" in kwargs

    def test_add_experience_invalid_type(self, memory_system):
        """Test add_experience with invalid experience type."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.add_experience("not an experience record")

        assert "Experience must be an ExperienceRecord instance" in str(exc_info.value)

    def test_add_experience_ingestion_error(self, memory_system):
        """Test add_experience when IngestionError is raised."""
        from gui_agent_memory.ingestion import IngestionError
        from gui_agent_memory.models import ActionStep, ExperienceRecord

        experience = ExperienceRecord(
            task_description="Test task",
            keywords=["test"],
            action_flow=[
                ActionStep(
                    thought="Test thought",
                    action="click",
                    target_element_description="button",
                )
            ],
            preconditions="Test preconditions",
            is_successful=True,
            source_task_id="task_123",
        )

        memory_system._mock_ingestion.add_experience.side_effect = IngestionError(
            "Test ingestion error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.add_experience(experience)

        assert "Adding experience failed" in str(exc_info.value)

    def test_batch_add_facts_success(self, memory_system):
        """Test successful batch_add_facts."""
        facts_data = [
            {
                "content": "Fact 1",
                "keywords": ["fact1"],
                "source": "test",
            },
            {
                "content": "Fact 2",
                "keywords": ["fact2"],
            },
        ]

        memory_system._mock_ingestion.batch_add_facts.return_value = [
            "fact_1",
            "fact_2",
        ]

        result = memory_system.batch_add_facts(facts_data)

        assert result == ["fact_1", "fact_2"]
        memory_system._mock_ingestion.batch_add_facts.assert_called_once_with(
            facts_data
        )

    def test_batch_add_facts_empty_data(self, memory_system):
        """Test batch_add_facts with empty facts_data."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.batch_add_facts([])

        assert "Facts data cannot be empty" in str(exc_info.value)

    def test_batch_add_facts_invalid_data_type(self, memory_system):
        """Test batch_add_facts with invalid fact data type."""
        facts_data = ["not a dict"]

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.batch_add_facts(facts_data)

        assert "Fact data at index 0 must be a dictionary" in str(exc_info.value)

    def test_batch_add_facts_missing_content(self, memory_system):
        """Test batch_add_facts with missing content in fact."""
        facts_data = [{"keywords": ["test"]}]

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.batch_add_facts(facts_data)

        assert "Fact at index 0 must have non-empty content" in str(exc_info.value)

    def test_batch_add_facts_empty_content(self, memory_system):
        """Test batch_add_facts with empty content in fact."""
        facts_data = [{"content": "", "keywords": ["test"]}]

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.batch_add_facts(facts_data)

        assert "Fact at index 0 must have non-empty content" in str(exc_info.value)

    def test_batch_add_facts_whitespace_content(self, memory_system):
        """Test batch_add_facts with whitespace-only content in fact."""
        facts_data = [{"content": "   ", "keywords": ["test"]}]

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.batch_add_facts(facts_data)

        assert "Fact at index 0 must have non-empty content" in str(exc_info.value)

    def test_batch_add_facts_ingestion_error(self, memory_system):
        """Test batch_add_facts when IngestionError is raised."""
        from gui_agent_memory.ingestion import IngestionError

        facts_data = [{"content": "Test fact", "keywords": ["test"]}]

        memory_system._mock_ingestion.batch_add_facts.side_effect = IngestionError(
            "Test ingestion error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.batch_add_facts(facts_data)

        assert "Batch adding facts failed" in str(exc_info.value)

    def test_get_similar_experiences_success(self, memory_system):
        """Test successful get_similar_experiences."""
        from gui_agent_memory.models import ExperienceRecord

        mock_experiences = [Mock(spec=ExperienceRecord), Mock(spec=ExperienceRecord)]
        memory_system._mock_retriever.get_similar_experiences.return_value = (
            mock_experiences
        )

        result = memory_system.get_similar_experiences("Test task", top_n=5)

        assert result == mock_experiences
        memory_system._mock_retriever.get_similar_experiences.assert_called_once_with(
            "Test task", 5
        )

    def test_get_similar_experiences_empty_task_description(self, memory_system):
        """Test get_similar_experiences with empty task_description."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.get_similar_experiences("", top_n=5)

        assert "Task description cannot be empty" in str(exc_info.value)

    def test_get_similar_experiences_whitespace_task_description(self, memory_system):
        """Test get_similar_experiences with whitespace-only task_description."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.get_similar_experiences("   ", top_n=5)

        assert "Task description cannot be empty" in str(exc_info.value)

    def test_get_similar_experiences_retrieval_error(self, memory_system):
        """Test get_similar_experiences when RetrievalError is raised."""
        from gui_agent_memory.retriever import RetrievalError

        memory_system._mock_retriever.get_similar_experiences.side_effect = (
            RetrievalError("Test retrieval error")
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.get_similar_experiences("Test task", top_n=5)

        assert "Getting similar experiences failed" in str(exc_info.value)

    def test_get_related_facts_success(self, memory_system):
        """Test successful get_related_facts."""
        from gui_agent_memory.models import FactRecord

        mock_facts = [Mock(spec=FactRecord), Mock(spec=FactRecord)]
        memory_system._mock_retriever.get_related_facts.return_value = mock_facts

        result = memory_system.get_related_facts("Test topic", top_n=5)

        assert result == mock_facts
        memory_system._mock_retriever.get_related_facts.assert_called_once_with(
            "Test topic", 5
        )

    def test_get_related_facts_empty_topic(self, memory_system):
        """Test get_related_facts with empty topic."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.get_related_facts("", top_n=5)

        assert "Topic cannot be empty" in str(exc_info.value)

    def test_get_related_facts_whitespace_topic(self, memory_system):
        """Test get_related_facts with whitespace-only topic."""
        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.get_related_facts("   ", top_n=5)

        assert "Topic cannot be empty" in str(exc_info.value)

    def test_get_related_facts_retrieval_error(self, memory_system):
        """Test get_related_facts when RetrievalError is raised."""
        from gui_agent_memory.retriever import RetrievalError

        memory_system._mock_retriever.get_related_facts.side_effect = RetrievalError(
            "Test retrieval error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.get_related_facts("Test topic", top_n=5)

        assert "Getting related facts failed" in str(exc_info.value)

    def test_get_system_stats_success(self, memory_system):
        """Test successful get_system_stats."""
        storage_stats = {"experiences_count": 10, "facts_count": 5}
        memory_system._mock_storage.get_collection_stats.return_value = storage_stats

        # Configure the mock config object
        memory_system.config.embedding_model = "test-embedding-model"
        memory_system.config.reranker_model = "test-reranker-model"
        memory_system.config.experience_llm_model = "test-llm-model"
        memory_system.config.chroma_db_path = "/test/path"

        result = memory_system.get_system_stats()

        expected = {
            "storage": storage_stats,
            "configuration": {
                "embedding_model": "test-embedding-model",
                "reranker_model": "test-reranker-model",
                "experience_llm_model": "test-llm-model",
                "chroma_db_path": "/test/path",
            },
            "version": "1.0.0",
        }

        assert result == expected
        memory_system._mock_storage.get_collection_stats.assert_called_once()

    def test_get_system_stats_storage_error(self, memory_system):
        """Test get_system_stats when StorageError is raised."""
        from gui_agent_memory.storage import StorageError

        memory_system._mock_storage.get_collection_stats.side_effect = StorageError(
            "Test storage error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.get_system_stats()

        assert "Getting system stats failed" in str(exc_info.value)

    def test_validate_system_success(self, memory_system):
        """Test successful validate_system."""
        memory_system.config.validate_configuration.return_value = True
        memory_system._mock_storage.get_collection_stats.return_value = {}

        result = memory_system.validate_system()

        assert result is True
        memory_system.config.validate_configuration.assert_called_once()
        memory_system._mock_storage.get_collection_stats.assert_called_once()

    def test_validate_system_configuration_error(self, memory_system):
        """Test validate_system when ConfigurationError is raised."""
        from gui_agent_memory.config import ConfigurationError

        memory_system.config.validate_configuration.side_effect = ConfigurationError(
            "Test configuration error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.validate_system()

        assert "System validation failed" in str(exc_info.value)

    def test_validate_system_storage_error(self, memory_system):
        """Test validate_system when StorageError is raised."""
        from gui_agent_memory.storage import StorageError

        memory_system.config.validate_configuration.return_value = True
        memory_system._mock_storage.get_collection_stats.side_effect = StorageError(
            "Test storage error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.validate_system()

        assert "System validation failed" in str(exc_info.value)

    def test_clear_all_memories_success(self, memory_system):
        """Test successful clear_all_memories."""
        result = memory_system.clear_all_memories()

        assert "Successfully cleared all memories" in result
        memory_system._mock_storage.clear_collections.assert_called_once()

    def test_clear_all_memories_storage_error(self, memory_system):
        """Test clear_all_memories when StorageError is raised."""
        from gui_agent_memory.storage import StorageError

        memory_system._mock_storage.clear_collections.side_effect = StorageError(
            "Test storage error"
        )

        with pytest.raises(MemorySystemError) as exc_info:
            memory_system.clear_all_memories()

        assert "Clearing memories failed" in str(exc_info.value)


class TestCreateMemorySystemFunction:
    """Test the create_memory_system convenience function."""

    def test_create_memory_system_success(self, mock_config):
        """Test successful create_memory_system."""
        with (
            patch("gui_agent_memory.main.get_config", return_value=mock_config),
            patch("gui_agent_memory.main.MemoryStorage"),
            patch("gui_agent_memory.main.MemoryIngestion"),
            patch("gui_agent_memory.main.MemoryRetriever"),
        ):
            result = create_memory_system()

            assert isinstance(result, MemorySystem)

    def test_create_memory_system_initialization_failure(self, mock_config):
        """Test create_memory_system when initialization fails."""
        with (
            patch("gui_agent_memory.main.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.main.MemoryStorage",
                side_effect=Exception("Test error"),
            ),
        ):
            with pytest.raises(MemorySystemError) as exc_info:
                create_memory_system()

            assert "Failed to initialize memory system" in str(exc_info.value)


class TestConfigurationDefaultValues:
    """Test that configuration default values are properly used."""

    @pytest.fixture
    def memory_system_with_custom_config(self):
        """Create MemorySystem with custom configuration."""
        from unittest.mock import Mock, patch

        from gui_agent_memory.config import MemoryConfig

        # Create a custom config with specific default_top_n
        custom_config = Mock(spec=MemoryConfig)
        custom_config.default_top_n = 2
        custom_config.default_top_k = 15
        custom_config.embedding_model = "test-model"
        custom_config.reranker_model = "test-reranker"
        custom_config.experience_llm_model = "test-llm"
        custom_config.chroma_db_path = "/test/path"
        custom_config.logs_base_dir = "./test_data/test_logs"
        custom_config.log_enabled = True

        with (
            patch("gui_agent_memory.main.get_config", return_value=custom_config),
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

    def test_retrieve_memories_uses_config_default_top_n(
        self, memory_system_with_custom_config
    ):
        """Test that retrieve_memories uses config.default_top_n when top_n not provided."""
        from gui_agent_memory.models import RetrievalResult

        mock_result = Mock(spec=RetrievalResult)
        memory_system_with_custom_config._mock_retriever.retrieve_memories.return_value = mock_result

        # Call without providing top_n - should use config.default_top_n (2)
        result = memory_system_with_custom_config.retrieve_memories("test query")

        assert result == mock_result
        memory_system_with_custom_config._mock_retriever.retrieve_memories.assert_called_once_with(
            "test query",
            2,  # Should use config.default_top_n
        )

    def test_retrieve_memories_uses_explicit_top_n_when_provided(
        self, memory_system_with_custom_config
    ):
        """Test that retrieve_memories uses explicit top_n when provided."""
        from gui_agent_memory.models import RetrievalResult

        mock_result = Mock(spec=RetrievalResult)
        memory_system_with_custom_config._mock_retriever.retrieve_memories.return_value = mock_result

        # Call with explicit top_n - should use the provided value
        result = memory_system_with_custom_config.retrieve_memories(
            "test query", top_n=7
        )

        assert result == mock_result
        memory_system_with_custom_config._mock_retriever.retrieve_memories.assert_called_once_with(
            "test query",
            7,  # Should use explicit value
        )

    def test_get_similar_experiences_uses_config_default_top_n(
        self, memory_system_with_custom_config
    ):
        """Test that get_similar_experiences uses config.default_top_n when top_n not provided."""
        from gui_agent_memory.models import ExperienceRecord

        mock_experiences = [Mock(spec=ExperienceRecord)]
        memory_system_with_custom_config._mock_retriever.get_similar_experiences.return_value = mock_experiences

        # Call without providing top_n - should use config.default_top_n (2)
        result = memory_system_with_custom_config.get_similar_experiences("test task")

        assert result == mock_experiences
        memory_system_with_custom_config._mock_retriever.get_similar_experiences.assert_called_once_with(
            "test task",
            2,  # Should use config.default_top_n
        )

    def test_get_similar_experiences_uses_explicit_top_n_when_provided(
        self, memory_system_with_custom_config
    ):
        """Test that get_similar_experiences uses explicit top_n when provided."""
        from gui_agent_memory.models import ExperienceRecord

        mock_experiences = [Mock(spec=ExperienceRecord)]
        memory_system_with_custom_config._mock_retriever.get_similar_experiences.return_value = mock_experiences

        # Call with explicit top_n - should use the provided value
        result = memory_system_with_custom_config.get_similar_experiences(
            "test task", top_n=8
        )

        assert result == mock_experiences
        memory_system_with_custom_config._mock_retriever.get_similar_experiences.assert_called_once_with(
            "test task",
            8,  # Should use explicit value
        )

    def test_get_related_facts_uses_config_default_top_n(
        self, memory_system_with_custom_config
    ):
        """Test that get_related_facts uses config.default_top_n when top_n not provided."""
        from gui_agent_memory.models import FactRecord

        mock_facts = [Mock(spec=FactRecord)]
        memory_system_with_custom_config._mock_retriever.get_related_facts.return_value = mock_facts

        # Call without providing top_n - should use config.default_top_n (2)
        result = memory_system_with_custom_config.get_related_facts("test topic")

        assert result == mock_facts
        memory_system_with_custom_config._mock_retriever.get_related_facts.assert_called_once_with(
            "test topic",
            2,  # Should use config.default_top_n
        )

    def test_get_related_facts_uses_explicit_top_n_when_provided(
        self, memory_system_with_custom_config
    ):
        """Test that get_related_facts uses explicit top_n when provided."""
        from gui_agent_memory.models import FactRecord

        mock_facts = [Mock(spec=FactRecord)]
        memory_system_with_custom_config._mock_retriever.get_related_facts.return_value = mock_facts

        # Call with explicit top_n - should use the provided value
        result = memory_system_with_custom_config.get_related_facts(
            "test topic", top_n=9
        )

        assert result == mock_facts
        memory_system_with_custom_config._mock_retriever.get_related_facts.assert_called_once_with(
            "test topic",
            9,  # Should use explicit value
        )
