"""
Test suite for the ingestion layer.
Tests experience learning and knowledge ingestion functionality.
"""

import json
from unittest.mock import Mock, patch

import pytest

from gui_agent_memory.ingestion import MemoryIngestion


class TestMemoryIngestion:
    """Test cases for MemoryIngestion class."""

    @pytest.fixture
    def mock_storage(self):
        """Mock storage for testing."""
        storage = Mock()
        storage.query.return_value = []  # No existing records
        storage.add_facts.return_value = ["test_fact_id"]
        storage.add_experiences.return_value = ["test_exp_id"]
        storage.experience_exists.return_value = False
        return storage

    @pytest.fixture
    def ingestion(self, mock_config, mock_storage):
        """Create MemoryIngestion instance with mocked dependencies."""
        with (
            patch("gui_agent_memory.ingestion.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.ingestion.MemoryStorage", return_value=mock_storage
            ),
        ):
            return MemoryIngestion()

    @pytest.fixture
    def sample_raw_history(self):
        """Sample raw execution history for testing."""
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
        ]

    @pytest.fixture
    def mock_experience_response(self):
        """Mock response from experience distillation LLM."""
        return {
            "task_description": "User login process",
            "keywords": ["login", "authentication", "user"],
            "action_flow": [
                {
                    "thought": "I need to click the login button",
                    "action": "click",
                    "target_element_description": "login button",
                },
                {
                    "thought": "Now I'll enter the username",
                    "action": "type",
                    "target_element_description": "username input field",
                },
            ],
            "preconditions": "User must be on the login page",
        }

    @pytest.mark.unit
    def test_init(self, mock_config, mock_storage):
        """Test MemoryIngestion initialization."""
        with (
            patch("gui_agent_memory.ingestion.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.ingestion.MemoryStorage", return_value=mock_storage
            ),
        ):
            ingestion = MemoryIngestion()
            assert ingestion.storage == mock_storage
            assert ingestion.config == mock_config

    @pytest.mark.unit
    def test_generate_embedding(self, ingestion, mock_config):
        """Test embedding generation."""
        # Arrange
        text = "Test text for embedding"
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = mock_response

        # Act
        result = ingestion._generate_embedding(text)

        # Assert
        assert result == [0.1, 0.2, 0.3]
        mock_config.embedding_client.embeddings.create.assert_called_once()

    @pytest.mark.unit
    def test_learn_from_task_success(
        self, ingestion, mock_config, sample_raw_history, mock_experience_response
    ):
        """Test successful learning from task execution."""
        # Arrange
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(mock_experience_response)
        mock_config.experience_llm_client.chat.completions.create.return_value.choices = [
            mock_choice
        ]

        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Act
        result = ingestion.learn_from_task(
            raw_history=sample_raw_history,
            task_description="Login to application",
            is_successful=True,
            source_task_id="task_123",
            app_name="TestApp",
        )

        # Assert - check that it returns a success message, not just the task ID
        assert "Successfully learned experience from task 'task_123'" in result
        ingestion.storage.add_experiences.assert_called_once()

    @pytest.mark.unit
    def test_add_fact_success(self, ingestion, mock_config):
        """Test adding fact record."""
        # Arrange
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_config.embedding_client.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Act
        result = ingestion.add_fact(
            content="Python is a programming language",
            keywords=["python", "programming", "language"],
            source="manual",
        )

        # Assert
        assert result is not None
        assert "Successfully added fact" in result
        ingestion.storage.add_facts.assert_called_once()
