"""
Test suite for the ingestion layer.
Tests experience learning and knowledge ingestion functionality.
"""

import json
from unittest.mock import Mock, mock_open, patch

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

    @pytest.fixture
    def complex_raw_history(self):
        """Complex raw history with various action types."""
        return [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "thought": "I need to navigate to the login page",
                "action": "navigate",
                "target": "https://example.com/login",
                "result": "success",
                "screenshot": "base64_image_data",
                "context": {
                    "page_title": "Login Page",
                    "url": "https://example.com/login",
                },
            },
            {
                "timestamp": "2024-01-01T10:00:05Z",
                "thought": "I should wait for the page to load completely",
                "action": "wait",
                "target": "login_form",
                "duration": 2.5,
                "result": "success",
                "context": {"element_visible": True},
            },
            {
                "timestamp": "2024-01-01T10:00:08Z",
                "thought": "Now I'll enter the username",
                "action": "type",
                "target": "username_field",
                "text": "user@example.com",
                "result": "success",
                "context": {"field_type": "email", "validation": "passed"},
            },
            {
                "timestamp": "2024-01-01T10:00:12Z",
                "thought": "I need to enter the password securely",
                "action": "type",
                "target": "password_field",
                "text": "***hidden***",
                "result": "success",
                "context": {"field_type": "password", "masked": True},
            },
            {
                "timestamp": "2024-01-01T10:00:15Z",
                "thought": "I should submit the login form",
                "action": "click",
                "target": "submit_button",
                "result": "success",
                "context": {"button_text": "Login", "form_valid": True},
            },
            {
                "timestamp": "2024-01-01T10:00:18Z",
                "thought": "I need to wait for the redirect after login",
                "action": "wait",
                "target": "dashboard",
                "duration": 3.0,
                "result": "success",
                "context": {"redirect_url": "https://example.com/dashboard"},
            },
        ]

    def test_prompt_template_loading_success(self, mock_config, mock_storage):
        """Test successful loading of prompt templates."""
        experience_prompt = "Experience distillation prompt: {raw_history}"
        keyword_prompt = "Keyword extraction prompt: {text}"

        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.return_value.read.side_effect = [
                experience_prompt,
                keyword_prompt,
            ]

            with (
                patch(
                    "gui_agent_memory.ingestion.get_config", return_value=mock_config
                ),
                patch(
                    "gui_agent_memory.ingestion.MemoryStorage",
                    return_value=mock_storage,
                ),
            ):
                ingestion = MemoryIngestion()

                assert ingestion.experience_distillation_prompt == experience_prompt
                assert ingestion.keyword_extraction_prompt == keyword_prompt

    def test_prompt_template_loading_failure(self, mock_config, mock_storage):
        """Test error handling when prompt templates fail to load."""
        from gui_agent_memory.ingestion import IngestionError

        with (
            patch(
                "gui_agent_memory.ingestion.get_config", return_value=mock_config
            ),
            patch(
                "gui_agent_memory.ingestion.MemoryStorage",
                return_value=mock_storage,
            ),
        ):
            # Mock the prompt file loading specifically to fail
            original_open = open
            def mock_open_func(path, *args, **kwargs):
                if "prompts/" in str(path):
                    raise FileNotFoundError("Prompt file not found")
                return original_open(path, *args, **kwargs)

            with patch("builtins.open", side_effect=mock_open_func):
                with pytest.raises(IngestionError) as exc_info:
                    MemoryIngestion()

                assert "Failed to load prompt templates" in str(exc_info.value)

    def test_learn_from_complex_task(self, ingestion, mock_config, complex_raw_history):
        """Test learning from complex task with multiple action types."""
        # Mock LLM response
        complex_experience = {
            "task_description": "Complete user login process with validation",
            "keywords": ["login", "authentication", "navigation", "form", "validation"],
            "action_flow": [
                {
                    "thought": "Navigate to login page",
                    "action": "navigate",
                    "target_element_description": "Login page URL",
                },
                {
                    "thought": "Wait for page load",
                    "action": "wait",
                    "target_element_description": "Login form container",
                },
                {
                    "thought": "Enter username",
                    "action": "type",
                    "target_element_description": "Username input field",
                },
                {
                    "thought": "Enter password",
                    "action": "type",
                    "target_element_description": "Password input field",
                },
                {
                    "thought": "Submit form",
                    "action": "click",
                    "target_element_description": "Submit button",
                },
                {
                    "thought": "Wait for redirect",
                    "action": "wait",
                    "target_element_description": "Dashboard page",
                },
            ],
            "preconditions": "User has valid credentials and internet connection",
        }

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(complex_experience)
        mock_config.get_experience_llm_client.return_value.chat.completions.create.return_value.choices = [
            mock_choice
        ]

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Execute
        result = ingestion.learn_from_task(
            raw_history=complex_raw_history,
            task_description="Complete user login with validation",
            is_successful=True,
            source_task_id="complex_task_001",
            app_name="WebApp",
        )

        # Verify
        assert "Successfully learned experience" in result
        ingestion.storage.add_experiences.assert_called_once()

        # Verify the experience record structure
        call_args = ingestion.storage.add_experiences.call_args
        experience_records = call_args[0][0]
        assert len(experience_records) == 1
        assert len(experience_records[0].action_flow) == 6

    def test_learn_from_failed_task(self, ingestion, mock_config):
        """Test learning from a failed task execution."""
        failed_history = [
            {
                "thought": "I'll try to click the login button",
                "action": "click",
                "target": "login_button",
                "result": "failed",
                "error": "Element not found",
            },
            {
                "thought": "Let me try a different selector",
                "action": "click",
                "target": "#login-btn",
                "result": "failed",
                "error": "Element still not found",
            },
        ]

        failed_experience = {
            "task_description": "Attempt to login (failed)",
            "keywords": ["login", "failed", "element", "selector"],
            "action_flow": [
                {
                    "thought": "Try to click login button",
                    "action": "click",
                    "target_element_description": "Login button (element not found)",
                },
                {
                    "thought": "Try alternative selector",
                    "action": "click",
                    "target_element_description": "Login button with CSS selector (still not found)",
                },
            ],
            "preconditions": "Login page should be loaded with login button visible",
        }

        mock_choice = Mock()
        mock_choice.message.content = json.dumps(failed_experience)
        mock_config.get_experience_llm_client.return_value.chat.completions.create.return_value.choices = [
            mock_choice
        ]

        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Execute
        result = ingestion.learn_from_task(
            raw_history=failed_history,
            task_description="Login attempt (failed)",
            is_successful=False,  # Failed task
            source_task_id="failed_task_001",
            app_name="WebApp",
        )

        # Should still learn from failed tasks
        assert "Successfully learned experience" in result

        # Verify the experience is marked as unsuccessful
        call_args = ingestion.storage.add_experiences.call_args
        experience_records = call_args[0][0]
        assert experience_records[0].is_successful is False

    def test_duplicate_task_handling(self, ingestion, mock_config, sample_raw_history):
        """Test handling of duplicate task IDs."""
        # Mock storage to indicate task already exists
        ingestion.storage.experience_exists.return_value = True

        # Execute
        result = ingestion.learn_from_task(
            raw_history=sample_raw_history,
            task_description="Duplicate task",
            is_successful=True,
            source_task_id="duplicate_task_001",
            app_name="WebApp",
        )

        # Should return early with message about existing experience
        assert "already exists" in result.lower() or "duplicate" in result.lower()

        # Should not add to storage
        ingestion.storage.add_experiences.assert_not_called()

    def test_batch_add_facts(self, ingestion, mock_config):
        """Test batch addition of multiple facts."""
        facts_data = [
            {"content": "Python is interpreted", "keywords": ["python", "interpreted"], "source": "doc1"},
            {"content": "JavaScript runs in browsers", "keywords": ["javascript", "browser"], "source": "doc2"},
            {"content": "SQL manages databases", "keywords": ["sql", "database"], "source": "doc3"},
        ]

        # Mock embedding generation for each fact
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock storage to return the correct number of IDs for batch operation
        ingestion.storage.add_facts.return_value = ["id_1", "id_2", "id_3"]

        # Add facts in batch using the actual batch method
        results = ingestion.batch_add_facts(facts_data)

        # Verify all succeeded
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)
        assert all("Successfully added fact" in result for result in results)
        # Should be called once for the batch operation
        assert ingestion.storage.add_facts.call_count == 1

    def test_unicode_content_in_fact(self, ingestion, mock_config):
        """Test handling of Unicode content in facts."""
        unicode_content = "Python支持Unicode字符串处理 🐍 with emojis and 特殊字符"

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        result = ingestion.add_fact(
            content=unicode_content,
            keywords=["python", "unicode", "字符串"],
            source="documentation",
        )

        # Should handle Unicode content without issues
        assert "Successfully added fact" in result

    def test_very_long_fact_content(self, ingestion, mock_config):
        """Test handling of very long fact content."""
        long_content = "This is a very long fact content. " * 1000  # Very long content

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        result = ingestion.add_fact(
            content=long_content, keywords=["long", "content"], source="test"
        )

        # Should handle long content
        assert "Successfully added fact" in result

    def test_empty_fact_content(self, ingestion):
        """Test handling of empty fact content."""
        from gui_agent_memory.ingestion import IngestionError

        with pytest.raises(IngestionError):
            ingestion.add_fact(
                content="",  # Empty content
                keywords=["empty"],
                source="test",
            )
