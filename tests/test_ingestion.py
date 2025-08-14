"""
Test suite for the ingestion layer.
Tests experience learning and knowledge ingestion functionality.
"""

import json
from unittest.mock import Mock, patch

import pytest

from gui_agent_memory.ingestion import IngestionError, MemoryIngestion
from gui_agent_memory.models import ActionStep, ExperienceRecord


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
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )
        # Set the threshold to a real number
        mock_config.similarity_threshold_judge = 0.9

        # Mock the storage methods
        from gui_agent_memory.models import StoredFact

        mock_stored_fact = StoredFact(
            record_id="fact_123",
            document="Python is a programming language",
            metadata={"source": "manual", "keywords": "python,programming,language"},
            embedding=[0.1, 0.2, 0.3],
        )
        ingestion.storage.add_fact.return_value = mock_stored_fact
        ingestion.storage.compute_fact_output_fp.return_value = "test_fp"
        ingestion.storage.fact_exists_by_output_fp.return_value = False

        # Act - use the upsert_fact_with_policy method which is the current interface
        from gui_agent_memory.models import FactRecord

        fact = FactRecord(
            content="Python is a programming language",
            keywords=["python", "programming", "language"],
            source="manual",
        )
        result = ingestion.upsert_fact_with_policy(fact)

        # Assert
        assert result is not None
        assert result.result in [
            "added_new",
            "discarded_by_fingerprint",
            "updated_existing",
            "kept_new_deleted_old",
            "kept_old_discarded_new",
        ]
        # Check that storage.add_fact was called (not add_facts)
        ingestion.storage.add_fact.assert_called_once()

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
        """Test successful loading of prompt templates (three files)."""
        experience_prompt = "Experience distillation prompt: {raw_history}"
        keyword_prompt = "Keyword extraction prompt: {text}"
        judge_prompt = '{"decision":"add_new","target_id":null,"updated_record":null,"reason":"ok"}'

        with (
            patch("gui_agent_memory.ingestion.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.ingestion.MemoryStorage",
                return_value=mock_storage,
            ),
            patch("pathlib.Path.read_text") as mock_read_text,
        ):
            mock_read_text.side_effect = [
                experience_prompt,
                keyword_prompt,
                judge_prompt,
            ]
            ingestion = MemoryIngestion()

            assert ingestion.experience_distillation_prompt == experience_prompt
            assert ingestion.keyword_extraction_prompt == keyword_prompt
            assert ingestion.judge_decision_prompt == judge_prompt

    # Removed: external prompt directory feature no longer supported; test deleted.

    def test_prompt_template_loading_failure(self, mock_config, mock_storage):
        """Test error handling when prompt templates fail to load."""
        from gui_agent_memory.ingestion import IngestionError

        with (
            patch("gui_agent_memory.ingestion.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.ingestion.MemoryStorage",
                return_value=mock_storage,
            ),
            patch(
                "pathlib.Path.read_text",
                side_effect=FileNotFoundError("Prompt file not found"),
            ),
        ):
            with pytest.raises(IngestionError) as exc_info:
                MemoryIngestion()

            # New message after consolidation
            assert "Failed to load bundled prompt templates" in str(exc_info.value)

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

    def test_duplicate_task_handling(self, ingestion, sample_raw_history):
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
            {
                "content": "Python is interpreted",
                "keywords": ["python", "interpreted"],
                "source": "doc1",
            },
            {
                "content": "JavaScript runs in browsers",
                "keywords": ["javascript", "browser"],
                "source": "doc2",
            },
            {
                "content": "SQL manages databases",
                "keywords": ["sql", "database"],
                "source": "doc3",
            },
        ]

        # Set the threshold to a real number
        mock_config.similarity_threshold_judge = 0.9

        # Mock embedding generation for each fact
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock the storage methods
        from gui_agent_memory.models import StoredFact

        mock_stored_fact = StoredFact(
            record_id="fact_123",
            document="test content",
            metadata={"source": "test"},
            embedding=[0.1] * 1024,
        )
        ingestion.storage.add_fact.return_value = mock_stored_fact
        ingestion.storage.compute_fact_output_fp.return_value = "test_fp"
        ingestion.storage.fact_exists_by_output_fp.return_value = False

        # Add facts in batch using the actual batch method
        results = ingestion.batch_add_facts(facts_data)

        # Verify all succeeded (new behavior: per-item upsert)
        assert len(results) == 3
        assert all(isinstance(result, str) for result in results)
        assert all(
            "Successfully added fact" in result or "discarded" in result
            for result in results
        )
        # add_fact should be called for each item since we're using the new policy
        assert ingestion.storage.add_fact.call_count == 3

    def test_unicode_content_in_fact(self, ingestion, mock_config):
        """Test handling of Unicode content in facts."""
        unicode_content = "Python支持Unicode字符串处理 🐍 with emojis and 特殊字符"

        # Set the threshold to a real number
        mock_config.similarity_threshold_judge = 0.9

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock the storage methods
        from gui_agent_memory.models import StoredFact

        mock_stored_fact = StoredFact(
            record_id="fact_123",
            document=unicode_content,
            metadata={"source": "documentation"},
            embedding=[0.1] * 1024,
        )
        ingestion.storage.add_fact.return_value = mock_stored_fact
        ingestion.storage.compute_fact_output_fp.return_value = "test_fp"
        ingestion.storage.fact_exists_by_output_fp.return_value = False

        # Act - Use batch_add_facts which is available in the current interface
        result = ingestion.batch_add_facts(
            [
                {
                    "content": unicode_content,
                    "keywords": ["python", "unicode", "字符串"],
                    "source": "documentation",
                }
            ]
        )

        # Should handle Unicode content without issues
        assert len(result) == 1
        assert "Successfully added fact" in result[0] or "discarded" in result[0]

    def test_very_long_fact_content(self, ingestion, mock_config):
        """Test handling of very long fact content."""
        long_content = "This is a very long fact content. " * 1000  # Very long content

        # Set the threshold to a real number
        mock_config.similarity_threshold_judge = 0.9

        # Mock embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock the storage methods
        from gui_agent_memory.models import StoredFact

        mock_stored_fact = StoredFact(
            record_id="fact_123",
            document=long_content[:100] + "...",
            metadata={"source": "test"},
            embedding=[0.1] * 1024,
        )
        ingestion.storage.add_fact.return_value = mock_stored_fact
        ingestion.storage.compute_fact_output_fp.return_value = "test_fp"
        ingestion.storage.fact_exists_by_output_fp.return_value = False

        # Act - Use batch_add_facts which is available in the current interface
        result = ingestion.batch_add_facts(
            [
                {
                    "content": long_content,
                    "keywords": ["long", "content"],
                    "source": "test",
                }
            ]
        )

        # Should handle long content
        assert len(result) == 1
        assert "Successfully added fact" in result[0] or "discarded" in result[0]

    def test_empty_fact_content(self, ingestion):
        """Test handling of empty fact content."""
        from gui_agent_memory.ingestion import IngestionError

        with pytest.raises(IngestionError):
            ingestion.batch_add_facts(
                [
                    {
                        "content": "",  # Empty content
                        "keywords": ["empty"],
                        "source": "test",
                    }
                ]
            )


class TestMemoryIngestionAdvanced:
    """Advanced test cases for MemoryIngestion error handling and edge cases."""

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

    def test_extract_keywords_with_llm_success(self, ingestion, mock_config):
        """Test successful keyword extraction with LLM."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["login", "authentication", "user"]'

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_config.get_experience_llm_client.return_value = mock_client

        result = ingestion._extract_keywords_with_llm("How to login to the system")

        assert result == ["login", "authentication", "user"]
        mock_client.chat.completions.create.assert_called_once()

    def test_extract_keywords_with_llm_empty_response(self, ingestion, mock_config):
        """Test LLM keyword extraction when LLM returns empty response."""
        # Mock LLM response with None content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_config.get_experience_llm_client.return_value = mock_client

        # Should fallback to jieba when LLM returns empty response
        result = ingestion._extract_keywords_with_llm("test query")

        assert isinstance(result, list)
        # Should contain keywords extracted by jieba
        assert len(result) > 0

    def test_extract_keywords_with_llm_invalid_json(self, ingestion, mock_config):
        """Test LLM keyword extraction with invalid JSON fallback to jieba."""
        # Mock LLM response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json response"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_config.get_experience_llm_client.return_value = mock_client

        # Should fallback to jieba
        result = ingestion._extract_keywords_with_llm("login to system")

        assert isinstance(result, list)
        assert len(result) > 0  # Should extract some keywords with jieba

    def test_extract_keywords_with_llm_api_error(self, ingestion, mock_config):
        """Test LLM keyword extraction when API fails, fallback to jieba."""
        # Mock LLM client to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_config.get_experience_llm_client.return_value = mock_client

        # Should fallback to jieba
        result = ingestion._extract_keywords_with_llm("login to system")

        assert isinstance(result, list)
        assert len(result) > 0  # Should extract some keywords with jieba

    def test_extract_keywords_with_jieba_chinese_text(self, ingestion):
        """Test jieba keyword extraction with Chinese text."""
        chinese_text = "如何使用这个应用程序进行登录"

        result = ingestion._extract_keywords_with_jieba(chinese_text)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should filter out common stop words
        assert "的" not in result
        assert "是" not in result

    def test_extract_keywords_with_jieba_mixed_language(self, ingestion):
        """Test jieba keyword extraction with mixed language text."""
        mixed_text = "How to login 如何登录 to the application"

        result = ingestion._extract_keywords_with_jieba(mixed_text)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should filter out English stop words
        assert "the" not in result
        assert "to" not in result

    def test_extract_keywords_with_jieba_short_tokens(self, ingestion):
        """Test jieba keyword extraction filters out short tokens."""
        text = "a b c login system test"

        result = ingestion._extract_keywords_with_jieba(text)

        # Should filter out single character tokens
        assert "a" not in result
        assert "b" not in result
        assert "c" not in result
        # Should keep longer tokens
        assert "login" in result
        assert "system" in result
        assert "test" in result

    def test_distill_experience_with_llm_success(self, ingestion, mock_config):
        """Test successful experience distillation with LLM."""
        from gui_agent_memory.models import LearningRequest

        # Mock LLM response
        distilled_experience = {
            "task_description": "Login to application",
            "keywords": ["login", "authentication"],
            "action_flow": [
                {
                    "thought": "Click login button",
                    "action": "click",
                    "target_element_description": "login button",
                }
            ],
            "preconditions": "User has valid credentials",
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(distilled_experience)

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_config.get_experience_llm_client.return_value = mock_client

        learning_request = LearningRequest(
            raw_history=[{"action": "click", "target": "button"}],
            task_description="Test task",
            is_successful=True,
            source_task_id="test_123",
            app_name="TestApp",
        )

        result = ingestion._distill_experience_with_llm(learning_request)

        assert result == distilled_experience
        mock_client.chat.completions.create.assert_called_once()

    def test_distill_experience_with_llm_empty_response(self, ingestion, mock_config):
        """Test experience distillation when LLM returns empty response."""
        from gui_agent_memory.ingestion import IngestionError
        from gui_agent_memory.models import LearningRequest

        # Mock LLM response with None content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_config.get_experience_llm_client.return_value = mock_client

        learning_request = LearningRequest(
            raw_history=[{"action": "click", "target": "button"}],
            task_description="Test task",
            is_successful=True,
            source_task_id="test_123",
            app_name="TestApp",
        )

        with pytest.raises(IngestionError) as exc_info:
            ingestion._distill_experience_with_llm(learning_request)

        assert "LLM returned empty response" in str(exc_info.value)

    def test_distill_experience_with_llm_json_in_markdown(self, ingestion, mock_config):
        """Test experience distillation with JSON wrapped in markdown code blocks."""
        from gui_agent_memory.models import LearningRequest

        distilled_experience = {
            "task_description": "Login to application",
            "keywords": ["login", "authentication"],
            "action_flow": [],
            "preconditions": "User has credentials",
        }

        # Mock LLM response with markdown code blocks
        markdown_response = f"Here's the distilled experience:\n```json\n{json.dumps(distilled_experience)}\n```\nEnd of response."

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = markdown_response

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_config.get_experience_llm_client.return_value = mock_client

        learning_request = LearningRequest(
            raw_history=[{"action": "click", "target": "button"}],
            task_description="Test task",
            is_successful=True,
            source_task_id="test_123",
            app_name="TestApp",
        )

        result = ingestion._distill_experience_with_llm(learning_request)

        assert result == distilled_experience

    def test_distill_experience_with_llm_invalid_json(self, ingestion, mock_config):
        """Test experience distillation when LLM returns invalid JSON."""
        from gui_agent_memory.ingestion import IngestionError
        from gui_agent_memory.models import LearningRequest

        # Mock LLM response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json response"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_config.get_experience_llm_client.return_value = mock_client

        learning_request = LearningRequest(
            raw_history=[{"action": "click", "target": "button"}],
            task_description="Test task",
            is_successful=True,
            source_task_id="test_123",
            app_name="TestApp",
        )

        with pytest.raises(IngestionError) as exc_info:
            ingestion._distill_experience_with_llm(learning_request)

        assert "Failed to distill experience with LLM" in str(exc_info.value)

    def test_distill_experience_with_llm_api_error(self, ingestion, mock_config):
        """Test experience distillation when LLM API fails."""
        from gui_agent_memory.ingestion import IngestionError
        from gui_agent_memory.models import LearningRequest

        # Mock LLM client to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_config.get_experience_llm_client.return_value = mock_client

        learning_request = LearningRequest(
            raw_history=[{"action": "click", "target": "button"}],
            task_description="Test task",
            is_successful=True,
            source_task_id="test_123",
            app_name="TestApp",
        )

        with pytest.raises(IngestionError) as exc_info:
            ingestion._distill_experience_with_llm(learning_request)

        assert "Failed to distill experience with LLM" in str(exc_info.value)

    def test_learn_from_task_error_logging(self, ingestion, mock_config):
        """Test that learning failures are properly logged."""
        from gui_agent_memory.ingestion import IngestionError

        # Mock LLM client to raise an exception
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("LLM API Error")
        mock_config.get_experience_llm_client.return_value = mock_client

        with patch("gui_agent_memory.ingestion.logger") as mock_logger:
            # This should handle the exception and log it
            try:
                ingestion.learn_from_task(
                    raw_history=[{"action": "click", "target": "button"}],
                    is_successful=True,
                    source_task_id="test_task_error",
                    app_name="TestApp",
                )
            except IngestionError:
                pass  # Expected

            # Verify error was logged
            mock_logger.error.assert_called()

    def test_add_experience_with_existing_record(self, ingestion):
        """Test adding experience when record with same source_task_id exists."""
        from gui_agent_memory.models import ActionStep, ExperienceRecord

        # Mock storage to indicate experience exists
        ingestion.storage.experience_exists.return_value = True

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
            source_task_id="existing_task_123",
        )

        result = ingestion.add_experience(experience)

        assert "already exists" in result.lower()
        # Should not call storage.add_experiences
        ingestion.storage.add_experiences.assert_not_called()

    def test_generate_embedding_api_error(self, ingestion, mock_config):
        """Test embedding generation when API fails."""
        from gui_agent_memory.ingestion import IngestionError

        # Mock embedding client to raise an exception
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_config.get_embedding_client.return_value = mock_client

        with pytest.raises(IngestionError) as exc_info:
            ingestion._generate_embedding("test content")

        assert "Failed to generate embedding" in str(exc_info.value)

    def test_batch_add_facts_with_validation_errors(self, ingestion, mock_config):
        """Test batch_add_facts with validation errors in some facts."""
        from gui_agent_memory.ingestion import IngestionError

        facts_data = [
            {
                "content": "Valid fact 1",
                "keywords": ["valid", "fact1"],
                "source": "test1",
            },
            {
                "content": "",  # Invalid: empty content
                "keywords": ["invalid"],
                "source": "test2",
            },
            {
                "content": "Valid fact 2",
                "keywords": ["valid", "fact2"],
                "source": "test3",
            },
        ]

        # Should raise error for invalid facts
        with pytest.raises(IngestionError) as exc_info:
            ingestion.batch_add_facts(facts_data)

        assert "Content cannot be empty" in str(exc_info.value)

    def test_batch_add_facts_embedding_error(self, ingestion, mock_config):
        """Test batch_add_facts when embedding generation fails."""
        from gui_agent_memory.ingestion import IngestionError

        facts_data = [
            {
                "content": "Valid fact",
                "keywords": ["valid"],
                "source": "test",
            }
        ]

        # Set the threshold to a real number
        mock_config.similarity_threshold_judge = 0.9

        # Mock storage fingerprint methods to bypass fingerprint checks
        ingestion.storage.compute_fact_output_fp.return_value = "test_fp"
        ingestion.storage.fact_exists_by_output_fp.return_value = False

        # Mock embedding client to raise an exception
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("Embedding API Error")
        mock_config.get_embedding_client.return_value = mock_client

        with pytest.raises(IngestionError) as exc_info:
            ingestion.batch_add_facts(facts_data)

        assert "Failed to generate embedding" in str(exc_info.value)

    def test_batch_add_facts_storage_error(self, ingestion, mock_config):
        """Test batch_add_facts when storage operation fails."""
        from gui_agent_memory.ingestion import IngestionError
        from gui_agent_memory.storage import StorageError

        facts_data = [
            {
                "content": "Valid fact",
                "keywords": ["valid"],
                "source": "test",
            }
        ]

        # Set the threshold to a real number
        mock_config.similarity_threshold_judge = 0.9

        # Mock successful embedding generation
        mock_embedding_response = Mock()
        mock_embedding_response.data = [Mock(embedding=[0.1] * 1024)]
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            mock_embedding_response
        )

        # Mock the storage methods to avoid validation errors
        from gui_agent_memory.models import StoredFact

        mock_stored_fact = StoredFact(
            record_id="fact_123",
            document="Valid fact",
            metadata={"source": "test"},
            embedding=[0.1] * 1024,
        )
        ingestion.storage.add_fact.return_value = mock_stored_fact
        ingestion.storage.compute_fact_output_fp.return_value = "test_fp"
        ingestion.storage.fact_exists_by_output_fp.return_value = False

        # Mock storage to raise an error on add operation
        ingestion.storage.add_fact.side_effect = StorageError("Storage failed")

        with pytest.raises(IngestionError) as exc_info:
            ingestion.batch_add_facts(facts_data)

        assert "Failed to add facts to storage" in str(exc_info.value)


class TestIngestionCoverage:
    """Tests for uncovered code paths in ingestion.py"""

    @pytest.fixture
    def ingestion(self, mock_config, mock_storage):
        """Create MemoryIngestion instance for testing."""
        with patch("gui_agent_memory.ingestion.get_config", return_value=mock_config):
            return MemoryIngestion(mock_storage)

    def test_safe_slug_empty_text(self, ingestion):
        """Test _safe_slug with empty text (line 94)."""
        # Test empty string
        result = ingestion._safe_slug("")
        assert result == "unknown"

        # Test None
        result = ingestion._safe_slug(None)
        assert result == "unknown"

        # Test whitespace only
        result = ingestion._safe_slug("   ")
        assert result == "unknown"

    def test_write_text_file_exception(self, ingestion, tmp_path):
        """Test _write_text_file exception handling (cross-platform)."""
        test_file = tmp_path / "readonly" / "test.txt"

        # Force write failure by patching Path.write_text
        with patch("pathlib.Path.write_text", side_effect=PermissionError("denied")):
            with patch.object(ingestion, "logger") as mock_logger:
                # This should not raise an exception, just log it
                ingestion._write_text_file(test_file, "test content")

                # Verify that exception was logged
                mock_logger.exception.assert_called_once()
                assert "Failed to write log file" in str(
                    mock_logger.exception.call_args[0][0]
                )

    def test_add_experience_existing_record_skip(self, ingestion, mock_config):
        """Test add_experience when record already exists (lines 466-474)."""
        # Create sample experience
        experience = ExperienceRecord(
            task_description="Test task",
            keywords=["test"],
            action_flow=[
                ActionStep(
                    thought="test thought",
                    action="test action",
                    target_element_description="test target",
                )
            ],
            preconditions="test preconditions",
            is_successful=True,
            source_task_id="existing_task_123",
        )

        # Mock storage to return True for experience_exists
        ingestion.storage.experience_exists.return_value = True

        result = ingestion.add_experience(experience)

        assert "already exists, skipping" in result
        assert "existing_task_123" in result

        # Verify that experience_exists was called but add_experiences was not
        ingestion.storage.experience_exists.assert_called_once_with("existing_task_123")
        ingestion.storage.add_experiences.assert_not_called()

    def test_add_experience_success_path(self, ingestion, mock_config):
        """Test successful add_experience path."""
        # Create sample experience
        experience = ExperienceRecord(
            task_description="Test task",
            keywords=["test"],
            action_flow=[
                ActionStep(
                    thought="test thought",
                    action="test action",
                    target_element_description="test target",
                )
            ],
            preconditions="test preconditions",
            is_successful=True,
            source_task_id="new_task_456",
        )

        # Mock storage to return False for experience_exists (new record)
        ingestion.storage.experience_exists.return_value = False
        ingestion.storage.add_experiences.return_value = ["record_id_123"]

        # Mock embedding generation
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        )

        result = ingestion.add_experience(experience)

        assert "Successfully added experience" in result
        assert "new_task_456" in result
        assert "record_id_123" in result

        # Verify the full flow was executed
        ingestion.storage.experience_exists.assert_called_once_with("new_task_456")
        ingestion.storage.add_experiences.assert_called_once()

    def test_add_experience_embedding_error(self, ingestion, mock_config):
        """Test add_experience when embedding generation fails."""
        experience = ExperienceRecord(
            task_description="Test task",
            keywords=["test"],
            action_flow=[
                ActionStep(
                    thought="test thought",
                    action="test action",
                    target_element_description="test target",
                )
            ],
            preconditions="test preconditions",
            is_successful=True,
            source_task_id="error_task_789",
        )

        # Mock storage to return False for experience_exists
        ingestion.storage.experience_exists.return_value = False

        # Mock embedding client to raise an exception
        mock_config.get_embedding_client.return_value.embeddings.create.side_effect = (
            Exception("Embedding API error")
        )

        with pytest.raises(IngestionError) as exc_info:
            ingestion.add_experience(experience)

        assert "Failed to add experience" in str(exc_info.value)
        ingestion.storage.experience_exists.assert_called_once_with("error_task_789")

    def test_add_experience_storage_error(self, ingestion, mock_config):
        """Test add_experience when storage operation fails."""
        experience = ExperienceRecord(
            task_description="Test task",
            keywords=["test"],
            action_flow=[
                ActionStep(
                    thought="test thought",
                    action="test action",
                    target_element_description="test target",
                )
            ],
            preconditions="test preconditions",
            is_successful=True,
            source_task_id="storage_error_task",
        )

        # Mock storage to return False for experience_exists but fail on add
        ingestion.storage.experience_exists.return_value = False
        ingestion.storage.add_experiences.side_effect = Exception("Storage error")

        # Mock successful embedding generation
        mock_config.get_embedding_client.return_value.embeddings.create.return_value = (
            Mock(data=[Mock(embedding=[0.1, 0.2, 0.3])])
        )

        with pytest.raises(IngestionError) as exc_info:
            ingestion.add_experience(experience)

        assert "Failed to add experience" in str(exc_info.value)


class TestUpsertPolicy:
    """Additional coverage for upsert_fact_with_policy branches."""

    @pytest.fixture
    def mock_storage(self):
        from gui_agent_memory.models import StoredFact

        storage = Mock()
        storage.add_facts.return_value = ["fact_id"]
        storage.add_fact.return_value = StoredFact(
            record_id="fact_id",
            document="test content",
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3],
        )
        storage.update_fact = Mock()
        return storage

    @pytest.fixture
    def ingestion(self, mock_config, mock_storage):
        with (
            patch("gui_agent_memory.ingestion.get_config", return_value=mock_config),
            patch(
                "gui_agent_memory.ingestion.MemoryStorage", return_value=mock_storage
            ),
        ):
            return MemoryIngestion()

    def test_discard_by_fingerprint(self, ingestion, mock_config):
        """Existing output_fp should short-circuit to discard before add/judge."""
        from gui_agent_memory.models import FactRecord

        mock_config.similarity_threshold_judge = 0.8
        # Use real callables (not Mock) so pre-check path executes before embedding
        ingestion.storage.compute_fact_output_fp = lambda _fact: "fp_same"
        ingestion.storage.fact_exists_by_output_fp = lambda _fp: True

        dbg = ingestion.upsert_fact_with_policy(
            FactRecord(content="dup", keywords=["k"], source="t")
        )

        assert dbg.result == "discarded_by_fingerprint"
        assert dbg.fingerprint_discarded is True
        ingestion.storage.add_facts.assert_not_called()

    def test_add_below_threshold(self, ingestion, mock_config):
        """Below threshold → add_new and pass output_fps through to storage."""
        from gui_agent_memory.models import FactRecord

        mock_config.similarity_threshold_judge = 0.9
        ingestion.storage.compute_fact_output_fp = Mock(return_value="fp_new")
        ingestion.storage.fact_exists_by_output_fp = Mock(return_value=False)
        with patch.object(
            ingestion, "_generate_embedding", return_value=[0.1, 0.2, 0.3]
        ):
            with patch.object(
                ingestion, "_top1_similarity_fact", return_value=(None, 0.1)
            ):
                ingestion.storage.add_facts.return_value = ["fact_123"]
                dbg = ingestion.upsert_fact_with_policy(
                    FactRecord(content="new", keywords=["k"], source="t")
                )

        assert dbg.result == "added_new"
        ingestion.storage.add_fact.assert_called_once()
        # Check that output_fp was passed correctly
        call_args = ingestion.storage.add_fact.call_args
        assert call_args.kwargs["output_fp"] == "fp_new"

    def test_judge_add_new_but_fp_recheck_discards(self, ingestion, mock_config):
        """Above threshold + judge says add_new, but fp safety recheck discards before add."""
        from gui_agent_memory.models import FactRecord

        mock_config.similarity_threshold_judge = 0.5
        ingestion.storage.compute_fact_output_fp = lambda _fact: "fp_safe"
        # Ensure the safety re-check (the first actual exists call) returns True
        ingestion.storage.fact_exists_by_output_fp = Mock(return_value=True)

        ingestion.storage.declarative_collection = Mock()
        ingestion.storage.declarative_collection.get.return_value = {
            "documents": ["old content"],
            "metadatas": [{"source": "s"}],
        }

        with (
            patch.object(ingestion, "_generate_embedding", return_value=[0.1] * 4),
            patch.object(
                ingestion, "_top1_similarity_fact", return_value=("old_id", 0.9)
            ),
            patch.object(
                ingestion,
                "_call_llm_judge",
                return_value={"decision": "add_new", "reason": "ok"},
            ),
        ):
            dbg = ingestion.upsert_fact_with_policy(
                FactRecord(content="new", keywords=["k"], source="t")
            )

        assert dbg.result == "discarded_by_fingerprint"
        ingestion.storage.add_fact.assert_not_called()

    def test_judge_keep_new_delete_old(self, ingestion, mock_config):
        """Judge decides keep_new_delete_old → add new then delete old."""
        from gui_agent_memory.models import FactRecord, StoredFact

        mock_config.similarity_threshold_judge = 0.5
        ingestion.storage.compute_fact_output_fp = lambda _fact: "fp_knew"
        ingestion.storage.fact_exists_by_output_fp = Mock(return_value=False)
        ingestion.storage.declarative_collection = Mock()
        ingestion.storage.declarative_collection.get.return_value = {
            "documents": ["old content"],
            "metadatas": [{"source": "s"}],
        }
        ingestion.storage.add_fact.return_value = StoredFact(
            record_id="new_id",
            document="new content",
            metadata={"source": "t"},
            embedding=[0.3] * 4,
        )

        with (
            patch.object(ingestion, "_generate_embedding", return_value=[0.3] * 4),
            patch.object(
                ingestion, "_top1_similarity_fact", return_value=("old_id", 0.96)
            ),
            patch.object(
                ingestion,
                "_call_llm_judge",
                return_value={"decision": "keep_new_delete_old", "reason": "swap"},
            ),
        ):
            dbg = ingestion.upsert_fact_with_policy(
                FactRecord(content="new", keywords=["k"], source="t")
            )

        assert dbg.result == "kept_new_deleted_old"
        ingestion.storage.add_fact.assert_called_once()
        ingestion.storage.delete_records.assert_called_once()

    def test_judge_keep_old_delete_new(self, ingestion, mock_config):
        """Judge decides keep_old_delete_new → discard new, no add."""
        from gui_agent_memory.models import FactRecord

        mock_config.similarity_threshold_judge = 0.5
        ingestion.storage.compute_fact_output_fp = lambda _fact: "fp_kold"
        ingestion.storage.fact_exists_by_output_fp = Mock(return_value=False)
        ingestion.storage.declarative_collection = Mock()
        ingestion.storage.declarative_collection.get.return_value = {
            "documents": ["old content"],
            "metadatas": [{"source": "s"}],
        }

        with (
            patch.object(ingestion, "_generate_embedding", return_value=[0.4] * 4),
            patch.object(
                ingestion, "_top1_similarity_fact", return_value=("old_id", 0.97)
            ),
            patch.object(
                ingestion,
                "_call_llm_judge",
                return_value={
                    "decision": "keep_old_delete_new",
                    "reason": "old better",
                },
            ),
        ):
            dbg = ingestion.upsert_fact_with_policy(
                FactRecord(content="new", keywords=["k"], source="t")
            )

        assert dbg.result == "kept_old_discarded_new"
        ingestion.storage.add_facts.assert_not_called()

    def test_judge_update_existing_invalid_payload_fallback_add_new(
        self, ingestion, mock_config
    ):
        """Invalid updated_record should fallback to add_new path without crashing."""
        from gui_agent_memory.models import FactRecord

        mock_config.similarity_threshold_judge = 0.5
        ingestion.storage.compute_fact_output_fp = lambda _fact: "fp_inv"
        ingestion.storage.fact_exists_by_output_fp = Mock(return_value=False)
        ingestion.storage.declarative_collection = Mock()
        ingestion.storage.declarative_collection.get.return_value = {
            "documents": ["old content"],
            "metadatas": [{"source": "s"}],
        }

        with (
            patch.object(ingestion, "_generate_embedding", return_value=[0.5] * 4),
            patch.object(
                ingestion, "_top1_similarity_fact", return_value=("old_id", 0.95)
            ),
            patch.object(
                ingestion,
                "_call_llm_judge",
                return_value={"decision": "update_existing", "updated_record": {}},
            ),
        ):
            dbg = ingestion.upsert_fact_with_policy(
                FactRecord(content="new", keywords=["k"], source="t")
            )

        assert dbg.result == "added_new"
        ingestion.storage.add_facts.assert_called()

    def test_judge_update_existing(self, ingestion, mock_config):
        """Judge decides update_existing → call storage.update_fact with new embedding."""
        from gui_agent_memory.models import FactRecord

        mock_config.similarity_threshold_judge = 0.5
        ingestion.storage.compute_fact_output_fp = Mock(return_value="fp_u")
        ingestion.storage.fact_exists_by_output_fp = Mock(return_value=False)

        ingestion.storage.declarative_collection = Mock()
        ingestion.storage.declarative_collection.get.return_value = {
            "documents": ["old content"],
            "metadatas": [{"source": "s"}],
        }

        updated = {"content": "updated", "keywords": ["a"], "source": "t"}
        with (
            patch.object(ingestion, "_generate_embedding", return_value=[0.2] * 4),
            patch.object(
                ingestion, "_top1_similarity_fact", return_value=("old_id", 0.95)
            ),
            patch.object(
                ingestion,
                "_call_llm_judge",
                return_value={"decision": "update_existing", "updated_record": updated},
            ),
        ):
            dbg = ingestion.upsert_fact_with_policy(
                FactRecord(content="new", keywords=["k"], source="t")
            )

        assert dbg.result in {"updated_existing", "update_existing", "updated"}
        ingestion.storage.update_fact.assert_called_once()
