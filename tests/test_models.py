"""
Unit tests for the data models module.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from gui_agent_memory.models import (
    ActionStep,
    ExperienceRecord,
    FactRecord,
    LearningRequest,
    RetrievalResult,
)


class TestActionStep:
    """Test cases for ActionStep model."""

    def test_action_step_creation(self):
        """Test creating a valid ActionStep."""
        step = ActionStep(
            thought="Click the login button",
            action="click",
            target_element_description="Login button on the homepage",
        )

        assert step.thought == "Click the login button"
        assert step.action == "click"
        assert step.target_element_description == "Login button on the homepage"

    def test_action_step_validation_missing_fields(self):
        """Test ActionStep validation with missing required fields."""
        with pytest.raises(ValidationError):
            ActionStep(
                thought="Click button",
                action="click",
                # Missing target_element_description
            )

    def test_action_step_empty_fields(self):
        """Test ActionStep with empty string fields."""
        step = ActionStep(thought="", action="", target_element_description="")

        assert step.thought == ""
        assert step.action == ""
        assert step.target_element_description == ""


class TestExperienceRecord:
    """Test cases for ExperienceRecord model."""

    def test_experience_record_creation(self, sample_experience_record):
        """Test creating a valid ExperienceRecord."""
        experience = sample_experience_record

        assert experience.task_description == "Log into Gmail using Chrome browser"
        assert len(experience.keywords) == 4
        assert len(experience.action_flow) == 3
        assert experience.is_successful is True
        assert experience.source_task_id == "test_task_001"
        assert experience.usage_count == 0
        assert isinstance(experience.last_used_at, datetime)

    def test_experience_record_defaults(self):
        """Test ExperienceRecord with default values."""
        experience = ExperienceRecord(
            task_description="Test task",
            action_flow=[],
            preconditions="None",
            is_successful=True,
            source_task_id="test_001",
        )

        assert experience.keywords == []
        assert experience.usage_count == 0
        assert isinstance(experience.last_used_at, datetime)

    def test_experience_record_validation(self):
        """Test ExperienceRecord validation."""
        with pytest.raises(ValidationError):
            ExperienceRecord(
                # Missing required fields
                task_description="Test"
            )

    def test_experience_record_action_flow_validation(self):
        """Test that action_flow items are validated as ActionStep objects."""
        with pytest.raises(ValidationError):
            ExperienceRecord(
                task_description="Test task",
                action_flow=["invalid_action_step"],  # Should be ActionStep objects
                preconditions="None",
                is_successful=True,
                source_task_id="test_001",
            )


class TestFactRecord:
    """Test cases for FactRecord model."""

    def test_fact_record_creation(self, sample_fact_record):
        """Test creating a valid FactRecord."""
        fact = sample_fact_record

        assert fact.content == "Chrome browser stores passwords in the password manager"
        assert len(fact.keywords) == 4
        assert fact.source == "documentation"
        assert fact.usage_count == 0
        assert isinstance(fact.last_used_at, datetime)

    def test_fact_record_defaults(self):
        """Test FactRecord with default values."""
        fact = FactRecord(content="Test fact content")

        assert fact.keywords == []
        assert fact.source == "manual"
        assert fact.usage_count == 0
        assert isinstance(fact.last_used_at, datetime)

    def test_fact_record_validation(self):
        """Test FactRecord validation."""
        with pytest.raises(ValidationError):
            FactRecord()  # Missing required content field


class TestRetrievalResult:
    """Test cases for RetrievalResult model."""

    def test_retrieval_result_creation(
        self, sample_experience_record, sample_fact_record
    ):
        """Test creating a valid RetrievalResult."""
        result = RetrievalResult(
            experiences=[sample_experience_record],
            facts=[sample_fact_record],
            query="test query",
            total_results=2,
        )

        assert len(result.experiences) == 1
        assert len(result.facts) == 1
        assert result.query == "test query"
        assert result.total_results == 2

    def test_retrieval_result_defaults(self):
        """Test RetrievalResult with default values."""
        result = RetrievalResult(query="test query", total_results=0)

        assert result.experiences == []
        assert result.facts == []
        assert result.query == "test query"
        assert result.total_results == 0


class TestLearningRequest:
    """Test cases for LearningRequest model."""

    def test_learning_request_creation(self, sample_raw_history):
        """Test creating a valid LearningRequest."""
        request = LearningRequest(
            raw_history=sample_raw_history,
            is_successful=True,
            source_task_id="test_task_001",
            app_name="Gmail",
            task_description="Login to Gmail",
        )

        assert len(request.raw_history) == 5
        assert request.is_successful is True
        assert request.source_task_id == "test_task_001"
        assert request.app_name == "Gmail"
        assert request.task_description == "Login to Gmail"

    def test_learning_request_defaults(self, sample_raw_history):
        """Test LearningRequest with default values."""
        request = LearningRequest(
            raw_history=sample_raw_history,
            is_successful=True,
            source_task_id="test_task_001",
        )

        assert request.app_name == ""
        assert request.task_description == ""

    def test_learning_request_validation(self):
        """Test LearningRequest validation."""
        with pytest.raises(ValidationError):
            LearningRequest(
                # Missing required fields
                raw_history=[]
            )
