"""
Data models for the memory system using Pydantic for validation and serialization.

This module defines the core data structures for both types of memories:
- ExperienceRecord: For storing operational experiences (episodic memory)
- FactRecord: For storing semantic knowledge (declarative memory)
"""

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ActionStep(BaseModel):
    """
    Defines the structure of a single action step within an operational experience.
    Serves as a basic unit of an "experience" containing the thought process and action.
    """

    thought: str = Field(description="The thought process when executing this step")
    action: str = Field(description="The type of action (e.g., click, type, scroll)")
    target_element_description: str = Field(
        description="Description of the action's target element"
    )


class ExperienceRecord(BaseModel):
    """
    Model for storing operational experiences (episodic memory).
    
    This represents a complete operational flow that the agent has learned,
    including both successful and failed experiences.
    """

    task_description: str = Field(
        description="Description of the task this experience relates to"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for hybrid retrieval, prioritizing app names and features",
    )
    action_flow: List[ActionStep] = Field(
        description="Complete, ordered list of action steps to complete the task"
    )
    preconditions: str = Field(
        description="Required conditions or context for this experience to be applicable"
    )
    is_successful: bool = Field(
        description="Whether this experience represents a successful task completion"
    )
    
    # Metadata fields (reserved for future use in V1.0)
    usage_count: int = Field(
        default=0, description="Number of times this experience has been retrieved"
    )
    last_used_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of when this experience was last used",
    )
    source_task_id: str = Field(
        description="Unique identifier of the original task that generated this experience"
    )


class FactRecord(BaseModel):
    """
    Model for storing semantic knowledge (declarative memory).
    
    This represents objective facts and rules about operating systems,
    applications, or general domains.
    """

    content: str = Field(
        description="The core content of the fact or knowledge, used for vector retrieval"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for hybrid retrieval, focusing on domain and application terms",
    )
    source: str = Field(
        default="manual",
        description="Source of this knowledge (e.g., 'manual', 'documentation', 'inference')",
    )
    
    # Metadata fields (reserved for future use in V1.0)
    usage_count: int = Field(
        default=0, description="Number of times this fact has been retrieved"
    )
    last_used_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of when this fact was last used",
    )


class RetrievalResult(BaseModel):
    """
    Model for structured retrieval results returned by the memory system.
    """

    experiences: List[ExperienceRecord] = Field(
        default_factory=list, description="Retrieved operational experiences"
    )
    facts: List[FactRecord] = Field(
        default_factory=list, description="Retrieved semantic facts"
    )
    query: str = Field(description="The original query used for retrieval")
    total_results: int = Field(description="Total number of results found")


class LearningRequest(BaseModel):
    """
    Model for requests to learn from task execution history.
    """

    raw_history: List[Dict[str, Any]] = Field(
        description="Raw operational history from task execution"
    )
    is_successful: bool = Field(
        description="Whether the task was completed successfully"
    )
    source_task_id: str = Field(
        description="Unique identifier for the source task"
    )
    app_name: str = Field(
        default="", description="Name of the application being operated on"
    )
    task_description: str = Field(
        default="", description="Optional description of the task being learned"
    )