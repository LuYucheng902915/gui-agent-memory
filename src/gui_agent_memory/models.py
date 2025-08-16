"""
Data models for the memory system using Pydantic for validation and serialization.

This module defines the core data structures for both types of memories:
- ExperienceRecord: For storing operational experiences (episodic memory)
- FactRecord: For storing semantic knowledge (declarative memory)
"""

from datetime import datetime
from typing import Any, Literal

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
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for hybrid retrieval, prioritizing app names and features",
    )
    action_flow: list[ActionStep] = Field(
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
    keywords: list[str] = Field(
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

    experiences: list[ExperienceRecord] = Field(
        default_factory=list, description="Retrieved operational experiences"
    )
    facts: list[FactRecord] = Field(
        default_factory=list, description="Retrieved semantic facts"
    )
    query: str = Field(description="The original query used for retrieval")
    total_results: int = Field(description="Total number of results found")


class LearningRequest(BaseModel):
    """
    Model for requests to learn from task execution history.
    """

    raw_history: list[dict[str, Any]] = Field(
        description="Raw operational history from task execution"
    )
    is_successful: bool = Field(
        description="Whether the task was completed successfully"
    )
    source_task_id: str = Field(description="Unique identifier for the source task")
    app_name: str = Field(
        default="", description="Name of the application being operated on"
    )
    task_description: str = Field(
        default="", description="Optional description of the task being learned"
    )


class UpsertResult(BaseModel):
    """
    Fixed-shape result model for upsert operations.

    - result: the decision taken by the ingestion policy
    - the rest fields are optional but always present in schema for stability
    - details: bag for rare/diagnostic fields without breaking the schema
    """

    result: Literal[
        "added_new",
        "discarded_by_fingerprint",
        "updated_existing",
        "kept_new_deleted_old",
        "kept_old_discarded_new",
    ]
    new_record_id: str | None = None
    top_id: str | None = None
    similarity: float | None = None
    threshold: float | None = None
    invoked_judge: bool | None = None
    judge_decision: str | None = None
    judge_log_dir: str | None = None
    fingerprint_discarded: bool | None = None
    # Where the similarity/decision signal comes from: fingerprint or vector
    similarity_origin: Literal["fingerprint", "vector"] | None = None
    # Hits (None = phase not executed)
    pre_dedupe_hit: bool | None = None
    db_hit: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "extra": "forbid",
    }


class AddFactResponse(BaseModel):
    """
    Public API response for adding a fact.

    Stable schema for external callers/UI with a success indicator and
    normalized fields mapped from UpsertResult.
    """

    success: bool
    result: Literal[
        "added_new",
        "discarded_by_fingerprint",
        "updated_existing",
        "kept_new_deleted_old",
        "kept_old_discarded_new",
    ]
    record_id: str | None = None
    message: str
    # Intentionally no fingerprint flags in external response (debug-only internally)


class StoredFact(BaseModel):
    """
    Canonical shape of a fact record as it is persisted in the vector store.

    Returned by low-level storage APIs after a successful write so that callers
    can log exactly what was stored (id, document and sanitized metadata).
    """

    record_id: str
    document: str
    metadata: dict[str, Any]
    embedding: list[float] = Field(
        default_factory=list, description="Embedding vector stored alongside the record"
    )
