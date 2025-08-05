"""
Main API interface for the Enhanced Memory Module.

This module provides the high-level API for interacting with the memory system,
including retrieval, learning, and knowledge management operations.
"""

from typing import Any, Dict, List

from .config import ConfigurationError, get_config
from .ingestion import IngestionError, MemoryIngestion
from .models import ExperienceRecord, FactRecord, RetrievalResult
from .retriever import MemoryRetriever, RetrievalError
from .storage import MemoryStorage, StorageError


class MemorySystemError(Exception):
    """General exception for memory system operations."""

    pass


class MemorySystem:
    """
    Main interface for the Enhanced Memory Module.
    
    Provides high-level APIs for:
    - Retrieving relevant memories based on queries
    - Learning from task execution history
    - Adding structured experiences and facts
    - Managing the memory system configuration
    """

    def __init__(self) -> None:
        """
        Initialize the memory system.
        
        Raises:
            ConfigurationError: If system configuration is invalid
            MemorySystemError: If initialization fails
        """
        try:
            # Initialize configuration (includes validation)
            self.config = get_config()
            
            # Initialize core components
            self.storage = MemoryStorage()
            self.ingestion = MemoryIngestion()
            self.retriever = MemoryRetriever()
            
        except ConfigurationError:
            raise  # Re-raise configuration errors directly
        except Exception as e:
            raise MemorySystemError(f"Failed to initialize memory system: {e}")

    def retrieve_memories(self, query: str, top_n: int = 3) -> RetrievalResult:
        """
        Retrieve relevant memories based on a query.
        
        This is the primary retrieval interface that uses hybrid search
        (vector + keyword) with unified re-ranking to find the most relevant
        experiences and facts for the given query.
        
        Args:
            query: A grammatically complete sentence describing the current intent
            top_n: Number of results to return for each memory type (default: 3)
            
        Returns:
            RetrievalResult containing relevant experiences and facts
            
        Raises:
            MemorySystemError: If retrieval operation fails
            
        Example:
            >>> memory_system = MemorySystem()
            >>> result = memory_system.retrieve_memories(
            ...     "How to log into Gmail using Chrome browser"
            ... )
            >>> print(f"Found {len(result.experiences)} experiences and {len(result.facts)} facts")
        """
        if not query or not query.strip():
            raise MemorySystemError("Query cannot be empty")
        
        try:
            return self.retriever.retrieve_memories(query, top_n)
        except RetrievalError as e:
            raise MemorySystemError(f"Memory retrieval failed: {e}")

    def learn_from_task(
        self,
        raw_history: List[Dict[str, Any]],
        is_successful: bool,
        source_task_id: str,
        app_name: str = "",
        task_description: str = ""
    ) -> str:
        """
        Learn from task execution history (V1.0 temporary interface).
        
        This method processes raw task execution history and converts it into
        structured experience records using LLM-based distillation.
        
        Args:
            raw_history: Raw operational history from task execution
            is_successful: Whether the task was completed successfully
            source_task_id: Unique identifier for the source task
            app_name: Name of the application being operated on (optional)
            task_description: Optional description of the task (optional)
            
        Returns:
            Success message with details about the learned experience
            
        Raises:
            MemorySystemError: If learning process fails
            
        Example:
            >>> memory_system = MemorySystem()
            >>> history = [
            ...     {"action": "click", "target": "login button", "thought": "need to login"},
            ...     {"action": "type", "target": "email field", "text": "user@example.com"}
            ... ]
            >>> result = memory_system.learn_from_task(
            ...     raw_history=history,
            ...     is_successful=True,
            ...     source_task_id="task_123",
            ...     app_name="Gmail"
            ... )
            >>> print(result)
        """
        if not raw_history:
            raise MemorySystemError("Raw history cannot be empty")
        
        if not source_task_id or not source_task_id.strip():
            raise MemorySystemError("Source task ID cannot be empty")
        
        try:
            return self.ingestion.learn_from_task(
                raw_history=raw_history,
                is_successful=is_successful,
                source_task_id=source_task_id,
                app_name=app_name,
                task_description=task_description
            )
        except IngestionError as e:
            raise MemorySystemError(f"Learning from task failed: {e}")

    def add_experience(self, experience: ExperienceRecord) -> str:
        """
        Add a pre-structured experience record (future-facing interface).
        
        This method allows adding complete experience records that have been
        pre-processed and structured, bypassing the LLM distillation step.
        
        Args:
            experience: Complete experience record to add
            
        Returns:
            Success message with record ID
            
        Raises:
            MemorySystemError: If adding experience fails
            
        Example:
            >>> from memory_system.models import ExperienceRecord, ActionStep
            >>> experience = ExperienceRecord(
            ...     task_description="Log into Gmail",
            ...     keywords=["gmail", "login", "chrome"],
            ...     action_flow=[
            ...         ActionStep(
            ...             thought="Need to click login button",
            ...             action="click",
            ...             target_element_description="Gmail login button"
            ...         )
            ...     ],
            ...     preconditions="Chrome browser is open",
            ...     is_successful=True,
            ...     source_task_id="manual_001"
            ... )
            >>> result = memory_system.add_experience(experience)
        """
        if not isinstance(experience, ExperienceRecord):
            raise MemorySystemError("Experience must be an ExperienceRecord instance")
        
        try:
            return self.ingestion.add_experience(experience)
        except IngestionError as e:
            raise MemorySystemError(f"Adding experience failed: {e}")

    def add_fact(self, content: str, keywords: List[str], source: str = "manual") -> str:
        """
        Add a semantic fact to the knowledge base.
        
        Args:
            content: The factual content to add
            keywords: List of keywords for retrieval
            source: Source of this knowledge (default: "manual")
            
        Returns:
            Success message with record ID
            
        Raises:
            MemorySystemError: If adding fact fails
            
        Example:
            >>> memory_system = MemorySystem()
            >>> result = memory_system.add_fact(
            ...     content="Chrome browser stores passwords in the password manager",
            ...     keywords=["chrome", "password", "security", "browser"],
            ...     source="documentation"
            ... )
            >>> print(result)
        """
        if not content or not content.strip():
            raise MemorySystemError("Fact content cannot be empty")
        
        if not keywords:
            keywords = []
        
        try:
            return self.ingestion.add_fact(content, keywords, source)
        except IngestionError as e:
            raise MemorySystemError(f"Adding fact failed: {e}")

    def batch_add_facts(self, facts_data: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple facts in batch for efficient bulk operations.
        
        Args:
            facts_data: List of dictionaries containing fact data.
                       Each dict should have: content, keywords (optional), source (optional)
            
        Returns:
            List of success messages for each added fact
            
        Raises:
            MemorySystemError: If batch operation fails
            
        Example:
            >>> facts = [
            ...     {
            ...         "content": "Gmail supports 2FA authentication",
            ...         "keywords": ["gmail", "2fa", "security"],
            ...         "source": "documentation"
            ...     },
            ...     {
            ...         "content": "Chrome can sync bookmarks across devices",
            ...         "keywords": ["chrome", "sync", "bookmarks"]
            ...     }
            ... ]
            >>> results = memory_system.batch_add_facts(facts)
        """
        if not facts_data:
            raise MemorySystemError("Facts data cannot be empty")
        
        # Validate each fact data
        for i, fact_data in enumerate(facts_data):
            if not isinstance(fact_data, dict):
                raise MemorySystemError(f"Fact data at index {i} must be a dictionary")
            if "content" not in fact_data or not fact_data["content"].strip():
                raise MemorySystemError(f"Fact at index {i} must have non-empty content")
        
        try:
            return self.ingestion.batch_add_facts(facts_data)
        except IngestionError as e:
            raise MemorySystemError(f"Batch adding facts failed: {e}")

    def get_similar_experiences(self, task_description: str, top_n: int = 5) -> List[ExperienceRecord]:
        """
        Get experiences similar to a given task description.
        
        Args:
            task_description: Description of the task to find similar experiences for
            top_n: Number of similar experiences to return
            
        Returns:
            List of similar experiences
            
        Raises:
            MemorySystemError: If retrieval fails
        """
        if not task_description or not task_description.strip():
            raise MemorySystemError("Task description cannot be empty")
        
        try:
            return self.retriever.get_similar_experiences(task_description, top_n)
        except RetrievalError as e:
            raise MemorySystemError(f"Getting similar experiences failed: {e}")

    def get_related_facts(self, topic: str, top_n: int = 5) -> List[FactRecord]:
        """
        Get facts related to a specific topic.
        
        Args:
            topic: Topic to search for related facts
            top_n: Number of related facts to return
            
        Returns:
            List of related facts
            
        Raises:
            MemorySystemError: If retrieval fails
        """
        if not topic or not topic.strip():
            raise MemorySystemError("Topic cannot be empty")
        
        try:
            return self.retriever.get_related_facts(topic, top_n)
        except RetrievalError as e:
            raise MemorySystemError(f"Getting related facts failed: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary containing system statistics
            
        Raises:
            MemorySystemError: If stats retrieval fails
        """
        try:
            storage_stats = self.storage.get_collection_stats()
            
            return {
                "storage": storage_stats,
                "configuration": {
                    "embedding_model": self.config.embedding_model,
                    "reranker_model": self.config.reranker_model,
                    "experience_llm_model": self.config.experience_llm_model,
                    "chroma_db_path": self.config.chroma_db_path
                },
                "version": "1.0.0"
            }
        except StorageError as e:
            raise MemorySystemError(f"Getting system stats failed: {e}")

    def validate_system(self) -> bool:
        """
        Validate that the memory system is properly configured and operational.
        
        Returns:
            True if system is valid and operational
            
        Raises:
            MemorySystemError: If validation fails
        """
        try:
            # Validate configuration
            self.config.validate_configuration()
            
            # Test basic storage operations
            stats = self.storage.get_collection_stats()
            
            return True
            
        except (ConfigurationError, StorageError) as e:
            raise MemorySystemError(f"System validation failed: {e}")

    def clear_all_memories(self) -> str:
        """
        Clear all stored memories (for testing/reset purposes).
        
        WARNING: This operation is irreversible and will delete all stored experiences and facts.
        
        Returns:
            Success message
            
        Raises:
            MemorySystemError: If clearing operation fails
        """
        try:
            self.storage.clear_collections()
            return "Successfully cleared all memories from the system"
        except StorageError as e:
            raise MemorySystemError(f"Clearing memories failed: {e}")


# Convenience function for quick initialization
def create_memory_system() -> MemorySystem:
    """
    Create and initialize a MemorySystem instance.
    
    Returns:
        Initialized MemorySystem instance
        
    Raises:
        MemorySystemError: If initialization fails
    """
    return MemorySystem()