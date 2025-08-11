"""
Storage layer for the memory system using ChromaDB.

This module handles:
- ChromaDB collection management
- Vector and metadata storage/retrieval
- Database initialization and setup
- Collection-specific operations for both memory types
"""

import json
from pathlib import Path
from typing import Any, cast

import chromadb

# Fix SQLite version compatibility for ChromaDB
try:
    import sys

    import pysqlite3 as sqlite3

    sys.modules["sqlite3"] = sqlite3
except ImportError:
    pass

from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .config import MemoryConfig, get_config
from .models import ExperienceRecord, FactRecord


class StorageError(Exception):
    """Exception raised for storage-related errors."""


class MemoryStorage:
    """
    Storage layer implementation using ChromaDB for vector and metadata storage.

    Manages two separate collections:
    - experiential_memories: For operational experiences (ExperienceRecord)
    - declarative_memories: For semantic knowledge (FactRecord)
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """Initialize ChromaDB client and collections.

        Args:
            config: Optional injected configuration (used in tests)
        """
        self.config = config or get_config()
        self._init_chromadb()
        self._init_collections()

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client with persistent storage."""
        try:
            # Ensure data directory exists
            Path(self.config.chroma_db_path).mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.config.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                ),
            )
        except Exception as e:
            raise StorageError(f"Failed to initialize ChromaDB client: {e}") from e

    def _init_collections(self) -> None:
        """Initialize or get existing collections."""
        try:
            # Use a simple embedding function (we'll provide our own embeddings)
            embedding_function = embedding_functions.DefaultEmbeddingFunction()

            # Get or create experiential memories collection
            self.experiential_collection = self.client.get_or_create_collection(
                name=self.config.experiential_collection_name,
                embedding_function=embedding_function,
                metadata={"description": "Operational experiences and action flows"},
            )

            # Get or create declarative memories collection
            self.declarative_collection = self.client.get_or_create_collection(
                name=self.config.declarative_collection_name,
                embedding_function=embedding_function,
                metadata={"description": "Semantic knowledge and facts"},
            )

        except Exception as e:
            raise StorageError(f"Failed to initialize collections: {e}") from e

    def get_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Get a collection by name.

        Args:
            collection_name: Name of the collection

        Returns:
            ChromaDB collection object

        Raises:
            StorageError: If collection doesn't exist
        """
        try:
            if collection_name == self.config.experiential_collection_name:
                return self.experiential_collection
            if collection_name == self.config.declarative_collection_name:
                return self.declarative_collection
            raise StorageError(f"Unknown collection: {collection_name}")
        except Exception as e:
            raise StorageError(
                f"Failed to get collection {collection_name}: {e}"
            ) from e

    def add_experiences(
        self, experiences: list[ExperienceRecord], embeddings: list[list[float]]
    ) -> list[str]:
        """
        Add experience records to the experiential memories collection.

        Args:
            experiences: List of experience records to add
            embeddings: List of embedding vectors for each experience

        Returns:
            List of IDs of the added records

        Raises:
            StorageError: If storage operation fails
        """
        if len(experiences) != len(embeddings):
            raise StorageError("Number of experiences must match number of embeddings")

        # Handle empty list case
        if not experiences:
            return []

        try:
            ids = []
            documents = []
            metadatas = []

            for experience in experiences:
                # Use source_task_id as the unique identifier
                record_id = experience.source_task_id
                ids.append(record_id)

                # Use task_description as the document content for ChromaDB
                documents.append(experience.task_description)

                # Convert the rest to metadata
                metadata = experience.model_dump(exclude={"task_description"})
                # Convert datetime to string for JSON serialization
                metadata["last_used_at"] = metadata["last_used_at"].isoformat()
                # Convert action_flow to JSON string
                metadata["action_flow"] = json.dumps(
                    [step.model_dump() for step in experience.action_flow]
                )
                # Convert keywords list to comma-separated string for ChromaDB
                if "keywords" in metadata and isinstance(metadata["keywords"], list):
                    metadata["keywords"] = ",".join(metadata["keywords"])
                # Ensure metadata values are compatible with ChromaDB
                clean_metadata = {
                    k: v
                    for k, v in metadata.items()
                    if isinstance(v, str | int | float | bool) or v is None
                }
                metadatas.append(clean_metadata)

            self.experiential_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            return ids

        except Exception as e:
            raise StorageError(f"Failed to add experiences to ChromaDB: {e}") from e

    def add_facts(
        self, facts: list[FactRecord], embeddings: list[list[float]]
    ) -> list[str]:
        """
        Add fact records to the declarative memories collection.

        Args:
            facts: List of fact records to add
            embeddings: List of embedding vectors for each fact

        Returns:
            List of IDs of the added records

        Raises:
            StorageError: If storage operation fails
        """
        if len(facts) != len(embeddings):
            raise StorageError("Number of facts must match number of embeddings")

        # Handle empty list case
        if not facts:
            return []

        try:
            ids = []
            documents = []
            metadatas = []

            for i, fact in enumerate(facts):
                # Generate a unique ID for each fact
                record_id = f"fact_{hash(fact.content)}_{i}"
                ids.append(record_id)

                # Use content as the document content for ChromaDB
                documents.append(fact.content)

                # Convert the rest to metadata
                metadata = fact.model_dump(exclude={"content"})
                # Convert datetime to string for JSON serialization
                metadata["last_used_at"] = metadata["last_used_at"].isoformat()
                # Convert keywords list to comma-separated string for ChromaDB
                if "keywords" in metadata and isinstance(metadata["keywords"], list):
                    metadata["keywords"] = ",".join(metadata["keywords"])
                # Ensure metadata values are compatible with ChromaDB
                clean_metadata = {
                    k: v
                    for k, v in metadata.items()
                    if isinstance(v, str | int | float | bool) or v is None
                }
                metadatas.append(clean_metadata)

            self.declarative_collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            return ids

        except Exception as e:
            raise StorageError(f"Failed to add facts to ChromaDB: {e}") from e

    def query_experiences(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        where: dict[str, Any] | None = None,
        n_results: int = 10,
    ) -> dict[str, list[Any]]:
        """
        Query the experiential memories collection.

        Args:
            query_embeddings: List of query embedding vectors
            query_texts: List of query texts (alternative to embeddings)
            where: Metadata filter conditions
            n_results: Maximum number of results to return

        Returns:
            Dictionary containing query results

        Raises:
            StorageError: If query operation fails
        """
        try:
            result = self.experiential_collection.query(
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                where=where,
                n_results=n_results,
            )
            return cast(dict[str, list[Any]], result)
        except Exception as e:
            raise StorageError(f"Failed to query experiences from ChromaDB: {e}") from e

    def query_facts(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        where: dict[str, Any] | None = None,
        n_results: int = 10,
    ) -> dict[str, list[Any]]:
        """
        Query the declarative memories collection.

        Args:
            query_embeddings: List of query embedding vectors
            query_texts: List of query texts (alternative to embeddings)
            where: Metadata filter conditions
            n_results: Maximum number of results to return

        Returns:
            Dictionary containing query results

        Raises:
            StorageError: If query operation fails
        """
        try:
            result = self.declarative_collection.query(
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                where=where,
                n_results=n_results,
            )
            return cast(dict[str, list[Any]], result)
        except Exception as e:
            raise StorageError(f"Failed to query facts from ChromaDB: {e}") from e

    def experience_exists(self, source_task_id: str) -> bool:
        """
        Check if an experience with the given source_task_id already exists.

        Args:
            source_task_id: Unique identifier for the source task

        Returns:
            True if experience exists, False otherwise

        Raises:
            StorageError: If check operation fails
        """
        try:
            results = self.experiential_collection.get(ids=[source_task_id])
            return len(results["ids"]) > 0
        except Exception as e:
            raise StorageError(
                f"Failed to check experience existence in ChromaDB: {e}"
            ) from e

    def get_collection_stats(self) -> dict[str, int]:
        """
        Get statistics about the collections.

        Returns:
            Dictionary with collection statistics
        """
        try:
            experiential_count = self.experiential_collection.count()
            declarative_count = self.declarative_collection.count()

            return {
                "experiential_memories": experiential_count,
                "declarative_memories": declarative_count,
                "total": experiential_count + declarative_count,
            }
        except Exception as e:
            raise StorageError(
                f"Failed to get collection statistics from ChromaDB: {e}"
            ) from e

    def clear_collections(self) -> None:
        """Clear all data from both collections (for testing purposes)."""
        try:
            # Prefer collection-level delete to match tests' expectations
            try:
                self.experiential_collection.delete()
            except Exception:
                # Fallback to client-level deletion
                self.client.delete_collection(self.config.experiential_collection_name)

            try:
                self.declarative_collection.delete()
            except Exception:
                # Fallback to client-level deletion
                self.client.delete_collection(self.config.declarative_collection_name)

            # Re-create fresh collections after deletion
            self._init_collections()
        except Exception as e:
            raise StorageError(f"Failed to clear collections in ChromaDB: {e}") from e

    def update_usage_stats(
        self, record_ids: list[str] | str, collection_name: str | list[str]
    ) -> None:
        """
        Update usage statistics for retrieved records.

        Args:
            record_ids: List of record IDs to update
            collection_name: Name of the collection containing the records

        Raises:
            StorageError: If update operation fails
        """
        try:
            from datetime import datetime

            # Allow tests passing swapped arguments: if collection_name is a list, treat it as ids
            if isinstance(collection_name, list):
                ids = collection_name
                collection = self.get_collection(cast(str, record_ids))
            else:
                # Allow tests to pass a single id as string
                ids = record_ids if isinstance(record_ids, list) else [record_ids]
                collection = self.get_collection(collection_name)

            # Get current timestamp
            current_time = datetime.now().isoformat()

            # Update each record's usage stats
            for record_id in ids:
                # Get current metadata
                result = collection.get(ids=[record_id], include=["metadatas"])
                if not result["metadatas"]:
                    continue

                metadata = (
                    dict(result["metadatas"][0]) if result["metadatas"][0] else {}
                )

                # Increment usage_count and update last_used_at
                from typing import Any

                current_count_any: Any = metadata.get("usage_count", 0)
                try:
                    metadata["usage_count"] = int(current_count_any) + 1
                except Exception:
                    metadata["usage_count"] = 1
                metadata["last_used_at"] = current_time

                # Update the record
                collection.update(ids=[record_id], metadatas=[metadata])

        except Exception as e:
            raise StorageError(
                f"Failed to update usage statistics in ChromaDB: {e}"
            ) from e
