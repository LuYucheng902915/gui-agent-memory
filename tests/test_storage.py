"""
Unit tests for the storage layer module.
"""

from unittest.mock import MagicMock, patch

import pytest

from gui_agent_memory.storage import MemoryStorage, StorageError


class TestMemoryStorage:
    """Test cases for MemoryStorage class."""

    @patch("gui_agent_memory.storage.chromadb")
    def test_storage_initialization(self, mock_chromadb, mock_chroma_collection):
        """Test storage initialization."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        assert storage.client is not None
        assert storage.experiential_collection is not None
        assert storage.declarative_collection is not None

    @patch("gui_agent_memory.storage.chromadb")
    def test_storage_initialization_failure(self, mock_chromadb):
        """Test storage initialization failure."""
        mock_chromadb.PersistentClient.side_effect = Exception("ChromaDB init failed")

        with pytest.raises(StorageError) as exc_info:
            MemoryStorage()

        assert "Failed to initialize ChromaDB client" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_experiences(
        self, mock_chromadb, mock_chroma_collection, sample_experience_record
    ):
        """Test adding experience records."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        result = storage.add_experiences([sample_experience_record], embeddings)

        assert len(result) == 1
        assert result[0] == sample_experience_record.source_task_id
        mock_chroma_collection.add.assert_called_once()

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_experiences_mismatch_length(
        self, mock_chromadb, mock_chroma_collection, sample_experience_record
    ):
        """Test adding experiences with mismatched lengths."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings for 1 experience

        with pytest.raises(StorageError) as exc_info:
            storage.add_experiences([sample_experience_record], embeddings)

        assert "Number of experiences must match number of embeddings" in str(
            exc_info.value
        )

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_facts(self, mock_chromadb, mock_chroma_collection, sample_fact_record):
        """Test adding fact records."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        result = storage.add_facts([sample_fact_record], embeddings)

        assert len(result) == 1
        assert "fact_" in result[0]
        mock_chroma_collection.add.assert_called_once()

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_experiences(self, mock_chromadb, mock_chroma_collection):
        """Test querying experience records."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock query response
        mock_chroma_collection.query.return_value = {
            "ids": [["exp_1", "exp_2"]],
            "documents": [["Doc 1", "Doc 2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        storage = MemoryStorage()
        result = storage.query_experiences(
            query_embeddings=[[0.1, 0.2, 0.3]], n_results=2
        )

        assert "ids" in result
        assert "documents" in result
        assert len(result["ids"][0]) == 2
        mock_chroma_collection.query.assert_called_once()

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_facts(self, mock_chromadb, mock_chroma_collection):
        """Test querying fact records."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock query response
        mock_chroma_collection.query.return_value = {
            "ids": [["fact_1"]],
            "documents": [["Fact content"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.1]],
        }

        storage = MemoryStorage()
        result = storage.query_facts(query_texts=["test query"], n_results=1)

        assert "ids" in result
        assert len(result["ids"][0]) == 1
        mock_chroma_collection.query.assert_called_once()

    @patch("gui_agent_memory.storage.chromadb")
    def test_experience_exists_true(self, mock_chromadb, mock_chroma_collection):
        """Test checking if experience exists - returns True."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock get response - experience exists
        mock_chroma_collection.get.return_value = {"ids": ["test_task_001"]}

        storage = MemoryStorage()
        result = storage.experience_exists("test_task_001")

        assert result is True
        mock_chroma_collection.get.assert_called_once_with(ids=["test_task_001"])

    @patch("gui_agent_memory.storage.chromadb")
    def test_experience_exists_false(self, mock_chromadb, mock_chroma_collection):
        """Test checking if experience exists - returns False."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock get response - experience doesn't exist
        mock_chroma_collection.get.return_value = {"ids": []}

        storage = MemoryStorage()
        result = storage.experience_exists("nonexistent_task")

        assert result is False

    @patch("gui_agent_memory.storage.chromadb")
    def test_get_collection_stats(self, mock_chromadb, mock_chroma_collection):
        """Test getting collection statistics."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock count responses
        mock_chroma_collection.count.side_effect = [
            5,
            3,
        ]  # experiential, then declarative

        storage = MemoryStorage()
        storage.experiential_collection = mock_chroma_collection
        storage.declarative_collection = mock_chroma_collection

        stats = storage.get_collection_stats()

        assert stats["experiential_memories"] == 5
        assert stats["declarative_memories"] == 3
        assert stats["total"] == 8

    @patch("gui_agent_memory.storage.chromadb")
    def test_clear_collections(self, mock_chromadb, mock_chroma_collection):
        """Test clearing all collections."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_client.delete_collection.return_value = None
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        storage.clear_collections()

        # Should delete both collections
        assert mock_client.delete_collection.call_count == 2

    @patch("gui_agent_memory.storage.chromadb")
    def test_get_collection_by_name(self, mock_chromadb, mock_chroma_collection):
        """Test getting collection by name."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Test valid collection names
        exp_collection = storage.get_collection("experiential_memories")
        dec_collection = storage.get_collection("declarative_memories")

        assert exp_collection is not None
        assert dec_collection is not None

        # Test invalid collection name
        with pytest.raises(StorageError) as exc_info:
            storage.get_collection("invalid_collection")

        assert "Unknown collection" in str(exc_info.value)


class TestMemoryStorageAdvanced:
    """Advanced test cases for MemoryStorage error handling and edge cases."""

    @patch("gui_agent_memory.storage.chromadb")
    def test_sqlite_compatibility_fix(self, mock_chromadb, mock_chroma_collection):
        """Test that SQLite compatibility fix is applied during initialization."""
        import sys

        # Store original state
        original_sqlite3 = sys.modules.get("sqlite3")

        try:
            # Remove sqlite3 from sys.modules to test the fix
            if "sqlite3" in sys.modules:
                del sys.modules["sqlite3"]

            mock_client = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_chroma_collection
            mock_chromadb.PersistentClient.return_value = mock_client

            # This should apply the SQLite fix and initialize successfully
            storage = MemoryStorage()

            assert storage.client is not None

        finally:
            # Restore original state
            if original_sqlite3:
                sys.modules["sqlite3"] = original_sqlite3

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_experiences_chromadb_error(
        self, mock_chromadb, mock_chroma_collection, sample_experience_record
    ):
        """Test adding experiences when ChromaDB operation fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock collection.add to raise an exception
        mock_chroma_collection.add.side_effect = Exception("ChromaDB add failed")

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        with pytest.raises(StorageError) as exc_info:
            storage.add_experiences([sample_experience_record], embeddings)

        assert "Failed to add experiences to ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_facts_chromadb_error(
        self, mock_chromadb, mock_chroma_collection, sample_fact_record
    ):
        """Test adding facts when ChromaDB operation fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock collection.add to raise an exception
        mock_chroma_collection.add.side_effect = Exception("ChromaDB add failed")

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        with pytest.raises(StorageError) as exc_info:
            storage.add_facts([sample_fact_record], embeddings)

        assert "Failed to add facts to ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_facts_mismatch_length(
        self, mock_chromadb, mock_chroma_collection, sample_fact_record
    ):
        """Test adding facts with mismatched embeddings length."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings for 1 fact

        with pytest.raises(StorageError) as exc_info:
            storage.add_facts([sample_fact_record], embeddings)

        assert "Number of facts must match number of embeddings" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_experiences_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test querying experiences when ChromaDB operation fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock collection.query to raise an exception
        mock_chroma_collection.query.side_effect = Exception("ChromaDB query failed")

        storage = MemoryStorage()

        with pytest.raises(StorageError) as exc_info:
            storage.query_experiences(query_embeddings=[[0.1, 0.2, 0.3]], n_results=2)

        assert "Failed to query experiences from ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_facts_chromadb_error(self, mock_chromadb, mock_chroma_collection):
        """Test querying facts when ChromaDB operation fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock collection.query to raise an exception
        mock_chroma_collection.query.side_effect = Exception("ChromaDB query failed")

        storage = MemoryStorage()

        with pytest.raises(StorageError) as exc_info:
            storage.query_facts(query_texts=["test"], n_results=1)

        assert "Failed to query facts from ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_experience_exists_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test experience_exists when ChromaDB operation fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock collection.get to raise an exception
        mock_chroma_collection.get.side_effect = Exception("ChromaDB get failed")

        storage = MemoryStorage()

        with pytest.raises(StorageError) as exc_info:
            storage.experience_exists("test_task")

        assert "Failed to check experience existence in ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_get_collection_stats_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test get_collection_stats when ChromaDB operation fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock collection.count to raise an exception
        mock_chroma_collection.count.side_effect = Exception("ChromaDB count failed")

        storage = MemoryStorage()

        with pytest.raises(StorageError) as exc_info:
            storage.get_collection_stats()

        assert "Failed to get collection statistics from ChromaDB" in str(
            exc_info.value
        )

    @patch("gui_agent_memory.storage.chromadb")
    def test_clear_collections_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test clear_collections when ChromaDB operation fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_client.delete_collection.side_effect = Exception("ChromaDB delete failed")
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        with pytest.raises(StorageError) as exc_info:
            storage.clear_collections()

        assert "Failed to clear collections in ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_client_initialization_with_custom_path(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test ChromaDB client initialization with custom path from config."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        with patch("gui_agent_memory.storage.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.chroma_db_path = "./test_db"  # Use a valid relative path
            mock_get_config.return_value = mock_config

            MemoryStorage()

            # Verify client was initialized with custom path (just check path argument)
            call_args = mock_chromadb.PersistentClient.call_args
            assert call_args.kwargs["path"] == "./test_db"

    @patch("gui_agent_memory.storage.chromadb")
    def test_collection_creation_with_embedding_function(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test that collections are created with proper embedding function."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        MemoryStorage()

        # Verify both collections were created with embedding function
        assert mock_client.get_or_create_collection.call_count == 2

        # Check that embedding function was passed to collection creation
        for call in mock_client.get_or_create_collection.call_args_list:
            args, kwargs = call
            assert "embedding_function" in kwargs

    @patch("gui_agent_memory.storage.chromadb")
    def test_update_usage_stats_success(self, mock_chromadb, mock_chroma_collection):
        """Test successful usage statistics update."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock the get method to return existing metadata
        mock_chroma_collection.get.return_value = {"metadatas": [{"usage_count": 1}]}
        # Mock successful update operations
        mock_chroma_collection.update.return_value = None

        storage = MemoryStorage()
        memory_ids = ["exp_1", "fact_1"]

        # This should not raise any exception
        storage.update_usage_stats(memory_ids, "experiential_memories")

        # Verify update was called for each memory
        assert mock_chroma_collection.update.call_count == 2

    @patch("gui_agent_memory.storage.chromadb")
    def test_update_usage_stats_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test update_usage_stats when ChromaDB operation fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock the get method to return existing metadata
        mock_chroma_collection.get.return_value = {"metadatas": [{"usage_count": 1}]}
        # Mock collection.update to raise an exception
        mock_chroma_collection.update.side_effect = Exception("ChromaDB update failed")

        storage = MemoryStorage()
        memory_ids = ["exp_1", "fact_1"]

        with pytest.raises(StorageError) as exc_info:
            storage.update_usage_stats(memory_ids, "experiential_memories")

        assert "Failed to update usage statistics in ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_experiences_with_embeddings_and_texts(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test querying experiences with both embeddings and texts."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock query response
        mock_chroma_collection.query.return_value = {
            "ids": [["exp_1"]],
            "documents": [["Doc 1"]],
            "metadatas": [[{"key": "value1"}]],
            "distances": [[0.1]],
        }

        storage = MemoryStorage()

        # Test with both query_embeddings and query_texts
        result = storage.query_experiences(
            query_embeddings=[[0.1, 0.2, 0.3]], query_texts=["test query"], n_results=1
        )

        assert "ids" in result
        mock_chroma_collection.query.assert_called_once()

        # Verify both embeddings and texts were passed
        call_args = mock_chroma_collection.query.call_args
        assert "query_embeddings" in call_args.kwargs
        assert "query_texts" in call_args.kwargs

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_facts_with_embeddings_and_texts(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test querying facts with both embeddings and texts."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock query response
        mock_chroma_collection.query.return_value = {
            "ids": [["fact_1"]],
            "documents": [["Fact content"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.1]],
        }

        storage = MemoryStorage()

        # Test with both query_embeddings and query_texts
        result = storage.query_facts(
            query_embeddings=[[0.1, 0.2, 0.3]], query_texts=["test query"], n_results=1
        )

        assert "ids" in result
        mock_chroma_collection.query.assert_called_once()

        # Verify both embeddings and texts were passed
        call_args = mock_chroma_collection.query.call_args
        assert "query_embeddings" in call_args.kwargs
        assert "query_texts" in call_args.kwargs

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_with_where_clause(self, mock_chromadb, mock_chroma_collection):
        """Test querying with where clause filtering."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # Mock query response
        mock_chroma_collection.query.return_value = {
            "ids": [["fact_1"]],
            "documents": [["Fact content"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.1]],
        }

        storage = MemoryStorage()

        # Test with where clause
        where_clause = {"source": {"$eq": "test"}}
        result = storage.query_facts(
            query_texts=["test query"], n_results=1, where=where_clause
        )

        assert "ids" in result
        mock_chroma_collection.query.assert_called_once()

        # Verify where clause was passed
        call_args = mock_chroma_collection.query.call_args
        assert "where" in call_args.kwargs
        assert call_args.kwargs["where"] == where_clause

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_experiences_empty_list(self, mock_chromadb, mock_chroma_collection):
        """Test adding empty list of experiences."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        result = storage.add_experiences([], [])

        assert result == []
        # Should not call ChromaDB add for empty list
        mock_chroma_collection.add.assert_not_called()

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_facts_empty_list(self, mock_chromadb, mock_chroma_collection):
        """Test adding empty list of facts."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        result = storage.add_facts([], [])

        assert result == []
        # Should not call ChromaDB add for empty list
        mock_chroma_collection.add.assert_not_called()
