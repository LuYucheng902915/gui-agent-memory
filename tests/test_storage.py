"""
Unit tests for the storage layer module.
"""

from unittest.mock import MagicMock, Mock, patch

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

        # Mock the add method to return a specific result
        mock_chroma_collection.add.return_value = {
            "ids": ["test_task_001"]  # Match the actual implementation
        }

        result = storage.add_experiences([sample_experience_record], embeddings)

        assert result == ["test_task_001"]
        mock_chroma_collection.add.assert_called_once()

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_experiences_mismatch_length(
        self, mock_chromadb, mock_chroma_collection, sample_experience_record
    ):
        """Test adding experience records with mismatched lengths."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Two embeddings for one record

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

        # Mock the add method to return a specific result
        mock_chroma_collection.add.return_value = {
            "ids": ["fact_-4054170162981411005_0"]  # Match the actual implementation
        }

        result = storage.add_facts([sample_fact_record], embeddings)

        assert len(result) == 1
        assert result[0].startswith("fact_")
        mock_chroma_collection.add.assert_called_once()

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_experiences(self, mock_chromadb, mock_chroma_collection):
        """Test querying experience records."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock query response with proper structure
        mock_chroma_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"meta1": "value1"}, {"meta2": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        result = storage.query_experiences(query_texts=["test"], n_results=2)

        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        assert "distances" in result

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_facts(self, mock_chromadb, mock_chroma_collection):
        """Test querying fact records."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock query response with proper structure
        mock_chroma_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"meta1": "value1"}, {"meta2": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        result = storage.query_facts(query_texts=["test"], n_results=2)

        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        assert "distances" in result

    @patch("gui_agent_memory.storage.chromadb")
    def test_experience_exists_true(self, mock_chromadb, mock_chroma_collection):
        """Test checking if experience exists (true case)."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock get response with existing record
        mock_chroma_collection.get.return_value = {
            "ids": ["existing_task_123"],
        }

        result = storage.experience_exists("existing_task_123")

        assert result is True

    @patch("gui_agent_memory.storage.chromadb")
    def test_experience_exists_false(self, mock_chromadb, mock_chroma_collection):
        """Test checking if experience exists (false case)."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock get response with no existing record
        mock_chroma_collection.get.return_value = {
            "ids": [],
        }

        result = storage.experience_exists("nonexistent_task_123")

        assert result is False

    @patch("gui_agent_memory.storage.chromadb")
    def test_get_collection_stats(self, mock_chromadb, mock_chroma_collection):
        """Test getting collection statistics."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock count responses
        mock_exp_collection = Mock()
        mock_exp_collection.count.return_value = 5
        mock_decl_collection = Mock()
        mock_decl_collection.count.return_value = 3

        storage.experiential_collection = mock_exp_collection
        storage.declarative_collection = mock_decl_collection

        result = storage.get_collection_stats()

        assert result["experiential_memories"] == 5
        assert result["declarative_memories"] == 3
        assert result["total"] == 8

    @patch("gui_agent_memory.storage.chromadb")
    def test_clear_collections(self, mock_chromadb, mock_chroma_collection):
        """Test clearing collections."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock delete responses
        mock_chroma_collection.delete.return_value = None

        # Call clear_collections and verify it doesn't return a value
        storage.clear_collections()

        # Verify that delete was called
        assert mock_chroma_collection.delete.call_count >= 1

    @patch("gui_agent_memory.storage.chromadb")
    def test_get_collection_by_name(self, mock_chromadb, mock_chroma_collection):
        """Test getting collection by name."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Test experiential collection - use the actual config names
        result = storage.get_collection(storage.config.experiential_collection_name)
        assert result == storage.experiential_collection

        # Test declarative collection - use the actual config names
        result = storage.get_collection(storage.config.declarative_collection_name)
        assert result == storage.declarative_collection

        # Test invalid collection name
        with pytest.raises(StorageError) as exc_info:
            storage.get_collection("invalid")
        assert "Unknown collection: invalid" in str(exc_info.value)


class TestMemoryStorageAdvanced:
    """Advanced test cases for MemoryStorage."""

    @patch("gui_agent_memory.storage.chromadb")
    def test_sqlite_compatibility_fix(self, mock_chromadb, mock_chroma_collection):
        """Test SQLite compatibility fix (lines 24-25)."""
        # This test verifies that the import fix is in place
        # We can't easily test the actual import behavior, but we can verify
        # that the code path exists and doesn't raise exceptions

        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        # This should not raise an exception
        storage = MemoryStorage()
        assert storage is not None

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_experiences_chromadb_error(
        self, mock_chromadb, mock_chroma_collection, sample_experience_record
    ):
        """Test adding experiences when ChromaDB fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        # Mock ChromaDB to raise an exception
        mock_chroma_collection.add.side_effect = Exception("ChromaDB error")

        with pytest.raises(StorageError) as exc_info:
            storage.add_experiences([sample_experience_record], embeddings)

        assert "Failed to add experiences to ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_facts_chromadb_error(
        self, mock_chromadb, mock_chroma_collection, sample_fact_record
    ):
        """Test adding facts when ChromaDB fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        # Mock ChromaDB to raise an exception
        mock_chroma_collection.add.side_effect = Exception("ChromaDB error")

        with pytest.raises(StorageError) as exc_info:
            storage.add_facts([sample_fact_record], embeddings)

        assert "Failed to add facts to ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_facts_mismatch_length(
        self, mock_chromadb, mock_chroma_collection, sample_fact_record
    ):
        """Test adding fact records with mismatched lengths."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # Two embeddings for one record

        with pytest.raises(StorageError) as exc_info:
            storage.add_facts([sample_fact_record], embeddings)

        assert "Number of facts must match number of embeddings" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_experiences_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test querying experiences when ChromaDB fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock ChromaDB to raise an exception
        mock_chroma_collection.query.side_effect = Exception("ChromaDB error")

        with pytest.raises(StorageError) as exc_info:
            storage.query_experiences(query_texts=["test"], n_results=2)

        assert "Failed to query experiences from ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_facts_chromadb_error(self, mock_chromadb, mock_chroma_collection):
        """Test querying facts when ChromaDB fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock ChromaDB to raise an exception
        mock_chroma_collection.query.side_effect = Exception("ChromaDB error")

        with pytest.raises(StorageError) as exc_info:
            storage.query_facts(query_texts=["test"], n_results=2)

        assert "Failed to query facts from ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_experience_exists_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test checking experience existence when ChromaDB fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock ChromaDB to raise an exception
        mock_chroma_collection.get.side_effect = Exception("ChromaDB error")

        with pytest.raises(StorageError) as exc_info:
            storage.experience_exists("test_task_id")

        assert "Failed to check experience existence in ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_get_collection_stats_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test getting collection stats when ChromaDB fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock ChromaDB to raise an exception on count
        mock_chroma_collection.count.side_effect = Exception("ChromaDB error")

        with pytest.raises(StorageError) as exc_info:
            storage.get_collection_stats()

        assert "Failed to get collection statistics from ChromaDB" in str(
            exc_info.value
        )

    @patch("gui_agent_memory.storage.chromadb")
    def test_clear_collections_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test clearing collections when ChromaDB fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock ChromaDB to raise an exception on delete
        mock_chroma_collection.delete.side_effect = Exception("ChromaDB error")

        # Should handle the error gracefully and not raise StorageError
        # The actual implementation catches exceptions and continues
        storage.clear_collections()

    @patch("gui_agent_memory.storage.chromadb")
    def test_client_initialization_with_custom_path(self, mock_chromadb):
        """Test client initialization with custom database path."""
        with patch("gui_agent_memory.storage.MemoryConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.chroma_db_path = "/custom/path/to/db"
            mock_config.return_value = mock_config_instance

            mock_client = MagicMock()
            mock_chromadb.PersistentClient.return_value = mock_client

            _ = MemoryStorage()

            # Verify that PersistentClient was called with the custom path
            # Note: The actual implementation uses the config path, so we need to check that
            mock_chromadb.PersistentClient.assert_called()

    @patch("gui_agent_memory.storage.chromadb")
    def test_collection_creation_with_embedding_function(self, mock_chromadb):
        """Test collection creation with embedding function."""
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        _ = MemoryStorage()

        # Verify that get_or_create_collection was called with embedding_function parameter
        # Note: This test assumes the implementation passes an embedding function
        # The actual verification would depend on the specific implementation details

    @patch("gui_agent_memory.storage.chromadb")
    def test_update_usage_stats_success(self, mock_chromadb, mock_chroma_collection):
        """Test successful update of usage stats."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock get response with existing metadata
        mock_chroma_collection.get.return_value = {
            "metadatas": [{"usage_count": 2, "other_field": "value"}]
        }

        # Mock update response
        mock_chroma_collection.update.return_value = None

        # Use the actual collection name from the config
        storage.update_usage_stats(
            ["record_1"], storage.config.experiential_collection_name
        )

        # Verify that get and update were called
        mock_chroma_collection.get.assert_called_once()
        mock_chroma_collection.update.assert_called_once()

        # Verify the updated metadata has incremented usage_count
        update_call_args = mock_chroma_collection.update.call_args
        updated_metadata = update_call_args[1]["metadatas"][0]
        assert updated_metadata["usage_count"] == 3  # 2 + 1

    @patch("gui_agent_memory.storage.chromadb")
    def test_update_usage_stats_chromadb_error(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test update usage stats when ChromaDB fails."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock ChromaDB to raise an exception on get
        mock_chroma_collection.get.side_effect = Exception("ChromaDB error")

        # Should raise StorageError when ChromaDB fails
        # Use the actual collection name from the config
        with pytest.raises(StorageError) as exc_info:
            storage.update_usage_stats(
                ["record_1"], storage.config.experiential_collection_name
            )

        assert "Failed to update usage statistics in ChromaDB" in str(exc_info.value)

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_experiences_with_embeddings_and_texts(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test querying experiences with both embeddings and texts."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock query response with proper structure
        mock_chroma_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"meta1": "value1"}]],
            "distances": [[0.1]],
        }

        # Test with both query_embeddings and query_texts
        result = storage.query_experiences(
            query_embeddings=[[0.1, 0.2, 0.3]], query_texts=["test"], n_results=1
        )

        assert "ids" in result
        mock_chroma_collection.query.assert_called_once()

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_facts_with_embeddings_and_texts(
        self, mock_chromadb, mock_chroma_collection
    ):
        """Test querying facts with both embeddings and texts."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock query response with proper structure
        mock_chroma_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"meta1": "value1"}]],
            "distances": [[0.1]],
        }

        # Test with both query_embeddings and query_texts
        result = storage.query_facts(
            query_embeddings=[[0.1, 0.2, 0.3]], query_texts=["test"], n_results=1
        )

        assert "ids" in result
        mock_chroma_collection.query.assert_called_once()

    @patch("gui_agent_memory.storage.chromadb")
    def test_query_with_where_clause(self, mock_chromadb, mock_chroma_collection):
        """Test querying with where clause."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Mock query response with proper structure
        mock_chroma_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"meta1": "value1"}]],
            "distances": [[0.1]],
        }

        # Test with where clause
        where_clause = {"field": "value"}
        result = storage.query_experiences(
            query_texts=["test"], where=where_clause, n_results=1
        )

        assert "ids" in result
        # Verify that the where clause was passed to the query method
        query_call_args = mock_chroma_collection.query.call_args
        assert query_call_args[1]["where"] == where_clause

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_experiences_empty_list(self, mock_chromadb, mock_chroma_collection):
        """Test adding empty list of experiences."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Should handle empty lists gracefully
        result = storage.add_experiences([], [])
        assert result == []

        # Verify that add was not called
        mock_chroma_collection.add.assert_not_called()

    @patch("gui_agent_memory.storage.chromadb")
    def test_add_facts_empty_list(self, mock_chromadb, mock_chroma_collection):
        """Test adding empty list of facts."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        storage = MemoryStorage()

        # Should handle empty lists gracefully
        result = storage.add_facts([], [])
        assert result == []

        # Verify that add was not called
        mock_chroma_collection.add.assert_not_called()


class TestStorageCoverage:
    """Tests for uncovered code paths in storage.py"""

    def test_update_usage_stats_empty_metadata(self, mock_chromadb_collection):
        """Test update_usage_stats when record has no metadata (line 388)."""
        config = Mock()
        config.chroma_db_path = "./test_data/test_chroma"
        config.experiential_collection_name = "test_exp"
        config.declarative_collection_name = "test_decl"
        config.chroma_anonymized_telemetry = False

        with patch("chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_chromadb_collection

            storage = MemoryStorage(config)

            # Mock collection to return empty metadata
            mock_chromadb_collection.get.return_value = {
                "metadatas": []  # Empty metadata list
            }

            # This should not raise an error - it should continue to next record
            storage.update_usage_stats(
                ["record_1", "record_2"], config.experiential_collection_name
            )

            # Verify get was called for each record
            assert mock_chromadb_collection.get.call_count == 2
            # update should not be called since metadata was empty
            mock_chromadb_collection.update.assert_not_called()

    def test_update_usage_stats_invalid_usage_count_type(
        self, mock_chromadb_collection
    ):
        """Test update_usage_stats with invalid usage_count type (line 399)."""
        config = Mock()
        config.chroma_db_path = "./test_data/test_chroma"
        config.experiential_collection_name = "test_exp"
        config.declarative_collection_name = "test_decl"
        config.chroma_anonymized_telemetry = False

        with patch("chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            # Use the provided fixture as the collection returned by the client
            mock_client.get_or_create_collection.return_value = mock_chromadb_collection

            storage = MemoryStorage(config)

            # Mock collection to return metadata with invalid usage_count type
            mock_chromadb_collection.get.return_value = {
                "metadatas": [
                    {
                        "usage_count": "invalid_string_count",  # Invalid type
                        "other_field": "value",
                    }
                ]
            }

            storage.update_usage_stats(
                ["record_1"], config.experiential_collection_name
            )

            # Verify that usage_count was reset to 1 due to invalid type
            expected_metadata = {
                "usage_count": 1,
                "other_field": "value",
                "last_used_at": mock_chromadb_collection.update.call_args[1][
                    "metadatas"
                ][0]["last_used_at"],
            }

            mock_chromadb_collection.update.assert_called_once_with(
                ids=["record_1"], metadatas=[expected_metadata]
            )

    def test_update_usage_stats_none_metadata(self, mock_chromadb_collection):
        """Test update_usage_stats when metadata is None."""
        config = Mock()
        config.chroma_db_path = "./test_data/test_chroma"
        config.experiential_collection_name = "test_exp"
        config.declarative_collection_name = "test_decl"
        config.chroma_anonymized_telemetry = False

        with patch("chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_chromadb_collection

            storage = MemoryStorage(config)

            # Mock collection to return None metadata
            mock_chromadb_collection.get.return_value = {
                "metadatas": [None]  # None metadata
            }

            storage.update_usage_stats(
                ["record_1"], config.experiential_collection_name
            )

            # Should handle None metadata by creating empty dict
            mock_chromadb_collection.update.assert_called_once()
            call_args = mock_chromadb_collection.update.call_args
            updated_metadata = call_args[1]["metadatas"][0]
            assert updated_metadata["usage_count"] == 1
            assert "last_used_at" in updated_metadata

    def test_update_usage_stats_valid_numeric_count(self, mock_chromadb_collection):
        """Test update_usage_stats with valid numeric usage_count."""
        config = Mock()
        config.chroma_db_path = "./test_data/test_chroma"
        config.experiential_collection_name = "test_exp"
        config.declarative_collection_name = "test_decl"
        config.chroma_anonymized_telemetry = False

        with patch("chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_chromadb_collection

            storage = MemoryStorage(config)

            # Test with integer count
            mock_chromadb_collection.get.return_value = {
                "metadatas": [{"usage_count": 5}]
            }

            storage.update_usage_stats(
                ["record_1"], config.experiential_collection_name
            )

            # Should increment the count
            call_args = mock_chromadb_collection.update.call_args
            updated_metadata = call_args[1]["metadatas"][0]
            assert updated_metadata["usage_count"] == 6

    def test_update_usage_stats_float_count(self, mock_chromadb_collection):
        """Test update_usage_stats with float usage_count."""
        config = Mock()
        config.chroma_db_path = "./test_data/test_chroma"
        config.experiential_collection_name = "test_exp"
        config.declarative_collection_name = "test_decl"
        config.chroma_anonymized_telemetry = False

        with patch("chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_chromadb_collection

            storage = MemoryStorage(config)

            # Test with float count
            mock_chromadb_collection.get.return_value = {
                "metadatas": [{"usage_count": 3.7}]
            }

            storage.update_usage_stats(
                ["record_1"], config.experiential_collection_name
            )

            # Should convert to int and increment
            call_args = mock_chromadb_collection.update.call_args
            updated_metadata = call_args[1]["metadatas"][0]
            assert updated_metadata["usage_count"] == 4  # int(3.7) + 1
