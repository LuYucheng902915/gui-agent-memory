"""
Unit tests for the storage layer module.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from gui_agent_memory.storage import MemoryStorage, StorageError
from gui_agent_memory.models import ActionStep, ExperienceRecord, FactRecord


class TestMemoryStorage:
    """Test cases for MemoryStorage class."""

    @patch('gui_agent_memory.storage.chromadb')
    def test_storage_initialization(self, mock_chromadb, mock_chroma_collection):
        """Test storage initialization."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        storage = MemoryStorage()
        
        assert storage.client is not None
        assert storage.experiential_collection is not None
        assert storage.declarative_collection is not None

    @patch('gui_agent_memory.storage.chromadb')
    def test_storage_initialization_failure(self, mock_chromadb):
        """Test storage initialization failure."""
        mock_chromadb.PersistentClient.side_effect = Exception("ChromaDB init failed")
        
        with pytest.raises(StorageError) as exc_info:
            MemoryStorage()
        
        assert "Failed to initialize ChromaDB client" in str(exc_info.value)

    @patch('gui_agent_memory.storage.chromadb')
    def test_add_experiences(self, mock_chromadb, mock_chroma_collection, sample_experience_record):
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

    @patch('gui_agent_memory.storage.chromadb')
    def test_add_experiences_mismatch_length(self, mock_chromadb, mock_chroma_collection, sample_experience_record):
        """Test adding experiences with mismatched lengths."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        storage = MemoryStorage()
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings for 1 experience
        
        with pytest.raises(StorageError) as exc_info:
            storage.add_experiences([sample_experience_record], embeddings)
        
        assert "Number of experiences must match number of embeddings" in str(exc_info.value)

    @patch('gui_agent_memory.storage.chromadb')
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

    @patch('gui_agent_memory.storage.chromadb')
    def test_query_experiences(self, mock_chromadb, mock_chroma_collection):
        """Test querying experience records."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock query response
        mock_chroma_collection.query.return_value = {
            'ids': [['exp_1', 'exp_2']],
            'documents': [['Doc 1', 'Doc 2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        storage = MemoryStorage()
        result = storage.query_experiences(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=2
        )
        
        assert 'ids' in result
        assert 'documents' in result
        assert len(result['ids'][0]) == 2
        mock_chroma_collection.query.assert_called_once()

    @patch('gui_agent_memory.storage.chromadb')
    def test_query_facts(self, mock_chromadb, mock_chroma_collection):
        """Test querying fact records."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock query response
        mock_chroma_collection.query.return_value = {
            'ids': [['fact_1']],
            'documents': [['Fact content']],
            'metadatas': [[{'source': 'test'}]],
            'distances': [[0.1]]
        }
        
        storage = MemoryStorage()
        result = storage.query_facts(
            query_texts=["test query"],
            n_results=1
        )
        
        assert 'ids' in result
        assert len(result['ids'][0]) == 1
        mock_chroma_collection.query.assert_called_once()

    @patch('gui_agent_memory.storage.chromadb')
    def test_experience_exists_true(self, mock_chromadb, mock_chroma_collection):
        """Test checking if experience exists - returns True."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock get response - experience exists
        mock_chroma_collection.get.return_value = {
            'ids': ['test_task_001']
        }
        
        storage = MemoryStorage()
        result = storage.experience_exists("test_task_001")
        
        assert result is True
        mock_chroma_collection.get.assert_called_once_with(ids=["test_task_001"])

    @patch('gui_agent_memory.storage.chromadb')
    def test_experience_exists_false(self, mock_chromadb, mock_chroma_collection):
        """Test checking if experience exists - returns False."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock get response - experience doesn't exist
        mock_chroma_collection.get.return_value = {
            'ids': []
        }
        
        storage = MemoryStorage()
        result = storage.experience_exists("nonexistent_task")
        
        assert result is False

    @patch('gui_agent_memory.storage.chromadb')
    def test_get_collection_stats(self, mock_chromadb, mock_chroma_collection):
        """Test getting collection statistics."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock count responses
        mock_chroma_collection.count.side_effect = [5, 3]  # experiential, then declarative
        
        storage = MemoryStorage()
        storage.experiential_collection = mock_chroma_collection
        storage.declarative_collection = mock_chroma_collection
        
        stats = storage.get_collection_stats()
        
        assert stats['experiential_memories'] == 5
        assert stats['declarative_memories'] == 3
        assert stats['total'] == 8

    @patch('gui_agent_memory.storage.chromadb')
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

    @patch('gui_agent_memory.storage.chromadb')
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