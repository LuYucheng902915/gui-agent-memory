"""
Test suite for __init__.py module.
Tests module initialization and import behavior.
"""

import sys
from unittest.mock import patch

import gui_agent_memory


class TestInitModule:
    """Test cases for the __init__ module."""

    def test_module_metadata(self):
        """Test that module metadata is correctly defined."""
        assert gui_agent_memory.__version__ == "0.1.0"
        assert gui_agent_memory.__author__ == "GUI Agent Team"

    def test_public_api_exports(self):
        """Test that all expected public API components are exported."""
        expected_exports = [
            "ActionStep",
            "ExperienceRecord",
            "FactRecord",
            "MemorySystem",
        ]

        for export in expected_exports:
            assert hasattr(gui_agent_memory, export), f"Missing export: {export}"

        assert gui_agent_memory.__all__ == expected_exports

    def test_import_memory_system(self):
        """Test that MemorySystem can be imported and used."""
        from gui_agent_memory import MemorySystem

        assert MemorySystem is not None
        assert hasattr(MemorySystem, "__init__")

    def test_import_models(self):
        """Test that model classes can be imported and used."""
        from gui_agent_memory import ActionStep, ExperienceRecord, FactRecord

        # Test ActionStep
        action = ActionStep(
            thought="test thought",
            action="click",
            target_element_description="test element",
        )
        assert action.thought == "test thought"
        assert action.action == "click"
        assert action.target_element_description == "test element"

        # Test ExperienceRecord with ActionStep
        experience = ExperienceRecord(
            task_description="test task",
            keywords=["test"],
            action_flow=[action],
            preconditions="test preconditions",
            is_successful=True,
            source_task_id="test_task_id",
        )
        assert experience.task_description == "test task"
        assert len(experience.action_flow) == 1
        assert experience.action_flow[0] == action

        # Test FactRecord
        fact = FactRecord(
            content="test fact content", keywords=["test", "fact"], source="test source"
        )
        assert fact.content == "test fact content"
        assert fact.keywords == ["test", "fact"]
        assert fact.source == "test source"


class TestSQLiteCompatibilityFix:
    """Test SQLite compatibility handling in __init__."""

    def test_pysqlite3_import_success(self):
        """Test successful pysqlite3 import and sys.modules replacement."""
        # This test verifies that if pysqlite3 is available, the compatibility fix works
        # We'll test by checking that the module initialization doesn't fail

        # Clear any existing sqlite3 from sys.modules for this test
        original_sqlite3 = sys.modules.get("sqlite3")
        if "sqlite3" in sys.modules:
            del sys.modules["sqlite3"]

        try:
            # Mock the pysqlite3 import to simulate it being available
            original_import = (
                __builtins__.__import__
                if hasattr(__builtins__, "__import__")
                else __builtins__["__import__"]  # type: ignore[index]
            )

            with patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "pysqlite3":
                        # Create a mock sqlite3 module
                        import types

                        mock_sqlite3 = types.ModuleType("pysqlite3")
                        return mock_sqlite3
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                # Re-import the module to trigger the initialization code
                import importlib

                importlib.reload(gui_agent_memory)

                # Verify that sqlite3 was replaced in sys.modules
                # The exact object doesn't matter, just that the replacement happened
                assert "sqlite3" in sys.modules

        finally:
            # Restore original state
            if original_sqlite3:
                sys.modules["sqlite3"] = original_sqlite3
            elif "sqlite3" in sys.modules:
                del sys.modules["sqlite3"]

    def test_pysqlite3_import_failure(self):
        """Test graceful handling when pysqlite3 import fails."""
        # This test verifies that when pysqlite3 is not available, the module still works

        # Clear any existing sqlite3 from sys.modules for this test
        original_sqlite3 = sys.modules.get("sqlite3")
        if "sqlite3" in sys.modules:
            del sys.modules["sqlite3"]

        try:
            # Mock the pysqlite3 import to simulate it not being available
            original_import = (
                __builtins__.__import__
                if hasattr(__builtins__, "__import__")
                else __builtins__["__import__"]  # type: ignore[index]
            )

            with patch("builtins.__import__") as mock_import:

                def side_effect(name, *args, **kwargs):
                    if name == "pysqlite3":
                        raise ImportError("No module named pysqlite3")
                    return original_import(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                # Re-import the module to trigger the initialization code
                import importlib

                importlib.reload(gui_agent_memory)

                # Verify that the ImportError was caught gracefully
                # and the module still works (can be imported without exceptions)
                # The test passes if we get here without exception
                assert True

        finally:
            # Restore original state
            if original_sqlite3:
                sys.modules["sqlite3"] = original_sqlite3
            elif "sqlite3" in sys.modules:
                del sys.modules["sqlite3"]

    def test_module_imports_after_sqlite_fix(self):
        """Test that main imports work correctly after SQLite compatibility fix."""
        # This test ensures that the SQLite fix doesn't interfere with normal imports
        from gui_agent_memory.main import MemorySystem
        from gui_agent_memory.models import ActionStep, ExperienceRecord, FactRecord

        # Test that classes are properly imported and functional
        assert MemorySystem is not None
        assert ActionStep is not None
        assert ExperienceRecord is not None
        assert FactRecord is not None

        # Test that we can create instances (basic functionality)
        action = ActionStep(
            thought="test", action="test", target_element_description="test"
        )
        assert action is not None
