"""
Unit tests for the log utilities module.
"""

from unittest.mock import Mock, patch

from gui_agent_memory.log_utils import safe_slug, write_json_file, write_text_file


class TestLogUtils:
    """Test cases for log utilities."""

    def test_safe_slug_basic(self):
        """Test safe_slug with basic text."""
        result = safe_slug("test string")
        assert result == "test-string"

    def test_safe_slug_with_special_chars(self):
        """Test safe_slug with special characters."""
        result = safe_slug("test@#$%string")
        assert result == "test-string"

    def test_safe_slug_with_underscores_and_dashes(self):
        """Test safe_slug preserves underscores and dashes."""
        result = safe_slug("test_string-with-dashes")
        assert result == "test_string-with-dashes"

    def test_safe_slug_with_unicode(self):
        """Test safe_slug with unicode characters."""
        result = safe_slug("test-测试-string")
        assert result == "test-测试-string"

    def test_safe_slug_empty_string(self):
        """Test safe_slug with empty string."""
        result = safe_slug("")
        assert result == "unknown"

    def test_safe_slug_none(self):
        """Test safe_slug with None."""
        result = safe_slug(None)
        assert result == "unknown"

    def test_new_operation_dir(self, tmp_path):
        """Test new_operation_dir creates directory."""
        from gui_agent_memory.log_utils import new_operation_dir

        with patch("gui_agent_memory.log_utils.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20240101_120000_000000"
            base_dir = tmp_path
            result = new_operation_dir(base_dir, "test_operation", "test_hint")

            # Should create path with timestamp and hint
            assert "test_operation_20240101_120000_000000" in str(result)
            assert "test_hint" in str(result)
            assert str(base_dir) in str(result)

    def test_write_text_file_success(self, tmp_path):
        """Test successful write_text_file."""
        test_file = tmp_path / "test.txt"
        content = "test content"

        write_text_file(test_file, content)

        assert test_file.exists()
        assert test_file.read_text() == content

    def test_write_json_file_success(self, tmp_path):
        """Test successful write_json_file."""
        test_file = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        write_json_file(test_file, data)

        assert test_file.exists()
        result = test_file.read_text()
        assert "key" in result
        assert "value" in result
        assert "42" in result


class TestLogUtilsCoverage:
    """Tests for uncovered code paths in log_utils.py"""

    def test_safe_slug_empty_none_text(self):
        """Test safe_slug with empty/None text (line 23)."""
        # Test None
        result = safe_slug(None)
        assert result == "unknown"

        # Test empty string
        result = safe_slug("")
        assert result == "unknown"

        # Test whitespace only
        result = safe_slug("   ")
        assert result == "unknown"

        # Test string that becomes empty after cleaning
        result = safe_slug("---___---")
        assert result == "unknown"

    def test_write_text_file_exception_handling(self, tmp_path):
        """Test write_text_file exception handling (cross-platform)."""
        test_file = tmp_path / "readonly" / "test.txt"
        # Mock the logger to capture the exception log and force write failure
        with (
            patch("gui_agent_memory.log_utils.logging.getLogger") as mock_get_logger,
            patch("pathlib.Path.write_text", side_effect=PermissionError("denied")),
        ):
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # This should not raise an exception, just log it
            write_text_file(test_file, "test content")

            # Verify that exception was logged
            mock_logger.exception.assert_called_once()
            args = mock_logger.exception.call_args[0]
            assert "Failed to write text file" in args[0]
            assert str(test_file) in args[1]

    def test_write_text_file_mkdir_exception(self, tmp_path):
        """Test write_text_file when mkdir fails (cross-platform)."""
        bad_path = tmp_path / "noaccess" / "test.txt"

        with (
            patch("gui_agent_memory.log_utils.logging.getLogger") as mock_get_logger,
            patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")),
        ):
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # This should not raise an exception, just log it
            write_text_file(bad_path, "test content")

            # Verify that exception was logged
            mock_logger.exception.assert_called_once()

    def test_write_json_file_exception_handling(self, tmp_path):
        """Test write_json_file exception handling (cross-platform)."""
        test_file = tmp_path / "readonly" / "test.json"
        # Mock the logger to capture the exception log and force write failure
        with (
            patch("gui_agent_memory.log_utils.logging.getLogger") as mock_get_logger,
            patch("pathlib.Path.write_text", side_effect=PermissionError("denied")),
        ):
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # This should not raise an exception, just log it
            write_json_file(test_file, {"test": "data"})

            # Verify that exception was logged
            mock_logger.exception.assert_called_once()
            args = mock_logger.exception.call_args[0]
            assert "Failed to write json file" in args[0]
            assert str(test_file) in args[1]

    def test_write_json_file_serialization_exception(self, tmp_path):
        """Test write_json_file when serialization fails."""
        test_file = tmp_path / "test.json"

        # Create an object that will cause JSON serialization to fail
        class UnserializableClass:
            def __str__(self):
                raise ValueError("Cannot convert to string")

        unserializable_obj = {"good_data": "test", "bad_data": UnserializableClass()}

        with patch("gui_agent_memory.log_utils.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # This should not raise an exception, just log it
            write_json_file(test_file, unserializable_obj)

            # Verify that exception was logged
            mock_logger.exception.assert_called_once()

    def test_write_json_file_mkdir_exception(self, tmp_path):
        """Test write_json_file when mkdir fails (cross-platform)."""
        bad_path = tmp_path / "noaccess" / "test.json"

        with (
            patch("gui_agent_memory.log_utils.logging.getLogger") as mock_get_logger,
            patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")),
        ):
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # This should not raise an exception, just log it
            write_json_file(bad_path, {"test": "data"})

            # Verify that exception was logged
            mock_logger.exception.assert_called_once()
