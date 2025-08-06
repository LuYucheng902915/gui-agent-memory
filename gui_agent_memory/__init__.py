"""
Enhanced Memory Module (RAG-based) for GUI Agent

This module provides long-term storage, management, and retrieval of:
- Episodic Memory: Reusable operational flows and experiences
- Semantic Memory: Objective facts and rules about systems and applications
"""

# Fix SQLite version compatibility for ChromaDB before any other imports
try:
    import sys

    import pysqlite3 as sqlite3

    sys.modules["sqlite3"] = sqlite3
except ImportError:
    pass

__version__ = "1.0.0"
__author__ = "GUI Agent Team"

from .main import MemorySystem
from .models import ActionStep, ExperienceRecord, FactRecord

__all__ = ["MemorySystem", "ActionStep", "ExperienceRecord", "FactRecord"]
