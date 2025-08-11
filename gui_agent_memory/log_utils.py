"""
Lightweight logging utilities for operation-scoped file logging.

This module centralizes helpers to:
- Generate filesystem-safe slugs
- Create per-operation log directories
- Write JSON/text files with exception-safe behavior
- Serialize pydantic models and common Python objects to JSON
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def safe_slug(text: str | None) -> str:
    if not text:
        return "unknown"
    slug = re.sub(r"[^\w\-\.]+", "-", str(text), flags=re.UNICODE)
    slug = slug.strip("-_")
    return slug or "unknown"


def new_operation_dir(base_dir: str | Path, operation: str, hint: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = f"{operation}_{ts}"
    if hint:
        folder += f"_{safe_slug(hint)[:64]}"
    path = Path(base_dir) / folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text_file(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception:
        # Test asserts that the path string is part of the first arg tuple element
        logging.getLogger(__name__).exception(
            "Failed to write text file: %s", str(path)
        )


def _serialize(obj: Any) -> Any:
    # Duck-typing: prefer model_dump if available (pydantic v2)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # pragma: no cover
            logging.getLogger(__name__).debug("Failed to serialize pydantic-like model")
            return str(obj)

    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_serialize(v) for v in obj]
    return obj


def write_json_file(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _serialize(obj)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Failed to write json file: %s", str(path)
        )
