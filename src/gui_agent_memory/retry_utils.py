"""
Retry utilities for wrapping LLM/API calls with a consistent policy.

Usage:

from .retry_utils import retry_llm_call

@retry_llm_call
def my_llm_func(...):
    ...
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _custom_before_sleep_log(logger: logging.Logger, decorated_name: str):
    """Creates a before_sleep callback that always logs the decorated function name."""

    def _log(retry_state) -> None:
        fn_name = decorated_name
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            f"Retrying {fn_name} in {retry_state.next_action.sleep:.2f}s "
            f"(attempt {retry_state.attempt_number}) due to: {exc}"
        )

    return _log


def retry_llm_call(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorator that retries LLM/API calls with exponential backoff + jitter.

    - 3 attempts by default
    - initial backoff ~0.5s, doubling up to ~4s
    - logs a warning before each retry
    - re-raises the last exception after retries are exhausted
    """

    logger = logging.getLogger(func.__module__)
    decorated_name = getattr(
        func, "__qualname__", getattr(func, "__name__", "function")
    )

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # Lazy import avoids circular deps at module import time
        from .config import get_config

        cfg = get_config()
        if not cfg.llm_retry_enabled:
            return cast(_R, func(*args, **kwargs))

        retrying = Retrying(
            stop=stop_after_attempt(max(1, int(cfg.llm_retry_attempts))),
            wait=wait_exponential_jitter(
                initial=max(0.0, float(cfg.llm_retry_initial_seconds)),
                max=max(0.0, float(cfg.llm_retry_max_seconds)),
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=_custom_before_sleep_log(logger, decorated_name),
        )
        for attempt in retrying:
            with attempt:
                return cast(_R, func(*args, **kwargs))
        raise RuntimeError("retry_llm_call: exhausted without raising")

    return wrapper
