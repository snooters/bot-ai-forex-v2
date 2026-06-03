import asyncio
import time
import functools
import traceback
from typing import Any, Callable, Optional, Type, Tuple, Union

from core.exceptions import BotError
from utils.logger import get_logger


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    default_return: Any = None,
    raise_on_failure: bool = True,
):
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_async:
                return async_retry_wrapper(*args, **kwargs)
            return sync_retry_wrapper(*args, **kwargs)

        async def async_retry_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            last_exc = None
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
            if raise_on_failure:
                raise last_exc
            return default_return

        def sync_retry_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            last_exc = None
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        try:
                            time.sleep(current_delay)
                        except KeyboardInterrupt:
                            raise
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} retries: {e}")
            if raise_on_failure:
                raise last_exc
            return default_return

        return wrapper
    return decorator


def safe_execute(
    default_return: Any = None,
    log_error: bool = True,
    raise_on_error: bool = False,
):
    def decorator(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if is_async:
                return async_safe_wrapper(*args, **kwargs)
            return sync_safe_wrapper(*args, **kwargs)

        async def async_safe_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            try:
                return await func(*args, **kwargs)
            except BotError as e:
                if log_error:
                    logger.error(f"{func.__name__} error: {e}")
                if raise_on_error:
                    raise
                return default_return
            except Exception as e:
                if log_error:
                    logger.error(f"{func.__name__} unexpected error: {e}\n{traceback.format_exc()}")
                if raise_on_error:
                    raise
                return default_return

        def sync_safe_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except BotError as e:
                if log_error:
                    logger.error(f"{func.__name__} error: {e}")
                if raise_on_error:
                    raise
                return default_return
            except Exception as e:
                if log_error:
                    logger.error(f"{func.__name__} unexpected error: {e}\n{traceback.format_exc()}")
                if raise_on_error:
                    raise
                return default_return

        return wrapper
    return decorator


def measure_time(func: Callable) -> Callable:
    is_async = asyncio.iscoroutinefunction(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_async:
            return async_measure(*args, **kwargs)
        return sync_measure(*args, **kwargs)

    async def async_measure(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 0.1:
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result

    def sync_measure(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 0.1:
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper


def log_entry_exit(func: Callable) -> Callable:
    is_async = asyncio.iscoroutinefunction(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_async:
            return async_log_wrapper(*args, **kwargs)
        return sync_log_wrapper(*args, **kwargs)

    async def async_log_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise

    def sync_log_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise

    return wrapper
