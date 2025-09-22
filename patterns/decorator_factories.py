"""Decorator Factories"""

import functools
import time
import random
from typing import Callable, Any, Optional, Type
import logging


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: tuple = (Exception,)):
    """Decorator factory that retries a function on failure."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e

                    print(f"Attempt {attempts} failed: {e}. Retrying in {current_delay:.1f}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def timer(unit: str = "seconds"):
    """Decorator factory that times function execution."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            duration = end_time - start_time
            if unit == "milliseconds":
                duration *= 1000
                unit_symbol = "ms"
            else:
                unit_symbol = "s"

            print(f"{func.__name__} took {duration:.4f}{unit_symbol}")
            return result

        return wrapper

    return decorator


def log_calls(logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """Decorator factory that logs function calls."""

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_str = ", ".join(repr(arg) for arg in args)
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [args_str, kwargs_str]))

            logger.log(level, f"Calling {func.__name__}({all_args})")

            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                logger.log(logging.ERROR, f"{func.__name__} raised {type(e).__name__}: {e}")
                raise

        return wrapper

    return decorator


def cache(maxsize: int = 128, ttl: Optional[float] = None):
    """Decorator factory that caches function results."""

    def decorator(func: Callable) -> Callable:
        cache_data = {}
        cache_times = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = (args, tuple(sorted(kwargs.items())))

            # Check TTL if specified
            if ttl is not None and key in cache_times:
                if time.time() - cache_times[key] > ttl:
                    cache_data.pop(key, None)
                    cache_times.pop(key, None)

            # Return cached result if available
            if key in cache_data:
                return cache_data[key]

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache (with size limit)
            if len(cache_data) >= maxsize:
                # Remove oldest entry
                oldest_key = next(iter(cache_data))
                cache_data.pop(oldest_key)
                cache_times.pop(oldest_key, None)

            cache_data[key] = result
            if ttl is not None:
                cache_times[key] = time.time()

            return result

        # Add cache inspection methods
        wrapper.cache_info = lambda: {
            'size': len(cache_data),
            'maxsize': maxsize,
            'ttl': ttl
        }
        wrapper.cache_clear = lambda: cache_data.clear() or cache_times.clear()

        return wrapper

    return decorator


def validate_args(**validators):
    """Decorator factory that validates function arguments."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for {param_name}: {value}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def rate_limit(calls_per_second: float):
    """Decorator factory that rate limits function calls."""

    def decorator(func: Callable) -> Callable:
        last_called = [0.0]  # Use list to make it mutable

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            time_since_last = now - last_called[0]
            min_interval = 1.0 / calls_per_second

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)

            last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def conditional(condition: Callable):
    """Decorator factory that conditionally executes a function."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition(*args, **kwargs):
                return func(*args, **kwargs)
            else:
                return None

        return wrapper

    return decorator


# Example usage functions
@retry(max_attempts=3, delay=0.5, exceptions=(ValueError, ConnectionError))
def unreliable_function():
    """Function that randomly fails."""
    if random.random() < 0.7:  # 70% chance of failure
        raise ValueError("Random failure!")
    return "Success!"


@timer(unit="milliseconds")
@cache(maxsize=10, ttl=5.0)
def expensive_calculation(n: int) -> int:
    """Expensive calculation with caching."""
    time.sleep(0.1)  # Simulate expensive operation
    return sum(i * i for i in range(n))


@log_calls()
@validate_args(
    name=lambda x: isinstance(x, str) and len(x) > 0,
    age=lambda x: isinstance(x, int) and 0 <= x <= 150
)
def create_person(name: str, age: int) -> dict:
    """Create a person with validation."""
    return {"name": name, "age": age}


@rate_limit(calls_per_second=2.0)
def api_call(endpoint: str) -> str:
    """Simulated API call with rate limiting."""
    return f"Called {endpoint} at {time.time():.2f}"


@conditional(lambda x: x > 0)
def positive_only_function(x: int) -> int:
    """Function that only executes for positive numbers."""
    return x * 2


def demonstrate_decorator_factories():
    """Demonstrate various decorator factory patterns."""

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print("=== Retry Decorator ===")
    try:
        result = unreliable_function()
        print(f"Result: {result}")
    except ValueError as e:
        print(f"Finally failed: {e}")

    print("\n=== Timer and Cache Decorators ===")
    print(f"First call: {expensive_calculation(100)}")
    print(f"Second call (cached): {expensive_calculation(100)}")
    print(f"Cache info: {expensive_calculation.cache_info()}")

    print("\n=== Validation Decorator ===")
    try:
        person = create_person("Alice", 25)
        print(f"Created: {person}")

        person = create_person("", -5)  # This will fail validation
    except ValueError as e:
        print(f"Validation failed: {e}")

    print("\n=== Rate Limiting ===")
    for i in range(3):
        result = api_call(f"/users/{i}")
        print(result)

    print("\n=== Conditional Execution ===")
    print(f"Positive input: {positive_only_function(5)}")
    print(f"Negative input: {positive_only_function(-3)}")


if __name__ == "__main__":
    demonstrate_decorator_factories()