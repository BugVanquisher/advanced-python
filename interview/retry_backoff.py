"""Retry with Exponential Backoff"""

import asyncio
import time
import random
from typing import Callable, Any, Optional, List
from functools import wraps


def retry_with_backoff(max_attempts: int = 3,
                      initial_delay: float = 1.0,
                      max_delay: float = 60.0,
                      backoff_factor: float = 2.0,
                      jitter: bool = True):
    """Retry decorator with exponential backoff.

    Perfect for interviews - demonstrates:
    - Exponential backoff algorithm
    - Decorator pattern
    - Error handling strategies
    - Jitter to prevent thundering herd
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        print(f"Success on attempt {attempt}")
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts:
                        print(f"Failed after {max_attempts} attempts")
                        break

                    # Calculate delay with jitter
                    actual_delay = delay
                    if jitter:
                        # Add random jitter (±25%)
                        jitter_range = delay * 0.25
                        actual_delay += random.uniform(-jitter_range, jitter_range)

                    actual_delay = min(actual_delay, max_delay)
                    print(f"Attempt {attempt} failed: {e}. Retrying in {actual_delay:.1f}s...")
                    time.sleep(actual_delay)
                    delay *= backoff_factor

            raise last_exception
        return wrapper
    return decorator


def async_retry_with_backoff(max_attempts: int = 3,
                             initial_delay: float = 1.0,
                             max_delay: float = 60.0,
                             backoff_factor: float = 2.0,
                             jitter: bool = True):
    """Async retry decorator with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 1:
                        print(f"Success on attempt {attempt}")
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts:
                        print(f"Failed after {max_attempts} attempts")
                        break

                    # Calculate delay with jitter
                    actual_delay = delay
                    if jitter:
                        jitter_range = delay * 0.25
                        actual_delay += random.uniform(-jitter_range, jitter_range)

                    actual_delay = min(actual_delay, max_delay)
                    print(f"Attempt {attempt} failed: {e}. Retrying in {actual_delay:.1f}s...")
                    await asyncio.sleep(actual_delay)
                    delay *= backoff_factor

            raise last_exception
        return wrapper
    return decorator


class RetryManager:
    """Class-based retry manager for more control."""

    def __init__(self,
                 max_attempts: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 retryable_exceptions: tuple = (Exception,)):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retryable_exceptions = retryable_exceptions

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        delay = self.initial_delay

        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_exception = e
                if attempt == self.max_attempts:
                    break

                print(f"Attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)
            except Exception as e:
                # Non-retryable exception
                print(f"Non-retryable exception: {e}")
                raise

        raise last_exception

    async def async_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        delay = self.initial_delay

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_exception = e
                if attempt == self.max_attempts:
                    break

                print(f"Attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(delay * self.backoff_factor, self.max_delay)
            except Exception as e:
                print(f"Non-retryable exception: {e}")
                raise

        raise last_exception


# Circuit breaker pattern (often asked together with retry)
class CircuitBreaker:
    """Simple circuit breaker to prevent cascading failures."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = 'HALF_OPEN'
                print("Circuit breaker: HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
                print("Circuit breaker: CLOSED (recovered)")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                print(f"Circuit breaker: OPEN (failures: {self.failure_count})")
            raise


# Example functions for testing
def unreliable_function(success_rate: float = 0.3) -> str:
    """Function that fails most of the time."""
    if random.random() < success_rate:
        return "Success!"
    else:
        raise ConnectionError("Network error")


async def async_unreliable_function(success_rate: float = 0.3) -> str:
    """Async function that fails most of the time."""
    await asyncio.sleep(0.1)  # Simulate async work
    if random.random() < success_rate:
        return "Async success!"
    else:
        raise ConnectionError("Async network error")


def demonstrate_retry_patterns():
    """Demonstrate retry implementations."""
    print("=== Retry with Exponential Backoff Demo ===")

    # 1. Decorator-based retry
    print("\n1. Decorator-based Retry:")

    @retry_with_backoff(max_attempts=3, initial_delay=0.5, backoff_factor=2.0)
    def flaky_api_call():
        return unreliable_function(success_rate=0.4)

    try:
        result = flaky_api_call()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Final failure: {e}")

    # 2. Class-based retry manager
    print("\n2. Class-based Retry Manager:")
    retry_manager = RetryManager(
        max_attempts=4,
        initial_delay=0.3,
        retryable_exceptions=(ConnectionError,)
    )

    try:
        result = retry_manager.execute(unreliable_function, success_rate=0.5)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Final failure: {e}")

    # 3. Circuit breaker
    print("\n3. Circuit Breaker Pattern:")
    circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=2.0)

    for i in range(8):
        try:
            result = circuit_breaker.call(unreliable_function, success_rate=0.1)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: {e}")
        time.sleep(0.5)


async def demonstrate_async_retry():
    """Demonstrate async retry patterns."""
    print("\n4. Async Retry:")

    @async_retry_with_backoff(max_attempts=3, initial_delay=0.2)
    async def async_flaky_call():
        return await async_unreliable_function(success_rate=0.4)

    try:
        result = await async_flaky_call()
        print(f"Async result: {result}")
    except Exception as e:
        print(f"Async final failure: {e}")

    # Async retry manager
    print("\n5. Async Retry Manager:")
    retry_manager = RetryManager(max_attempts=3, initial_delay=0.2)

    try:
        result = await retry_manager.async_execute(
            async_unreliable_function, success_rate=0.6
        )
        print(f"Async manager result: {result}")
    except Exception as e:
        print(f"Async manager failure: {e}")


def calculate_backoff_delays(initial_delay: float = 1.0,
                           backoff_factor: float = 2.0,
                           max_attempts: int = 5) -> List[float]:
    """Calculate backoff delays for demonstration."""
    delays = []
    delay = initial_delay
    for attempt in range(1, max_attempts):
        delays.append(delay)
        delay *= backoff_factor
    return delays


def interview_questions():
    """Common retry/backoff interview questions."""
    print("\n=== Interview Q&A ===")

    print("\nQ: Why use exponential backoff?")
    print("A: Reduces load on failing service, increases chance of recovery")

    print("\nQ: What's the purpose of jitter?")
    print("A: Prevents thundering herd - all clients retrying at same time")

    print("\nQ: When NOT to retry?")
    print("A: Client errors (4xx), authentication failures, validation errors")

    print("\nQ: What's a circuit breaker?")
    print("A: Prevents calls to failing service, allows recovery time")

    print("\nQ: Backoff sequence example:")
    delays = calculate_backoff_delays(1.0, 2.0, 6)
    print(f"A: {delays} seconds (1, 2, 4, 8, 16...)")

    print("\nQ: Alternative strategies?")
    print("A: Linear backoff, fixed delay, deadline-based retry")


async def main():
    """Main demo function."""
    demonstrate_retry_patterns()
    await demonstrate_async_retry()
    interview_questions()


if __name__ == "__main__":
    asyncio.run(main())