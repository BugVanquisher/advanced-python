"""Concurrency-Safe Counters"""

import threading
import asyncio
import time
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict


class ThreadSafeCounter:
    """Thread-safe counter using locks.

    Perfect for interviews - demonstrates:
    - Thread synchronization
    - Race condition prevention
    - Lock usage patterns
    """

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """Increment counter atomically."""
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """Decrement counter atomically."""
        with self._lock:
            self._value -= amount
            return self._value

    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value

    def reset(self) -> int:
        """Reset counter to zero."""
        with self._lock:
            old_value = self._value
            self._value = 0
            return old_value

    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """Atomic compare-and-swap operation."""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False


class AsyncCounter:
    """Async-safe counter using asyncio locks."""

    def __init__(self):
        self._value = 0
        self._lock = asyncio.Lock()

    async def increment(self, amount: int = 1) -> int:
        """Increment counter atomically."""
        async with self._lock:
            self._value += amount
            return self._value

    async def decrement(self, amount: int = 1) -> int:
        """Decrement counter atomically."""
        async with self._lock:
            self._value -= amount
            return self._value

    async def get(self) -> int:
        """Get current value."""
        async with self._lock:
            return self._value

    async def reset(self) -> int:
        """Reset counter to zero."""
        async with self._lock:
            old_value = self._value
            self._value = 0
            return old_value


class AtomicCounter:
    """Lock-free counter using atomic operations (simulation).

    Note: Python doesn't have true atomic operations,
    but this demonstrates the concept.
    """

    def __init__(self):
        self._value = 0
        # In real systems, you'd use atomic libraries like:
        # - threading.local for thread-local storage
        # - multiprocessing.Value for process sharing
        # - Redis for distributed counting

    def increment(self, amount: int = 1) -> int:
        """Simulated atomic increment."""
        # In reality, this would be a single atomic CPU instruction
        # Here we simulate it with a very brief lock
        import threading
        with threading.Lock():
            self._value += amount
            return self._value

    def get(self) -> int:
        """Get current value (atomic read)."""
        return self._value


class DistributedCounter:
    """Simulated distributed counter (like Redis)."""

    def __init__(self, name: str):
        self.name = name
        # Simulate Redis with in-memory dict
        self._storage = {}
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """Increment distributed counter."""
        with self._lock:
            current = self._storage.get(self.name, 0)
            new_value = current + amount
            self._storage[self.name] = new_value
            return new_value

    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._storage.get(self.name, 0)

    def reset(self) -> int:
        """Reset counter."""
        with self._lock:
            old_value = self._storage.get(self.name, 0)
            self._storage[self.name] = 0
            return old_value


class MultiCounter:
    """Multiple named counters with thread safety."""

    def __init__(self):
        self._counters = defaultdict(int)
        self._lock = threading.Lock()

    def increment(self, name: str, amount: int = 1) -> int:
        """Increment named counter."""
        with self._lock:
            self._counters[name] += amount
            return self._counters[name]

    def get(self, name: str) -> int:
        """Get counter value."""
        with self._lock:
            return self._counters[name]

    def get_all(self) -> Dict[str, int]:
        """Get all counters."""
        with self._lock:
            return dict(self._counters)

    def reset(self, name: str) -> int:
        """Reset specific counter."""
        with self._lock:
            old_value = self._counters[name]
            self._counters[name] = 0
            return old_value

    def reset_all(self) -> Dict[str, int]:
        """Reset all counters."""
        with self._lock:
            old_values = dict(self._counters)
            self._counters.clear()
            return old_values


class RateLimitedCounter:
    """Counter with rate limiting (common interview follow-up)."""

    def __init__(self, max_per_second: int = 10):
        self._value = 0
        self._max_per_second = max_per_second
        self._timestamps = []
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> bool:
        """Increment if rate limit allows. Returns success."""
        with self._lock:
            now = time.time()

            # Remove old timestamps (older than 1 second)
            self._timestamps = [t for t in self._timestamps if now - t < 1.0]

            # Check rate limit
            if len(self._timestamps) >= self._max_per_second:
                return False  # Rate limited

            # Allow increment
            self._value += amount
            self._timestamps.append(now)
            return True

    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value


def unsafe_counter_demo():
    """Demonstrate race conditions without synchronization."""
    print("=== Unsafe Counter (Race Condition Demo) ===")

    # Unsafe counter
    unsafe_value = 0

    def unsafe_increment():
        nonlocal unsafe_value
        for _ in range(1000):
            # This is NOT thread-safe!
            temp = unsafe_value
            temp += 1
            unsafe_value = temp

    # Run with multiple threads
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=unsafe_increment)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"Unsafe counter result: {unsafe_value} (expected: 5000)")
    print("^ Notice the race condition - we lost increments!")


def thread_safe_counter_demo():
    """Demonstrate thread-safe counter."""
    print("\n=== Thread-Safe Counter ===")

    counter = ThreadSafeCounter()

    def safe_increment():
        for _ in range(1000):
            counter.increment()

    # Run with multiple threads
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=safe_increment)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print(f"Safe counter result: {counter.get()} (expected: 5000)")
    print("^ Perfect! No race conditions.")


async def async_counter_demo():
    """Demonstrate async-safe counter."""
    print("\n=== Async-Safe Counter ===")

    counter = AsyncCounter()

    async def async_increment():
        for _ in range(100):
            await counter.increment()

    # Run with multiple coroutines
    tasks = []
    for _ in range(10):
        task = asyncio.create_task(async_increment())
        tasks.append(task)

    await asyncio.gather(*tasks)

    result = await counter.get()
    print(f"Async counter result: {result} (expected: 1000)")


def multi_counter_demo():
    """Demonstrate multiple named counters."""
    print("\n=== Multi-Counter ===")

    mc = MultiCounter()

    def increment_counters():
        for i in range(100):
            mc.increment("requests")
            if i % 10 == 0:
                mc.increment("errors")
            if i % 5 == 0:
                mc.increment("cache_hits")

    # Run with multiple threads
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(increment_counters) for _ in range(3)]
        for future in futures:
            future.result()

    counters = mc.get_all()
    print(f"Final counters: {counters}")


def rate_limited_counter_demo():
    """Demonstrate rate-limited counter."""
    print("\n=== Rate-Limited Counter ===")

    counter = RateLimitedCounter(max_per_second=5)

    print("Trying to increment 10 times quickly...")
    successes = 0
    for i in range(10):
        if counter.increment():
            successes += 1
            print(f"Increment {i+1}: SUCCESS")
        else:
            print(f"Increment {i+1}: RATE LIMITED")
        time.sleep(0.1)

    print(f"Total successful increments: {successes}")
    print(f"Counter value: {counter.get()}")


def benchmark_counters():
    """Simple performance comparison."""
    print("\n=== Performance Benchmark ===")

    # ThreadSafeCounter
    counter = ThreadSafeCounter()
    start = time.time()
    for _ in range(10000):
        counter.increment()
    thread_safe_time = time.time() - start

    # AtomicCounter (simulated)
    atomic_counter = AtomicCounter()
    start = time.time()
    for _ in range(10000):
        atomic_counter.increment()
    atomic_time = time.time() - start

    print(f"ThreadSafeCounter: {thread_safe_time:.3f}s")
    print(f"AtomicCounter: {atomic_time:.3f}s")
    print(f"Both reached: {counter.get()}, {atomic_counter.get()}")


def interview_questions():
    """Common counter interview questions."""
    print("\n=== Interview Q&A ===")

    print("\nQ: What's a race condition?")
    print("A: Multiple threads accessing shared data simultaneously, causing corruption")

    print("\nQ: How to make counters thread-safe?")
    print("A: Locks, atomic operations, or immutable data structures")

    print("\nQ: Lock vs atomic operations?")
    print("A: Locks: flexible but slower. Atomics: faster but limited operations")

    print("\nQ: What's compare-and-swap?")
    print("A: Atomic operation: update value only if it matches expected value")

    print("\nQ: Distributed counting challenges?")
    print("A: Network latency, consistency vs availability trade-offs")

    print("\nQ: How would you implement in Redis?")
    print("A: INCR command for atomic increment, or Lua scripts for complex operations")


async def main():
    """Main demonstration function."""
    print("=== Concurrency-Safe Counters Demo ===")

    unsafe_counter_demo()
    thread_safe_counter_demo()
    await async_counter_demo()
    multi_counter_demo()
    rate_limited_counter_demo()
    benchmark_counters()
    interview_questions()


if __name__ == "__main__":
    asyncio.run(main())