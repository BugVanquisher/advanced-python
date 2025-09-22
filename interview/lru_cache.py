"""Custom LRU Cache"""

from functools import lru_cache
import time
from typing import Optional, Any


class LRUCache:
    """Simple LRU (Least Recently Used) Cache implementation.

    Perfect for interviews - demonstrates:
    - Hash table + doubly linked list
    - O(1) get/put operations
    - Understanding of cache eviction
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> node

        # Create dummy head and tail nodes
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        """Get value by key. Returns -1 if not found."""
        if key in self.cache:
            node = self.cache[key]
            # Move to front (most recently used)
            self._move_to_front(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        """Put key-value pair. Evicts LRU if at capacity."""
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.value = value
            self._move_to_front(node)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove LRU (tail's previous)
                lru = self.tail.prev
                self._remove_node(lru)
                del self.cache[lru.key]

            # Add new node at front
            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def _move_to_front(self, node):
        """Move existing node to front."""
        self._remove_node(node)
        self._add_to_front(node)

    def _remove_node(self, node):
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node):
        """Add node right after head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def size(self) -> int:
        """Current cache size."""
        return len(self.cache)

    def keys(self) -> list:
        """Get all keys in order (most recent first)."""
        keys = []
        current = self.head.next
        while current != self.tail:
            keys.append(current.key)
            current = current.next
        return keys


class Node:
    """Doubly linked list node."""
    def __init__(self, key: int, value: int):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class SimpleLRU:
    """Even simpler LRU using OrderedDict (Python interview shortcut)."""

    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # Move to end (most recent)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove first (least recent)
            self.cache.popitem(last=False)


# Performance comparison function
def expensive_function(n: int) -> int:
    """Simulate expensive computation."""
    time.sleep(0.01)  # 10ms delay
    return n * n + n


@lru_cache(maxsize=128)
def cached_expensive_function(n: int) -> int:
    """Same function with built-in LRU cache."""
    time.sleep(0.01)
    return n * n + n


def demonstrate_lru_cache():
    """Demonstrate LRU cache implementations."""
    print("=== LRU Cache Demo ===")

    # Test custom LRU cache
    print("\n1. Custom LRU Cache:")
    cache = LRUCache(3)

    # Add some items
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    print(f"Added 1:10, 2:20, 3:30")
    print(f"Keys (most recent first): {cache.keys()}")

    # Access item 1 (moves to front)
    print(f"\nGet key 1: {cache.get(1)}")
    print(f"Keys after access: {cache.keys()}")

    # Add item 4 (should evict least recent)
    cache.put(4, 40)
    print(f"\nAdded 4:40 (should evict key 2)")
    print(f"Keys: {cache.keys()}")
    print(f"Try get key 2: {cache.get(2)}")  # Should return -1

    # Test SimpleLRU
    print("\n2. Simple LRU (OrderedDict):")
    simple = SimpleLRU(2)
    simple.put(1, 100)
    simple.put(2, 200)
    simple.put(3, 300)  # Should evict key 1
    print(f"Get key 1: {simple.get(1)}")  # Should return -1
    print(f"Get key 2: {simple.get(2)}")  # Should return 200

    # Performance comparison
    print("\n3. Performance Comparison:")

    # Without cache
    start = time.time()
    results = [expensive_function(i % 5) for i in range(20)]
    no_cache_time = time.time() - start
    print(f"Without cache: {no_cache_time:.3f}s")

    # With built-in cache
    start = time.time()
    results = [cached_expensive_function(i % 5) for i in range(20)]
    with_cache_time = time.time() - start
    print(f"With built-in cache: {with_cache_time:.3f}s")

    # Custom cache with function wrapper
    function_cache = LRUCache(10)

    def cached_function(n: int) -> int:
        cached_result = function_cache.get(n)
        if cached_result != -1:
            return cached_result

        result = expensive_function(n)
        function_cache.put(n, result)
        return result

    start = time.time()
    results = [cached_function(i % 5) for i in range(20)]
    custom_cache_time = time.time() - start
    print(f"With custom cache: {custom_cache_time:.3f}s")

    print(f"\nSpeedup with caching: {no_cache_time / with_cache_time:.1f}x")


# Interview questions and answers
def interview_questions():
    """Common LRU cache interview questions."""
    print("\n=== Interview Q&A ===")

    print("\nQ: What's the time complexity of LRU cache operations?")
    print("A: O(1) for both get() and put() operations")

    print("\nQ: What data structures are used?")
    print("A: Hash table for O(1) lookup + Doubly linked list for O(1) insertion/deletion")

    print("\nQ: Why doubly linked list?")
    print("A: Need to remove nodes from middle (requires prev pointer) and add to front/back")

    print("\nQ: How do you handle capacity overflow?")
    print("A: Remove the least recently used item (tail of linked list)")

    print("\nQ: Alternative implementation?")
    print("A: Python's OrderedDict provides move_to_end() for simpler implementation")


if __name__ == "__main__":
    demonstrate_lru_cache()
    interview_questions()