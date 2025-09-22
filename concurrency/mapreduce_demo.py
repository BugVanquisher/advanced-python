"""Parallel MapReduce Demo"""

import multiprocessing as mp
from multiprocessing import Pool, Queue, Process, Manager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Iterable, Dict, Tuple
from functools import reduce, partial
from collections import defaultdict, Counter
import time
import math
import random
import json
from dataclasses import dataclass


@dataclass
class MapReduceResult:
    """Result of a MapReduce operation."""
    result: Any
    map_time: float
    reduce_time: float
    total_time: float
    workers_used: int


class MapReduceFramework:
    """A simple MapReduce framework using multiprocessing."""

    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        print(f"MapReduce initialized with {self.num_workers} workers")

    def map_reduce(self,
                   data: Iterable[Any],
                   map_func: Callable,
                   reduce_func: Callable,
                   chunk_size: int = None) -> MapReduceResult:
        """Execute MapReduce operation."""
        start_time = time.time()

        # Convert to list if needed
        data_list = list(data)
        total_items = len(data_list)

        if chunk_size is None:
            chunk_size = max(1, total_items // self.num_workers)

        print(f"Processing {total_items} items with chunk size {chunk_size}")

        # Map phase
        map_start = time.time()
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Split data into chunks
            chunks = [data_list[i:i + chunk_size]
                     for i in range(0, total_items, chunk_size)]

            print(f"Created {len(chunks)} chunks for mapping")

            # Submit map tasks
            map_futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(self._map_chunk, map_func, chunk, i)
                map_futures.append(future)

            # Collect map results
            map_results = []
            for future in as_completed(map_futures):
                chunk_result = future.result()
                map_results.extend(chunk_result)

        map_time = time.time() - map_start
        print(f"Map phase completed in {map_time:.2f}s")

        # Reduce phase
        reduce_start = time.time()
        if map_results:
            final_result = reduce(reduce_func, map_results)
        else:
            final_result = None

        reduce_time = time.time() - reduce_start
        total_time = time.time() - start_time

        print(f"Reduce phase completed in {reduce_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")

        return MapReduceResult(
            result=final_result,
            map_time=map_time,
            reduce_time=reduce_time,
            total_time=total_time,
            workers_used=self.num_workers
        )

    @staticmethod
    def _map_chunk(map_func: Callable, chunk: List[Any], chunk_id: int) -> List[Any]:
        """Apply map function to a chunk of data."""
        print(f"Worker processing chunk {chunk_id} with {len(chunk)} items")
        results = []
        for item in chunk:
            try:
                result = map_func(item)
                if result is not None:  # Allow filtering
                    results.append(result)
            except Exception as e:
                print(f"Error processing item in chunk {chunk_id}: {e}")
        return results

    def word_count(self, texts: List[str]) -> Dict[str, int]:
        """Classic word count example."""
        def map_func(text: str) -> Dict[str, int]:
            words = text.lower().split()
            return Counter(words)

        def reduce_func(counter1: Counter, counter2: Counter) -> Counter:
            return counter1 + counter2

        result = self.map_reduce(texts, map_func, reduce_func)
        return result

    def sum_squares(self, numbers: List[float]) -> float:
        """Sum of squares example."""
        def map_func(x: float) -> float:
            return x * x

        def reduce_func(a: float, b: float) -> float:
            return a + b

        result = self.map_reduce(numbers, map_func, reduce_func)
        return result

    def find_prime_numbers(self, numbers: List[int]) -> List[int]:
        """Find prime numbers example."""
        def map_func(n: int) -> List[int]:
            if self._is_prime(n):
                return [n]
            return []

        def reduce_func(list1: List[int], list2: List[int]) -> List[int]:
            return list1 + list2

        result = self.map_reduce(numbers, map_func, reduce_func)
        return result

    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True


class DistributedMapReduce:
    """More advanced MapReduce with distributed processing simulation."""

    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.manager = Manager()

    def distributed_map_reduce(self,
                             data: List[Any],
                             map_func: Callable,
                             reduce_func: Callable,
                             partition_func: Callable = None) -> Any:
        """MapReduce with intermediate shuffling and partitioning."""
        print(f"Starting distributed MapReduce with {self.num_workers} workers")

        # Phase 1: Map
        mapped_data = self._distributed_map(data, map_func)

        # Phase 2: Shuffle and partition
        partitioned_data = self._shuffle_and_partition(mapped_data, partition_func)

        # Phase 3: Reduce
        final_result = self._distributed_reduce(partitioned_data, reduce_func)

        return final_result

    def _distributed_map(self, data: List[Any], map_func: Callable) -> List[Tuple[Any, Any]]:
        """Distributed map phase."""
        print("Starting map phase...")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            chunk_size = max(1, len(data) // self.num_workers)
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

            futures = []
            for chunk in chunks:
                future = executor.submit(self._map_and_emit, map_func, chunk)
                futures.append(future)

            mapped_results = []
            for future in as_completed(futures):
                chunk_results = future.result()
                mapped_results.extend(chunk_results)

        print(f"Map phase produced {len(mapped_results)} key-value pairs")
        return mapped_results

    @staticmethod
    def _map_and_emit(map_func: Callable, chunk: List[Any]) -> List[Tuple[Any, Any]]:
        """Apply map function and emit key-value pairs."""
        results = []
        for item in chunk:
            try:
                mapped = map_func(item)
                if isinstance(mapped, (list, tuple)):
                    results.extend(mapped)
                else:
                    results.append(mapped)
            except Exception as e:
                print(f"Error in map function: {e}")
        return results

    def _shuffle_and_partition(self,
                              mapped_data: List[Tuple[Any, Any]],
                              partition_func: Callable = None) -> Dict[int, List[Tuple[Any, Any]]]:
        """Shuffle and partition mapped data."""
        print("Starting shuffle and partition phase...")

        if partition_func is None:
            partition_func = lambda key: hash(key) % self.num_workers

        partitions = defaultdict(list)
        for key, value in mapped_data:
            partition_id = partition_func(key)
            partitions[partition_id].append((key, value))

        print(f"Data partitioned into {len(partitions)} partitions")
        return dict(partitions)

    def _distributed_reduce(self,
                           partitioned_data: Dict[int, List[Tuple[Any, Any]]],
                           reduce_func: Callable) -> Dict[Any, Any]:
        """Distributed reduce phase."""
        print("Starting reduce phase...")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for partition_id, partition_data in partitioned_data.items():
                future = executor.submit(self._reduce_partition, reduce_func, partition_data)
                futures.append(future)

            final_results = {}
            for future in as_completed(futures):
                partition_result = future.result()
                final_results.update(partition_result)

        print(f"Reduce phase completed with {len(final_results)} final results")
        return final_results

    @staticmethod
    def _reduce_partition(reduce_func: Callable,
                         partition_data: List[Tuple[Any, Any]]) -> Dict[Any, Any]:
        """Reduce a single partition."""
        # Group by key
        grouped = defaultdict(list)
        for key, value in partition_data:
            grouped[key].append(value)

        # Apply reduce function to each group
        results = {}
        for key, values in grouped.items():
            try:
                if len(values) == 1:
                    results[key] = values[0]
                else:
                    results[key] = reduce(reduce_func, values)
            except Exception as e:
                print(f"Error reducing key {key}: {e}")

        return results


# Example applications
def word_count_example():
    """Demonstrate word count with MapReduce."""
    print("\n=== Word Count Example ===")

    # Sample texts
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog sleeps in the sun",
        "quick brown fox runs through the forest",
        "the sun shines bright over the forest",
        "lazy dog and quick fox are friends"
    ] * 100  # Multiply to create more data

    mr = MapReduceFramework(num_workers=4)
    result = mr.word_count(texts)

    print(f"\nWord count results (top 10):")
    word_counts = dict(result.result.most_common(10))
    for word, count in word_counts.items():
        print(f"  {word}: {count}")

    print(f"\nPerformance: {result.total_time:.2f}s total")


def mathematical_computation_example():
    """Demonstrate mathematical computation with MapReduce."""
    print("\n=== Mathematical Computation Example ===")

    # Generate large dataset
    numbers = [random.uniform(1, 100) for _ in range(100000)]

    mr = MapReduceFramework(num_workers=4)

    # Sum of squares
    start_time = time.time()
    result = mr.sum_squares(numbers)
    parallel_time = time.time() - start_time

    print(f"Sum of squares (parallel): {result.result:.2f}")
    print(f"Parallel time: {parallel_time:.2f}s")

    # Compare with sequential
    start_time = time.time()
    sequential_result = sum(x * x for x in numbers)
    sequential_time = time.time() - start_time

    print(f"Sum of squares (sequential): {sequential_result:.2f}")
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")


def prime_numbers_example():
    """Demonstrate finding prime numbers with MapReduce."""
    print("\n=== Prime Numbers Example ===")

    # Generate test numbers
    numbers = list(range(1000, 2000))

    mr = MapReduceFramework(num_workers=4)
    result = mr.find_prime_numbers(numbers)

    primes = result.result
    print(f"Found {len(primes)} prime numbers between 1000 and 2000")
    print(f"First 10 primes: {primes[:10]}")
    print(f"Processing time: {result.total_time:.2f}s")


def distributed_word_count_example():
    """Demonstrate distributed MapReduce with word count."""
    print("\n=== Distributed Word Count Example ===")

    # Generate documents
    documents = [
        "machine learning algorithms are powerful tools",
        "deep learning models require large datasets",
        "algorithms process data efficiently",
        "machine learning transforms industries",
        "data science combines statistics and programming"
    ] * 50

    def map_func(document: str) -> List[Tuple[str, int]]:
        """Map function that emits (word, 1) pairs."""
        words = document.lower().split()
        return [(word, 1) for word in words]

    def reduce_func(count1: int, count2: int) -> int:
        """Reduce function that sums counts."""
        return count1 + count2

    dmr = DistributedMapReduce(num_workers=4)
    start_time = time.time()
    result = dmr.distributed_map_reduce(documents, map_func, reduce_func)
    total_time = time.time() - start_time

    print(f"\nDistributed word count results (top 10):")
    sorted_words = sorted(result.items(), key=lambda x: x[1], reverse=True)[:10]
    for word, count in sorted_words:
        print(f"  {word}: {count}")

    print(f"\nProcessing time: {total_time:.2f}s")


def log_analysis_example():
    """Demonstrate log analysis with MapReduce."""
    print("\n=== Log Analysis Example ===")

    # Simulate log entries
    log_entries = []
    ips = ['192.168.1.1', '10.0.0.1', '172.16.0.1', '203.0.113.1']
    status_codes = [200, 404, 500, 301, 403]

    for _ in range(1000):
        ip = random.choice(ips)
        status = random.choice(status_codes)
        size = random.randint(100, 10000)
        log_entries.append(f"{ip} - - [timestamp] GET /path {status} {size}")

    def map_func(log_line: str) -> List[Tuple[str, Dict]]:
        """Extract information from log line."""
        parts = log_line.split()
        if len(parts) >= 7:
            ip = parts[0]
            status = parts[6]
            size = int(parts[7]) if parts[7].isdigit() else 0

            return [
                (f"ip:{ip}", {'requests': 1, 'bytes': size}),
                (f"status:{status}", {'count': 1})
            ]
        return []

    def reduce_func(data1: Dict, data2: Dict) -> Dict:
        """Combine statistics."""
        result = {}
        for key in set(data1.keys()) | set(data2.keys()):
            result[key] = data1.get(key, 0) + data2.get(key, 0)
        return result

    dmr = DistributedMapReduce(num_workers=4)
    result = dmr.distributed_map_reduce(log_entries, map_func, reduce_func)

    print("\nLog analysis results:")
    for key, stats in sorted(result.items()):
        print(f"  {key}: {stats}")


def demonstrate_mapreduce():
    """Main demonstration function."""
    print("🚀 MapReduce Framework Demo")
    print(f"Available CPU cores: {mp.cpu_count()}")

    word_count_example()
    mathematical_computation_example()
    prime_numbers_example()
    distributed_word_count_example()
    log_analysis_example()

    print("\n✨ All MapReduce demonstrations completed!")


if __name__ == "__main__":
    # Ensure proper multiprocessing on all platforms
    mp.set_start_method('spawn', force=True)
    demonstrate_mapreduce()