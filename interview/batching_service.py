"""Batching Service"""

import asyncio
import threading
import time
from typing import List, Callable, Any, Optional
from dataclasses import dataclass
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor


@dataclass
class BatchItem:
    """Item to be batched and processed."""
    id: str
    data: Any
    timestamp: float
    callback: Optional[Callable] = None


class SimpleBatchingService:
    """Simple batching service for interviews.

    Perfect for interviews - demonstrates:
    - Producer-consumer pattern
    - Batch processing optimization
    - Threading and synchronization
    - Performance vs latency trade-offs
    """

    def __init__(self,
                 batch_size: int = 10,
                 max_wait_time: float = 1.0,
                 processor: Callable[[List[BatchItem]], List[Any]] = None):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor = processor or self._default_processor

        self.queue = Queue()
        self.running = False
        self.worker_thread = None
        self.stats = {
            'items_received': 0,
            'batches_processed': 0,
            'total_processing_time': 0.0
        }

    def start(self):
        """Start the batching service."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop)
        self.worker_thread.start()
        print(f"Batching service started (batch_size={self.batch_size}, max_wait={self.max_wait_time}s)")

    def stop(self):
        """Stop the batching service."""
        if not self.running:
            return

        self.running = False
        # Send sentinel to wake up worker
        self.queue.put(None)

        if self.worker_thread:
            self.worker_thread.join()

        print(f"Batching service stopped. Stats: {self.stats}")

    def submit(self, item_id: str, data: Any, callback: Callable = None) -> None:
        """Submit an item for batch processing."""
        if not self.running:
            raise RuntimeError("Service not running")

        item = BatchItem(
            id=item_id,
            data=data,
            timestamp=time.time(),
            callback=callback
        )

        self.queue.put(item)
        self.stats['items_received'] += 1

    def _worker_loop(self):
        """Main worker loop that collects and processes batches."""
        batch = []
        last_batch_time = time.time()

        while self.running:
            try:
                # Calculate timeout for queue.get()
                elapsed = time.time() - last_batch_time
                timeout = max(0.1, self.max_wait_time - elapsed)

                # Try to get an item
                try:
                    item = self.queue.get(timeout=timeout)
                    if item is None:  # Sentinel value
                        break
                    batch.append(item)
                except Empty:
                    pass  # Timeout, check if we should process current batch

                # Process batch if conditions are met
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - last_batch_time >= self.max_wait_time)
                )

                if should_process:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()

            except Exception as e:
                print(f"Error in worker loop: {e}")

        # Process remaining items before shutdown
        if batch:
            self._process_batch(batch)

    def _process_batch(self, batch: List[BatchItem]):
        """Process a batch of items."""
        if not batch:
            return

        start_time = time.time()
        print(f"Processing batch of {len(batch)} items...")

        try:
            # Process the batch
            results = self.processor(batch)

            # Call individual callbacks if provided
            for item, result in zip(batch, results):
                if item.callback:
                    try:
                        item.callback(result)
                    except Exception as e:
                        print(f"Callback error for item {item.id}: {e}")

            # Update stats
            processing_time = time.time() - start_time
            self.stats['batches_processed'] += 1
            self.stats['total_processing_time'] += processing_time

            print(f"Batch processed in {processing_time:.3f}s")

        except Exception as e:
            print(f"Batch processing failed: {e}")

    def _default_processor(self, batch: List[BatchItem]) -> List[Any]:
        """Default processor - just returns the data."""
        time.sleep(0.1)  # Simulate processing time
        return [f"processed_{item.data}" for item in batch]

    def get_stats(self) -> dict:
        """Get service statistics."""
        stats = self.stats.copy()
        if stats['batches_processed'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['batches_processed']
        else:
            stats['avg_processing_time'] = 0.0
        return stats


class AsyncBatchingService:
    """Async version of batching service."""

    def __init__(self,
                 batch_size: int = 10,
                 max_wait_time: float = 1.0,
                 processor: Callable = None):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor = processor or self._default_processor

        self.queue = asyncio.Queue()
        self.running = False
        self.worker_task = None
        self.stats = {
            'items_received': 0,
            'batches_processed': 0,
            'total_processing_time': 0.0
        }

    async def start(self):
        """Start the async batching service."""
        if self.running:
            return

        self.running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        print(f"Async batching service started")

    async def stop(self):
        """Stop the async batching service."""
        if not self.running:
            return

        self.running = False
        await self.queue.put(None)  # Sentinel

        if self.worker_task:
            await self.worker_task

        print(f"Async batching service stopped. Stats: {self.stats}")

    async def submit(self, item_id: str, data: Any) -> Any:
        """Submit item and wait for result."""
        if not self.running:
            raise RuntimeError("Service not running")

        future = asyncio.Future()
        item = BatchItem(
            id=item_id,
            data=data,
            timestamp=time.time(),
            callback=lambda result: future.set_result(result) if not future.done() else None
        )

        await self.queue.put(item)
        self.stats['items_received'] += 1
        return await future

    async def _worker_loop(self):
        """Async worker loop."""
        batch = []
        last_batch_time = time.time()

        while self.running:
            try:
                # Calculate timeout
                elapsed = time.time() - last_batch_time
                timeout = max(0.1, self.max_wait_time - elapsed)

                # Try to get an item
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    if item is None:  # Sentinel
                        break
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass  # Timeout, check if we should process

                # Process batch if conditions are met
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - last_batch_time >= self.max_wait_time)
                )

                if should_process:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()

            except Exception as e:
                print(f"Error in async worker loop: {e}")

        # Process remaining items
        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[BatchItem]):
        """Process batch asynchronously."""
        if not batch:
            return

        start_time = time.time()
        print(f"Async processing batch of {len(batch)} items...")

        try:
            results = await self.processor(batch)

            # Call callbacks
            for item, result in zip(batch, results):
                if item.callback:
                    try:
                        item.callback(result)
                    except Exception as e:
                        print(f"Async callback error for item {item.id}: {e}")

            # Update stats
            processing_time = time.time() - start_time
            self.stats['batches_processed'] += 1
            self.stats['total_processing_time'] += processing_time

        except Exception as e:
            print(f"Async batch processing failed: {e}")

    async def _default_processor(self, batch: List[BatchItem]) -> List[Any]:
        """Default async processor."""
        await asyncio.sleep(0.1)  # Simulate async processing
        return [f"async_processed_{item.data}" for item in batch]


# Example processors for different use cases
def database_batch_processor(batch: List[BatchItem]) -> List[Any]:
    """Simulate batch database operations."""
    print(f"  Executing batch SQL with {len(batch)} records...")
    time.sleep(0.05 * len(batch))  # Simulate DB time
    return [f"db_result_{item.data}" for item in batch]


async def api_batch_processor(batch: List[BatchItem]) -> List[Any]:
    """Simulate batch API calls."""
    print(f"  Making batch API call with {len(batch)} requests...")
    await asyncio.sleep(0.1)  # Simulate network time
    return [f"api_response_{item.data}" for item in batch]


def demonstrate_simple_batching():
    """Demonstrate simple batching service."""
    print("=== Simple Batching Service Demo ===")

    # Create service with custom processor
    service = SimpleBatchingService(
        batch_size=3,
        max_wait_time=2.0,
        processor=database_batch_processor
    )

    # Start service
    service.start()

    # Submit items with different timing
    print("\nSubmitting items...")
    for i in range(8):
        service.submit(f"item_{i}", f"data_{i}")
        if i == 2:
            time.sleep(1.5)  # Pause to trigger time-based batching

    # Let it process
    time.sleep(3)

    # Stop service
    service.stop()
    print(f"Final stats: {service.get_stats()}")


async def demonstrate_async_batching():
    """Demonstrate async batching service."""
    print("\n=== Async Batching Service Demo ===")

    # Create async service
    service = AsyncBatchingService(
        batch_size=2,
        max_wait_time=1.0,
        processor=api_batch_processor
    )

    # Start service
    await service.start()

    # Submit items and get results
    print("\nSubmitting async items...")
    tasks = []
    for i in range(5):
        task = asyncio.create_task(service.submit(f"async_item_{i}", f"async_data_{i}"))
        tasks.append(task)
        await asyncio.sleep(0.3)

    # Wait for all results
    results = await asyncio.gather(*tasks)
    print(f"Results: {results}")

    # Stop service
    await service.stop()


def demonstrate_performance_comparison():
    """Compare individual vs batch processing performance."""
    print("\n=== Performance Comparison ===")

    def individual_processor(data):
        """Process items one by one."""
        time.sleep(0.01)  # 10ms per item
        return f"individual_{data}"

    def batch_processor_efficient(batch: List[BatchItem]) -> List[Any]:
        """Efficient batch processor."""
        time.sleep(0.02)  # 20ms for entire batch (vs 10ms * batch_size)
        return [f"batch_{item.data}" for item in batch]

    # Individual processing
    start = time.time()
    individual_results = []
    for i in range(10):
        result = individual_processor(f"data_{i}")
        individual_results.append(result)
    individual_time = time.time() - start

    # Batch processing
    service = SimpleBatchingService(
        batch_size=5,
        max_wait_time=0.1,
        processor=batch_processor_efficient
    )
    service.start()

    start = time.time()
    for i in range(10):
        service.submit(f"item_{i}", f"data_{i}")
    time.sleep(1)  # Wait for processing
    service.stop()
    batch_time = time.time() - start

    print(f"Individual processing: {individual_time:.3f}s")
    print(f"Batch processing: {batch_time:.3f}s")
    print(f"Speedup: {individual_time / batch_time:.1f}x")


def interview_questions():
    """Common batching interview questions."""
    print("\n=== Interview Q&A ===")

    print("\nQ: Why use batching?")
    print("A: Reduce overhead, improve throughput, optimize I/O operations")

    print("\nQ: Trade-offs of batching?")
    print("A: Higher latency vs better throughput, memory usage, complexity")

    print("\nQ: When to process a batch?")
    print("A: When batch is full OR max wait time exceeded")

    print("\nQ: How to handle failures?")
    print("A: Retry entire batch, partial retry, dead letter queue")

    print("\nQ: Real-world examples?")
    print("A: Database bulk inserts, API rate limiting, log aggregation")

    print("\nQ: Alternative patterns?")
    print("A: Streaming, event sourcing, message queues (Kafka/RabbitMQ)")


async def main():
    """Main demonstration function."""
    print("=== Batching Service Demo ===")

    demonstrate_simple_batching()
    await demonstrate_async_batching()
    demonstrate_performance_comparison()
    interview_questions()


if __name__ == "__main__":
    asyncio.run(main())