"""Batching Workers"""

import asyncio
import threading
import time
from typing import List, Callable, Any, Optional, Dict
from dataclasses import dataclass
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import uuid


@dataclass
class BatchItem:
    """Item to be processed in a batch."""
    id: str
    data: Any
    timestamp: float
    future: Optional[asyncio.Future] = None
    callback: Optional[Callable] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


class AsyncBatchProcessor:
    """Asynchronous batch processor."""

    def __init__(self,
                 batch_size: int = 10,
                 max_wait_time: float = 1.0,
                 processor: Optional[Callable] = None):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor = processor or self._default_processor
        self.queue = asyncio.Queue()
        self.running = False
        self.worker_task = None

    async def _default_processor(self, items: List[BatchItem]) -> List[Any]:
        """Default processor that just returns the data."""
        await asyncio.sleep(0.1)  # Simulate processing
        return [item.data for item in items]

    async def start(self):
        """Start the batch processor."""
        if self.running:
            return

        self.running = True
        self.worker_task = asyncio.create_task(self._worker())
        print("AsyncBatchProcessor started")

    async def stop(self):
        """Stop the batch processor."""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        print("AsyncBatchProcessor stopped")

    async def submit(self, data: Any) -> Any:
        """Submit an item for batch processing."""
        future = asyncio.Future()
        item = BatchItem(
            id=str(uuid.uuid4()),
            data=data,
            timestamp=time.time(),
            future=future
        )

        await self.queue.put(item)
        return await future

    async def _worker(self):
        """Worker coroutine that processes batches."""
        batch = []
        last_batch_time = time.time()

        while self.running:
            try:
                # Try to get an item with timeout
                try:
                    item = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=max(0.1, self.max_wait_time - (time.time() - last_batch_time))
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass

                # Process batch if conditions are met
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - last_batch_time >= self.max_wait_time)
                )

                if should_process and batch:
                    await self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()

            except asyncio.CancelledError:
                # Process remaining items before shutting down
                if batch:
                    await self._process_batch(batch)
                break
            except Exception as e:
                print(f"Error in batch worker: {e}")

    async def _process_batch(self, batch: List[BatchItem]):
        """Process a batch of items."""
        try:
            print(f"Processing batch of {len(batch)} items")
            results = await self.processor(batch)

            # Set results for futures
            for item, result in zip(batch, results):
                if item.future and not item.future.done():
                    item.future.set_result(result)

        except Exception as e:
            print(f"Batch processing failed: {e}")
            # Set exception for all futures
            for item in batch:
                if item.future and not item.future.done():
                    item.future.set_exception(e)


class ThreadedBatchProcessor:
    """Threaded batch processor."""

    def __init__(self,
                 batch_size: int = 10,
                 max_wait_time: float = 1.0,
                 processor: Optional[Callable] = None,
                 num_workers: int = 2):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor = processor or self._default_processor
        self.num_workers = num_workers

        self.input_queue = Queue()
        self.result_futures = {}
        self.running = False
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _default_processor(self, items: List[BatchItem]) -> List[Any]:
        """Default processor that just returns the data."""
        time.sleep(0.1)  # Simulate processing
        return [item.data for item in items]

    def start(self):
        """Start the batch processor."""
        if self.running:
            return

        self.running = True

        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        print(f"ThreadedBatchProcessor started with {self.num_workers} workers")

    def stop(self):
        """Stop the batch processor."""
        self.running = False

        # Add sentinel values to wake up workers
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)

        self.executor.shutdown(wait=True)
        print("ThreadedBatchProcessor stopped")

    def submit(self, data: Any) -> Any:
        """Submit an item for batch processing."""
        future = self.executor.submit(lambda: None)  # Placeholder
        item = BatchItem(
            id=str(uuid.uuid4()),
            data=data,
            timestamp=time.time()
        )

        # Store the future for this item
        self.result_futures[item.id] = future
        self.input_queue.put(item)

        return future.result  # This will block until result is available

    def _worker(self, worker_id: int):
        """Worker thread that processes batches."""
        batch = []
        last_batch_time = time.time()

        print(f"Worker {worker_id} started")

        while self.running:
            try:
                # Try to get an item with timeout
                try:
                    timeout = max(0.1, self.max_wait_time - (time.time() - last_batch_time))
                    item = self.input_queue.get(timeout=timeout)

                    if item is None:  # Sentinel value
                        break

                    batch.append(item)

                except Empty:
                    pass

                # Process batch if conditions are met
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - last_batch_time >= self.max_wait_time)
                )

                if should_process and batch:
                    self._process_batch(batch, worker_id)
                    batch = []
                    last_batch_time = time.time()

            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")

        # Process remaining items
        if batch:
            self._process_batch(batch, worker_id)

        print(f"Worker {worker_id} stopped")

    def _process_batch(self, batch: List[BatchItem], worker_id: int):
        """Process a batch of items."""
        try:
            print(f"Worker {worker_id} processing batch of {len(batch)} items")
            results = self.processor(batch)

            # Store results (in a real implementation, you'd need proper result handling)
            for item, result in zip(batch, results):
                # This is simplified - in practice you'd need a better way to return results
                print(f"Processed item {item.id}: {result}")

        except Exception as e:
            print(f"Batch processing failed in worker {worker_id}: {e}")


class BatchingService:
    """High-level batching service that can use different backends."""

    def __init__(self, backend: str = "async", **kwargs):
        self.backend = backend

        if backend == "async":
            self.processor = AsyncBatchProcessor(**kwargs)
        elif backend == "threaded":
            self.processor = ThreadedBatchProcessor(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    async def start(self):
        """Start the service."""
        if hasattr(self.processor, 'start'):
            if asyncio.iscoroutinefunction(self.processor.start):
                await self.processor.start()
            else:
                self.processor.start()

    async def stop(self):
        """Stop the service."""
        if hasattr(self.processor, 'stop'):
            if asyncio.iscoroutinefunction(self.processor.stop):
                await self.processor.stop()
            else:
                self.processor.stop()

    async def submit(self, data: Any) -> Any:
        """Submit data for processing."""
        if hasattr(self.processor, 'submit'):
            if asyncio.iscoroutinefunction(self.processor.submit):
                return await self.processor.submit(data)
            else:
                return self.processor.submit(data)


# Example processors
async def database_batch_processor(items: List[BatchItem]) -> List[Any]:
    """Example processor for database operations."""
    print(f"Executing batch database operation for {len(items)} items")
    await asyncio.sleep(0.2)  # Simulate database I/O

    # Simulate processing each item
    results = []
    for item in items:
        # Simulate database operation
        result = f"DB_RESULT_{item.data}_{int(time.time())}"
        results.append(result)

    return results


def api_batch_processor(items: List[BatchItem]) -> List[Any]:
    """Example processor for API calls."""
    print(f"Making batch API call for {len(items)} items")
    time.sleep(0.3)  # Simulate API call

    # Simulate processing each item
    results = []
    for item in items:
        result = f"API_RESPONSE_{item.data}"
        results.append(result)

    return results


async def demonstrate_batching():
    """Demonstrate the batching workers."""

    print("=== Async Batch Processor Demo ===")

    # Create async batch processor
    async_processor = AsyncBatchProcessor(
        batch_size=3,
        max_wait_time=2.0,
        processor=database_batch_processor
    )

    await async_processor.start()

    # Submit some items
    tasks = []
    for i in range(8):
        task = asyncio.create_task(async_processor.submit(f"item_{i}"))
        tasks.append(task)
        await asyncio.sleep(0.1)  # Small delay between submissions

    # Wait for all results
    results = await asyncio.gather(*tasks)
    print(f"Async results: {results}")

    await async_processor.stop()

    print("\n=== Threaded Batch Processor Demo ===")

    # Create threaded batch processor
    threaded_processor = ThreadedBatchProcessor(
        batch_size=3,
        max_wait_time=2.0,
        processor=api_batch_processor,
        num_workers=2
    )

    threaded_processor.start()

    # Submit some items (this will block until processed)
    import threading
    results = []
    threads = []

    def submit_item(item_id):
        result = threaded_processor.submit(f"threaded_item_{item_id}")
        results.append(result)

    # Submit items in separate threads to avoid blocking
    for i in range(5):
        thread = threading.Thread(target=submit_item, args=(i,))
        thread.start()
        threads.append(thread)

    # Wait for all submissions
    for thread in threads:
        thread.join()

    threaded_processor.stop()

    print("\n=== High-level Service Demo ===")

    # Use the high-level service
    service = BatchingService(
        backend="async",
        batch_size=2,
        max_wait_time=1.0,
        processor=database_batch_processor
    )

    await service.start()

    # Submit items using the service
    service_tasks = []
    for i in range(4):
        task = asyncio.create_task(service.submit(f"service_item_{i}"))
        service_tasks.append(task)

    service_results = await asyncio.gather(*service_tasks)
    print(f"Service results: {service_results}")

    await service.stop()


if __name__ == "__main__":
    asyncio.run(demonstrate_batching())