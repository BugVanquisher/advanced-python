"""AsyncIO Patterns"""

import asyncio
import aiohttp
import aiofiles
from typing import AsyncGenerator, List, Optional, Any, Callable, Dict
from dataclasses import dataclass
from enum import Enum
import random
import time
import json
from contextlib import asynccontextmanager
import weakref
from collections import deque


class TaskStatus(Enum):
    """Status of async tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Message:
    """Message for producer-consumer pattern."""
    id: str
    data: Any
    timestamp: float
    priority: int = 0

    def __lt__(self, other):
        return self.priority < other.priority


class AsyncProducerConsumer:
    """Producer-Consumer pattern with asyncio."""

    def __init__(self, max_queue_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.consumers = []
        self.producers = []
        self.running = False
        self.stats = {
            'produced': 0,
            'consumed': 0,
            'errors': 0
        }

    async def producer(self, producer_id: str, produce_func: Callable):
        """Generic producer coroutine."""
        print(f"Producer {producer_id} started")
        try:
            while self.running:
                try:
                    message = await produce_func(producer_id)
                    if message:
                        await self.queue.put(message)
                        self.stats['produced'] += 1
                        print(f"Producer {producer_id} produced: {message.id}")
                    else:
                        await asyncio.sleep(0.1)  # Brief pause if no data
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Producer {producer_id} error: {e}")
                    self.stats['errors'] += 1
                    await asyncio.sleep(1)  # Error backoff
        finally:
            print(f"Producer {producer_id} stopped")

    async def consumer(self, consumer_id: str, consume_func: Callable):
        """Generic consumer coroutine."""
        print(f"Consumer {consumer_id} started")
        try:
            while self.running:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    await consume_func(consumer_id, message)
                    self.stats['consumed'] += 1
                    print(f"Consumer {consumer_id} consumed: {message.id}")
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    continue  # No message available, keep checking
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Consumer {consumer_id} error: {e}")
                    self.stats['errors'] += 1
                    await asyncio.sleep(1)  # Error backoff
        finally:
            print(f"Consumer {consumer_id} stopped")

    async def start(self, num_producers: int = 2, num_consumers: int = 3,
                   produce_func: Callable = None, consume_func: Callable = None):
        """Start producers and consumers."""
        self.running = True

        # Default functions if not provided
        if produce_func is None:
            produce_func = self._default_producer
        if consume_func is None:
            consume_func = self._default_consumer

        # Start producers
        for i in range(num_producers):
            task = asyncio.create_task(self.producer(f"P{i+1}", produce_func))
            self.producers.append(task)

        # Start consumers
        for i in range(num_consumers):
            task = asyncio.create_task(self.consumer(f"C{i+1}", consume_func))
            self.consumers.append(task)

        print(f"Started {num_producers} producers and {num_consumers} consumers")

    async def stop(self):
        """Stop all producers and consumers."""
        print("Stopping producer-consumer system...")
        self.running = False

        # Cancel all tasks
        all_tasks = self.producers + self.consumers
        for task in all_tasks:
            task.cancel()

        # Wait for cancellation
        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Clear task lists
        self.producers.clear()
        self.consumers.clear()

        print(f"System stopped. Stats: {self.stats}")

    async def _default_producer(self, producer_id: str) -> Optional[Message]:
        """Default producer that generates random messages."""
        await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate work
        return Message(
            id=f"{producer_id}-{time.time()}",
            data=f"Data from {producer_id}",
            timestamp=time.time(),
            priority=random.randint(1, 5)
        )

    async def _default_consumer(self, consumer_id: str, message: Message):
        """Default consumer that processes messages."""
        await asyncio.sleep(random.uniform(0.1, 1.0))  # Simulate processing
        print(f"  {consumer_id} processed {message.data}")


class AsyncSemaphorePool:
    """Async semaphore pool for rate limiting."""

    def __init__(self, max_concurrent: int):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = set()
        self.completed_tasks = 0

    @asynccontextmanager
    async def acquire(self, task_name: str = None):
        """Acquire semaphore with tracking."""
        async with self.semaphore:
            task_name = task_name or f"task-{len(self.active_tasks)}"
            self.active_tasks.add(task_name)
            try:
                print(f"Task {task_name} acquired slot ({len(self.active_tasks)} active)")
                yield task_name
            finally:
                self.active_tasks.remove(task_name)
                self.completed_tasks += 1
                print(f"Task {task_name} released slot ({len(self.active_tasks)} active)")

    async def run_limited(self, tasks: List[Callable], task_names: List[str] = None):
        """Run tasks with semaphore limiting."""
        if task_names is None:
            task_names = [f"task-{i}" for i in range(len(tasks))]

        async def run_task(task_func, name):
            async with self.acquire(name):
                return await task_func()

        # Execute all tasks concurrently but limited by semaphore
        results = await asyncio.gather(
            *[run_task(task, name) for task, name in zip(tasks, task_names)],
            return_exceptions=True
        )

        return results


class AsyncCircuitBreaker:
    """Circuit breaker pattern for async operations."""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        if self.state == "open":
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "half-open"
                print("Circuit breaker is HALF-OPEN")

        try:
            result = await func(*args, **kwargs)
            # Success - reset if we were half-open
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                print("Circuit breaker is CLOSED (recovered)")
            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print(f"Circuit breaker is OPEN (failures: {self.failure_count})")

            raise e


class AsyncRetry:
    """Async retry pattern with exponential backoff."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)
                if attempt > 1:
                    print(f"Success on attempt {attempt}")
                return result

            except Exception as e:
                last_exception = e
                if attempt == self.max_attempts:
                    break

                delay = min(
                    self.base_delay * (self.backoff_factor ** (attempt - 1)),
                    self.max_delay
                )

                print(f"Attempt {attempt} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        raise last_exception


class AsyncEventBus:
    """Simple async event bus for decoupled communication."""

    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.event_history = deque(maxlen=100)

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(handler)
        print(f"Subscribed handler to {event_type}")

    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type."""
        if event_type in self.listeners:
            try:
                self.listeners[event_type].remove(handler)
                print(f"Unsubscribed handler from {event_type}")
            except ValueError:
                pass

    async def publish(self, event_type: str, data: Any = None):
        """Publish an event to all subscribers."""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        self.event_history.append(event)

        if event_type in self.listeners:
            tasks = []
            for handler in self.listeners[event_type]:
                task = asyncio.create_task(self._safe_call_handler(handler, event))
                tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        print(f"Published event: {event_type}")

    async def _safe_call_handler(self, handler: Callable, event: dict):
        """Safely call event handler."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            print(f"Event handler error: {e}")


# Async generators and streaming
async def async_data_stream(count: int = 10) -> AsyncGenerator[dict, None]:
    """Generate async data stream."""
    for i in range(count):
        await asyncio.sleep(0.5)  # Simulate data arrival
        yield {
            'id': i,
            'data': f'stream_data_{i}',
            'timestamp': time.time()
        }


async def process_stream(stream: AsyncGenerator, batch_size: int = 3):
    """Process async stream in batches."""
    batch = []
    async for item in stream:
        batch.append(item)
        if len(batch) >= batch_size:
            await process_batch(batch)
            batch = []

    # Process remaining items
    if batch:
        await process_batch(batch)


async def process_batch(batch: List[dict]):
    """Process a batch of items."""
    print(f"Processing batch of {len(batch)} items")
    await asyncio.sleep(0.2)  # Simulate processing
    for item in batch:
        print(f"  Processed: {item['data']}")


# Example async functions for demonstrations
async def fetch_url(url: str, session_pool) -> dict:
    """Simulate fetching a URL."""
    async with session_pool.acquire(f"fetch-{url}") as task_name:
        await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate network delay

        # Simulate occasional failures
        if random.random() < 0.2:  # 20% failure rate
            raise aiohttp.ClientError(f"Failed to fetch {url}")

        return {
            'url': url,
            'status': 200,
            'data': f'Content from {url}',
            'timestamp': time.time()
        }


async def unreliable_service() -> str:
    """Simulate an unreliable service."""
    await asyncio.sleep(0.5)
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("Service unavailable")
    return "Service response"


# Demonstration functions
async def demonstrate_producer_consumer():
    """Demonstrate producer-consumer pattern."""
    print("\n=== Producer-Consumer Pattern ===")

    pc = AsyncProducerConsumer(max_queue_size=10)

    # Custom producer that generates work items
    async def work_producer(producer_id: str) -> Optional[Message]:
        await asyncio.sleep(random.uniform(0.3, 1.0))
        work_type = random.choice(['email', 'report', 'backup', 'cleanup'])
        return Message(
            id=f"{producer_id}-{work_type}-{time.time()}",
            data={'type': work_type, 'size': random.randint(100, 1000)},
            timestamp=time.time(),
            priority=random.randint(1, 3)
        )

    # Custom consumer that processes work
    async def work_consumer(consumer_id: str, message: Message):
        work_data = message.data
        processing_time = work_data['size'] / 1000  # Simulate processing time
        await asyncio.sleep(processing_time)
        print(f"  {consumer_id} completed {work_data['type']} (size: {work_data['size']})")

    await pc.start(num_producers=2, num_consumers=3,
                   produce_func=work_producer, consume_func=work_consumer)

    # Let it run for a while
    await asyncio.sleep(5)
    await pc.stop()


async def demonstrate_semaphore_pool():
    """Demonstrate semaphore pool for rate limiting."""
    print("\n=== Semaphore Pool (Rate Limiting) ===")

    pool = AsyncSemaphorePool(max_concurrent=3)

    # Create multiple tasks that simulate API calls
    async def api_call(call_id: int) -> str:
        await asyncio.sleep(random.uniform(1, 3))  # Simulate API delay
        return f"API call {call_id} completed"

    tasks = [lambda i=i: api_call(i) for i in range(8)]
    task_names = [f"api-call-{i}" for i in range(8)]

    start_time = time.time()
    results = await pool.run_limited(tasks, task_names)
    duration = time.time() - start_time

    print(f"\nCompleted {len(results)} tasks in {duration:.1f}s")
    print(f"Successful results: {len([r for r in results if not isinstance(r, Exception)])}")


async def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n=== Circuit Breaker Pattern ===")

    cb = AsyncCircuitBreaker(failure_threshold=3, timeout=5.0)

    # Test the circuit breaker
    for i in range(10):
        try:
            result = await cb.call(unreliable_service)
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")

        await asyncio.sleep(0.5)


async def demonstrate_retry_pattern():
    """Demonstrate async retry pattern."""
    print("\n=== Retry Pattern ===")

    retry = AsyncRetry(max_attempts=4, base_delay=0.5, backoff_factor=2.0)

    try:
        result = await retry.execute(unreliable_service)
        print(f"Retry succeeded: {result}")
    except Exception as e:
        print(f"Retry ultimately failed: {e}")


async def demonstrate_event_bus():
    """Demonstrate event bus pattern."""
    print("\n=== Event Bus Pattern ===")

    event_bus = AsyncEventBus()

    # Define event handlers
    async def user_created_handler(event):
        print(f"Handler 1: User created - {event['data']['username']}")
        await asyncio.sleep(0.1)  # Simulate work

    async def send_welcome_email(event):
        print(f"Handler 2: Sending welcome email to {event['data']['username']}")
        await asyncio.sleep(0.2)  # Simulate email sending

    def log_event(event):
        print(f"Logger: Event {event['type']} at {event['timestamp']}")

    # Subscribe handlers
    event_bus.subscribe('user.created', user_created_handler)
    event_bus.subscribe('user.created', send_welcome_email)
    event_bus.subscribe('user.created', log_event)

    # Publish events
    await event_bus.publish('user.created', {'username': 'alice', 'email': 'alice@example.com'})
    await event_bus.publish('user.created', {'username': 'bob', 'email': 'bob@example.com'})

    # Give handlers time to complete
    await asyncio.sleep(1)


async def demonstrate_async_streams():
    """Demonstrate async generators and streaming."""
    print("\n=== Async Streams and Generators ===")

    # Process data stream in batches
    stream = async_data_stream(count=10)
    await process_stream(stream, batch_size=3)


async def demonstrate_asyncio_patterns():
    """Main demonstration function."""
    print("🚀 AsyncIO Patterns Demo")

    await demonstrate_producer_consumer()
    await demonstrate_semaphore_pool()
    await demonstrate_circuit_breaker()
    await demonstrate_retry_pattern()
    await demonstrate_event_bus()
    await demonstrate_async_streams()

    print("\n✨ All AsyncIO pattern demonstrations completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_asyncio_patterns())