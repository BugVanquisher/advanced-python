"""Structured Concurrency (Task Groups)"""

import asyncio
import time
import random
from typing import List, Any, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum


class TaskResult(Enum):
    """Task execution results."""
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a task with its result."""
    name: str
    result: Any = None
    status: TaskResult = TaskResult.SUCCESS
    duration: float = 0.0
    error: Optional[Exception] = None


class TaskGroup:
    """A simple task group implementation for structured concurrency."""

    def __init__(self, name: str = "TaskGroup"):
        self.name = name
        self.tasks: List[asyncio.Task] = []
        self.results: List[Task] = []
        self._cancelled = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.wait_all()
        if exc_type:
            await self.cancel_all()

    def create_task(self, coro, name: str = None) -> asyncio.Task:
        """Create and track a new task."""
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        self.tasks.append(task)
        return task

    async def wait_all(self) -> List[Task]:
        """Wait for all tasks to complete and return results."""
        if not self.tasks:
            return []

        print(f"Waiting for {len(self.tasks)} tasks in {self.name}...")

        for task in asyncio.as_completed(self.tasks):
            start_time = time.time()
            try:
                result = await task
                duration = time.time() - start_time
                task_name = getattr(task, '_name', 'unnamed')

                self.results.append(Task(
                    name=task_name,
                    result=result,
                    status=TaskResult.SUCCESS,
                    duration=duration
                ))
                print(f"✅ Task '{task_name}' completed in {duration:.2f}s")

            except asyncio.CancelledError:
                duration = time.time() - start_time
                task_name = getattr(task, '_name', 'unnamed')

                self.results.append(Task(
                    name=task_name,
                    status=TaskResult.CANCELLED,
                    duration=duration
                ))
                print(f"🚫 Task '{task_name}' was cancelled")

            except Exception as e:
                duration = time.time() - start_time
                task_name = getattr(task, '_name', 'unnamed')

                self.results.append(Task(
                    name=task_name,
                    status=TaskResult.FAILED,
                    duration=duration,
                    error=e
                ))
                print(f"❌ Task '{task_name}' failed: {e}")

        return self.results

    async def cancel_all(self):
        """Cancel all running tasks."""
        print(f"Cancelling all tasks in {self.name}...")
        self._cancelled = True

        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def get_summary(self) -> dict:
        """Get a summary of task execution."""
        if not self.results:
            return {"total": 0, "success": 0, "failed": 0, "cancelled": 0}

        total = len(self.results)
        success = sum(1 for r in self.results if r.status == TaskResult.SUCCESS)
        failed = sum(1 for r in self.results if r.status == TaskResult.FAILED)
        cancelled = sum(1 for r in self.results if r.status == TaskResult.CANCELLED)
        total_duration = sum(r.duration for r in self.results)

        return {
            "total": total,
            "success": success,
            "failed": failed,
            "cancelled": cancelled,
            "total_duration": total_duration,
            "average_duration": total_duration / total if total > 0 else 0
        }


# Modern Python 3.11+ TaskGroup (fallback implementation)
try:
    from asyncio import TaskGroup as AsyncIOTaskGroup
except ImportError:
    # For Python < 3.11, use our custom implementation
    AsyncIOTaskGroup = TaskGroup


@asynccontextmanager
async def timeout_task_group(timeout: float, name: str = "TimeoutTaskGroup"):
    """Task group with timeout support."""
    async with TaskGroup(name) as tg:
        try:
            yield tg
        except asyncio.TimeoutError:
            print(f"⏰ Task group '{name}' timed out after {timeout}s")
            await tg.cancel_all()
            raise


# Example tasks for demonstration
async def fetch_data(url: str, delay: float = 1.0) -> dict:
    """Simulate fetching data from a URL."""
    print(f"🌐 Fetching data from {url}...")
    await asyncio.sleep(delay + random.uniform(0, 0.5))

    # Simulate occasional failures
    if random.random() < 0.1:  # 10% failure rate
        raise ConnectionError(f"Failed to connect to {url}")

    return {
        "url": url,
        "data": f"Data from {url}",
        "timestamp": time.time()
    }


async def process_data(data: dict, processing_time: float = 0.5) -> dict:
    """Simulate processing data."""
    print(f"⚙️ Processing data from {data.get('url', 'unknown')}...")
    await asyncio.sleep(processing_time)

    return {
        "processed": True,
        "original": data,
        "result": f"Processed: {data.get('data', '')}"
    }


async def save_result(result: dict, delay: float = 0.3) -> str:
    """Simulate saving a result."""
    print(f"💾 Saving result...")
    await asyncio.sleep(delay)

    return f"Saved: {result.get('result', 'unknown')}"


async def failing_task(task_name: str) -> str:
    """A task that always fails."""
    await asyncio.sleep(0.5)
    raise ValueError(f"Task {task_name} intentionally failed")


async def slow_task(duration: float, task_name: str) -> str:
    """A task that takes a long time."""
    print(f"🐌 Starting slow task: {task_name}")
    await asyncio.sleep(duration)
    return f"Completed: {task_name}"


async def demonstrate_basic_task_groups():
    """Demonstrate basic task group usage."""
    print("\n=== Basic Task Groups ===")

    urls = [
        "https://api1.example.com",
        "https://api2.example.com",
        "https://api3.example.com"
    ]

    async with TaskGroup("DataFetchers") as tg:
        # Create multiple fetch tasks
        for i, url in enumerate(urls):
            tg.create_task(fetch_data(url, delay=0.5), name=f"fetch-{i+1}")

    summary = tg.get_summary()
    print(f"\nSummary: {summary}")

    return tg.results


async def demonstrate_error_handling():
    """Demonstrate error handling in task groups."""
    print("\n=== Error Handling ===")

    async with TaskGroup("MixedTasks") as tg:
        # Mix of successful and failing tasks
        tg.create_task(fetch_data("https://good-api.com", 0.3), "good-task")
        tg.create_task(failing_task("bad-task"), "bad-task")
        tg.create_task(fetch_data("https://another-api.com", 0.4), "another-good-task")

    summary = tg.get_summary()
    print(f"\nError handling summary: {summary}")

    # Show failed tasks
    failed_tasks = [r for r in tg.results if r.status == TaskResult.FAILED]
    for task in failed_tasks:
        print(f"Failed task '{task.name}': {task.error}")


async def demonstrate_pipeline():
    """Demonstrate a data processing pipeline with task groups."""
    print("\n=== Pipeline with Task Groups ===")

    # Stage 1: Fetch data
    async with TaskGroup("Stage1-Fetch") as fetch_group:
        urls = ["https://source1.com", "https://source2.com"]
        for i, url in enumerate(urls):
            fetch_group.create_task(fetch_data(url, 0.3), f"fetch-{i+1}")

    # Get successful fetch results
    fetch_results = [r.result for r in fetch_group.results
                    if r.status == TaskResult.SUCCESS]

    if not fetch_results:
        print("❌ No data to process")
        return

    # Stage 2: Process data
    async with TaskGroup("Stage2-Process") as process_group:
        for i, data in enumerate(fetch_results):
            process_group.create_task(process_data(data, 0.2), f"process-{i+1}")

    # Get successful process results
    process_results = [r.result for r in process_group.results
                      if r.status == TaskResult.SUCCESS]

    # Stage 3: Save results
    async with TaskGroup("Stage3-Save") as save_group:
        for i, result in enumerate(process_results):
            save_group.create_task(save_result(result, 0.1), f"save-{i+1}")

    print("\nPipeline completed!")
    print(f"Fetch: {fetch_group.get_summary()}")
    print(f"Process: {process_group.get_summary()}")
    print(f"Save: {save_group.get_summary()}")


async def demonstrate_cancellation():
    """Demonstrate task cancellation."""
    print("\n=== Task Cancellation ===")

    try:
        async with asyncio.timeout(2.0):  # 2 second timeout
            async with TaskGroup("SlowTasks") as tg:
                # Create tasks that would take too long
                tg.create_task(slow_task(1.0, "quick-task"), "quick")
                tg.create_task(slow_task(3.0, "slow-task-1"), "slow-1")
                tg.create_task(slow_task(4.0, "slow-task-2"), "slow-2")

    except asyncio.TimeoutError:
        print("⏰ Tasks were cancelled due to timeout")

    summary = tg.get_summary() if 'tg' in locals() else {}
    print(f"Cancellation summary: {summary}")


async def demonstrate_modern_task_groups():
    """Demonstrate modern Python 3.11+ TaskGroup if available."""
    print("\n=== Modern AsyncIO TaskGroup ===")

    try:
        # Use Python 3.11+ TaskGroup if available
        async with AsyncIOTaskGroup() as tg:
            task1 = tg.create_task(fetch_data("https://modern-api.com", 0.3))
            task2 = tg.create_task(process_data({"data": "test"}, 0.2))
            task3 = tg.create_task(save_result({"result": "test"}, 0.1))

        print(f"✅ Modern TaskGroup completed")
        print(f"Results: {[task1.result(), task2.result(), task3.result()]}")

    except Exception as e:
        print(f"Modern TaskGroup not available or failed: {e}")
        # Fallback to our custom implementation
        async with TaskGroup("FallbackGroup") as tg:
            tg.create_task(fetch_data("https://fallback-api.com", 0.3), "fallback-fetch")
            tg.create_task(process_data({"data": "fallback"}, 0.2), "fallback-process")

        print("✅ Fallback TaskGroup completed")


async def demonstrate_task_groups():
    """Main demonstration function."""
    print("🚀 Task Groups and Structured Concurrency Demo")

    # Run all demonstrations
    await demonstrate_basic_task_groups()
    await demonstrate_error_handling()
    await demonstrate_pipeline()
    await demonstrate_cancellation()
    await demonstrate_modern_task_groups()

    print("\n✨ All task group demonstrations completed!")


if __name__ == "__main__":
    asyncio.run(demonstrate_task_groups())