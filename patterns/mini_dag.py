"""Mini DAG Framework"""

from typing import Dict, List, Set, Callable, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class Task:
    """A single task in the DAG."""

    def __init__(self, task_id: str, func: Callable, *args, **kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.dependencies: Set[str] = set()
        self.dependents: Set[str] = set()
        self.status = TaskStatus.PENDING
        self.result: Optional[TaskResult] = None

    def add_dependency(self, dependency_id: str) -> None:
        """Add a dependency to this task."""
        self.dependencies.add(dependency_id)

    def execute(self) -> TaskResult:
        """Execute the task."""
        self.status = TaskStatus.RUNNING
        result = TaskResult(
            task_id=self.task_id,
            status=TaskStatus.RUNNING,
            start_time=time.time()
        )

        try:
            print(f"Executing task: {self.task_id}")
            result.result = self.func(*self.args, **self.kwargs)
            result.status = TaskStatus.SUCCESS
            self.status = TaskStatus.SUCCESS
        except Exception as e:
            result.error = e
            result.status = TaskStatus.FAILED
            self.status = TaskStatus.FAILED
            print(f"Task {self.task_id} failed: {e}")
        finally:
            result.end_time = time.time()

        self.result = result
        return result


class DAG:
    """Directed Acyclic Graph for task execution."""

    def __init__(self, dag_id: str):
        self.dag_id = dag_id
        self.tasks: Dict[str, Task] = {}
        self._execution_results: Dict[str, TaskResult] = {}

    def add_task(self, task: Task) -> None:
        """Add a task to the DAG."""
        if task.task_id in self.tasks:
            raise ValueError(f"Task {task.task_id} already exists")
        self.tasks[task.task_id] = task

    def set_dependency(self, task_id: str, dependency_id: str) -> None:
        """Set a dependency between tasks."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        if dependency_id not in self.tasks:
            raise ValueError(f"Dependency {dependency_id} not found")

        self.tasks[task_id].add_dependency(dependency_id)
        self.tasks[dependency_id].dependents.add(task_id)

    def validate(self) -> bool:
        """Validate that the DAG has no cycles."""
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            for dependent_id in self.tasks[task_id].dependents:
                if dependent_id not in visited:
                    if has_cycle(dependent_id):
                        return True
                elif dependent_id in rec_stack:
                    return True

            rec_stack.remove(task_id)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    return False
        return True

    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []

        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.PENDING:
                dependencies_completed = all(
                    self.tasks[dep_id].status == TaskStatus.SUCCESS
                    for dep_id in task.dependencies
                )
                if dependencies_completed:
                    ready_tasks.append(task_id)

        return ready_tasks

    def execute_sequential(self) -> Dict[str, TaskResult]:
        """Execute the DAG sequentially."""
        if not self.validate():
            raise ValueError("DAG contains cycles")

        print(f"Starting sequential execution of DAG: {self.dag_id}")

        while True:
            ready_tasks = self.get_ready_tasks()
            if not ready_tasks:
                break

            for task_id in ready_tasks:
                task = self.tasks[task_id]
                result = task.execute()
                self._execution_results[task_id] = result

        return self._execution_results

    def execute_parallel(self, max_workers: int = 4) -> Dict[str, TaskResult]:
        """Execute the DAG in parallel where possible."""
        if not self.validate():
            raise ValueError("DAG contains cycles")

        print(f"Starting parallel execution of DAG: {self.dag_id} (max_workers={max_workers})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}

            while True:
                ready_tasks = self.get_ready_tasks()
                if not ready_tasks:
                    # Wait for running tasks if any
                    if future_to_task:
                        for future in as_completed(future_to_task):
                            task_id = future_to_task[future]
                            try:
                                result = future.result()
                                self._execution_results[task_id] = result
                            except Exception as e:
                                print(f"Task {task_id} failed with exception: {e}")
                            finally:
                                del future_to_task[future]
                    else:
                        break
                else:
                    # Submit ready tasks
                    for task_id in ready_tasks:
                        task = self.tasks[task_id]
                        future = executor.submit(task.execute)
                        future_to_task[future] = task_id

        return self._execution_results

    def get_execution_summary(self) -> dict:
        """Get a summary of the execution."""
        total_tasks = len(self.tasks)
        successful = sum(1 for r in self._execution_results.values() if r.status == TaskStatus.SUCCESS)
        failed = sum(1 for r in self._execution_results.values() if r.status == TaskStatus.FAILED)
        total_duration = sum(r.duration or 0 for r in self._execution_results.values())

        return {
            "dag_id": self.dag_id,
            "total_tasks": total_tasks,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_tasks if total_tasks > 0 else 0,
            "total_duration": total_duration
        }


# Helper functions for creating common task types
def create_function_task(task_id: str, func: Callable, *args, **kwargs) -> Task:
    """Create a task from a function."""
    return Task(task_id, func, *args, **kwargs)


def create_data_processing_task(task_id: str, data: Any, processor: Callable) -> Task:
    """Create a data processing task."""
    def process_data():
        return processor(data)
    return Task(task_id, process_data)


# Example tasks for demonstration
def load_data(source: str) -> dict:
    """Simulate loading data."""
    time.sleep(1)  # Simulate I/O
    return {"source": source, "data": [1, 2, 3, 4, 5]}


def transform_data(data: dict) -> dict:
    """Transform the data."""
    time.sleep(0.5)  # Simulate processing
    transformed = [x * 2 for x in data["data"]]
    return {"source": data["source"], "data": transformed, "transformed": True}


def validate_data(data: dict) -> dict:
    """Validate the data."""
    time.sleep(0.2)  # Simulate validation
    is_valid = all(isinstance(x, int) for x in data["data"])
    return {**data, "valid": is_valid}


def save_data(data: dict, destination: str) -> dict:
    """Save the data."""
    time.sleep(0.3)  # Simulate I/O
    return {"saved_to": destination, "record_count": len(data["data"])}


def demonstrate_dag():
    """Demonstrate the DAG framework."""

    print("=== DAG Framework Demo ===")

    # Create DAG
    dag = DAG("data_pipeline")

    # Create tasks
    load_task = create_function_task("load", load_data, "database")
    transform_task = create_function_task("transform", transform_data, None)  # Will get data from load_task
    validate_task = create_function_task("validate", validate_data, None)  # Will get data from transform_task
    save_task = create_function_task("save", save_data, None, "output.json")  # Will get data from validate_task

    # Modify tasks to use results from previous tasks
    def transform_with_dependency():
        load_result = dag._execution_results["load"].result
        return transform_data(load_result)

    def validate_with_dependency():
        transform_result = dag._execution_results["transform"].result
        return validate_data(transform_result)

    def save_with_dependency():
        validate_result = dag._execution_results["validate"].result
        return save_data(validate_result, "output.json")

    transform_task.func = transform_with_dependency
    validate_task.func = validate_with_dependency
    save_task.func = save_with_dependency

    # Add tasks to DAG
    dag.add_task(load_task)
    dag.add_task(transform_task)
    dag.add_task(validate_task)
    dag.add_task(save_task)

    # Set dependencies
    dag.set_dependency("transform", "load")
    dag.set_dependency("validate", "transform")
    dag.set_dependency("save", "validate")

    print(f"DAG validation: {dag.validate()}")

    # Execute sequentially
    print("\n--- Sequential Execution ---")
    start_time = time.time()
    results = dag.execute_sequential()
    sequential_time = time.time() - start_time

    print(f"Sequential execution completed in {sequential_time:.2f}s")
    for task_id, result in results.items():
        print(f"Task {task_id}: {result.status.value} ({result.duration:.2f}s)")

    # Reset for parallel execution
    for task in dag.tasks.values():
        task.status = TaskStatus.PENDING
        task.result = None
    dag._execution_results.clear()

    # Execute in parallel
    print("\n--- Parallel Execution ---")
    start_time = time.time()
    results = dag.execute_parallel(max_workers=2)
    parallel_time = time.time() - start_time

    print(f"Parallel execution completed in {parallel_time:.2f}s")
    for task_id, result in results.items():
        print(f"Task {task_id}: {result.status.value} ({result.duration:.2f}s)")

    # Show summary
    print("\n--- Execution Summary ---")
    summary = dag.get_execution_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demonstrate_dag()