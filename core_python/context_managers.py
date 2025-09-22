"""Context Managers Demo"""

import contextlib
import time
import threading
from typing import Any


class Timer:
    """A context manager that times code execution."""

    def __init__(self, description="Code block"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.description}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed = end_time - self.start_time
        print(f"{self.description} took {elapsed:.4f} seconds")


class DatabaseConnection:
    """Mock database connection with context manager."""

    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connected = False

    def __enter__(self):
        print(f"Connecting to {self.connection_string}")
        self.connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        self.connected = False
        if exc_type:
            print(f"Exception occurred: {exc_val}")
            # Return False to propagate the exception

    def query(self, sql):
        if not self.connected:
            raise RuntimeError("Not connected to database")
        return f"Result of: {sql}"


@contextlib.contextmanager
def temporary_setting(obj, attr, value):
    """Context manager that temporarily changes an object's attribute."""
    old_value = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield old_value
    finally:
        setattr(obj, attr, old_value)


class FileManager:
    """File manager with automatic cleanup."""

    def __init__(self):
        self.files = []

    @contextlib.contextmanager
    def open_file(self, filename, mode='r'):
        """Context manager for file operations with tracking."""
        print(f"Opening {filename}")
        file_obj = open(filename, mode)
        self.files.append(file_obj)
        try:
            yield file_obj
        finally:
            print(f"Closing {filename}")
            file_obj.close()
            self.files.remove(file_obj)

    def cleanup(self):
        """Close all remaining open files."""
        for file_obj in self.files[:]:
            file_obj.close()
            self.files.remove(file_obj)


class ThreadSafeCounter:
    """Thread-safe counter using context manager."""

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def modify(self):
        """Context manager for thread-safe modifications."""
        with self._lock:
            yield self

    def increment(self):
        self._value += 1

    def decrement(self):
        self._value -= 1

    @property
    def value(self):
        return self._value


if __name__ == "__main__":
    # Timer example
    with Timer("Sleep operation"):
        time.sleep(1)

    # Database connection example
    with DatabaseConnection("postgresql://localhost/mydb") as db:
        result = db.query("SELECT * FROM users")
        print(result)

    # Temporary setting example
    class Config:
        debug = False

    config = Config()
    print(f"Debug mode: {config.debug}")

    with temporary_setting(config, 'debug', True) as old_value:
        print(f"Debug mode: {config.debug} (was {old_value})")

    print(f"Debug mode: {config.debug}")

    # File manager example (would work with actual files)
    fm = FileManager()
    try:
        # This would work with actual files
        # with fm.open_file('test.txt', 'w') as f:
        #     f.write("Hello, world!")
        print("File manager example (files would be managed automatically)")
    finally:
        fm.cleanup()

    # Thread-safe counter example
    counter = ThreadSafeCounter()
    with counter.modify():
        counter.increment()
        counter.increment()
    print(f"Counter value: {counter.value}")