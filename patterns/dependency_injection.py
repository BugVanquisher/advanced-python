"""Dependency Injection"""

from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar, Callable, Any
import functools


T = TypeVar('T')


class DIContainer:
    """Simple dependency injection container."""

    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}

    def register(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a service implementation."""
        self._services[interface] = implementation

    def register_singleton(self, interface: Type[T], instance: T) -> None:
        """Register a singleton instance."""
        self._singletons[interface] = instance

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory function."""
        self._factories[interface] = factory

    def get(self, interface: Type[T]) -> T:
        """Resolve and return an instance of the requested interface."""
        # Check singletons first
        if interface in self._singletons:
            return self._singletons[interface]

        # Check factories
        if interface in self._factories:
            return self._factories[interface]()

        # Check registered services
        if interface in self._services:
            implementation = self._services[interface]
            return self._create_instance(implementation)

        raise ValueError(f"No service registered for {interface}")

    def _create_instance(self, cls: Type[T]) -> T:
        """Create an instance with dependency injection."""
        # Get constructor parameters
        import inspect
        signature = inspect.signature(cls.__init__)
        params = {}

        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue

            param_type = param.annotation
            if param_type != inspect.Parameter.empty:
                params[param_name] = self.get(param_type)

        return cls(**params)


def inject(container: DIContainer):
    """Decorator for automatic dependency injection."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            signature = inspect.signature(func)

            # Inject dependencies for missing parameters
            for param_name, param in signature.parameters.items():
                if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                    try:
                        kwargs[param_name] = container.get(param.annotation)
                    except ValueError:
                        pass  # Skip if service not available

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Example interfaces and implementations
class Logger(ABC):
    """Abstract logger interface."""

    @abstractmethod
    def log(self, message: str) -> None:
        pass


class ConsoleLogger(Logger):
    """Console logger implementation."""

    def log(self, message: str) -> None:
        print(f"[LOG] {message}")


class FileLogger(Logger):
    """File logger implementation."""

    def __init__(self, filename: str = "app.log"):
        self.filename = filename

    def log(self, message: str) -> None:
        print(f"[FILE LOG to {self.filename}] {message}")


class Database(ABC):
    """Abstract database interface."""

    @abstractmethod
    def query(self, sql: str) -> str:
        pass


class PostgreSQLDatabase(Database):
    """PostgreSQL database implementation."""

    def __init__(self, connection_string: str = "postgresql://localhost/db"):
        self.connection_string = connection_string

    def query(self, sql: str) -> str:
        return f"PostgreSQL query: {sql}"


class UserService:
    """User service with injected dependencies."""

    def __init__(self, logger: Logger, database: Database):
        self.logger = logger
        self.database = database

    def create_user(self, username: str) -> str:
        self.logger.log(f"Creating user: {username}")
        result = self.database.query(f"INSERT INTO users (username) VALUES ('{username}')")
        self.logger.log("User created successfully")
        return result

    def get_user(self, user_id: int) -> str:
        self.logger.log(f"Fetching user: {user_id}")
        return self.database.query(f"SELECT * FROM users WHERE id = {user_id}")


class EmailService:
    """Email service with injected logger."""

    def __init__(self, logger: Logger):
        self.logger = logger

    def send_email(self, to: str, subject: str, body: str) -> None:
        self.logger.log(f"Sending email to {to}: {subject}")
        print(f"Email sent to {to}")


def demonstrate_dependency_injection():
    """Demonstrate dependency injection patterns."""

    # Create container
    container = DIContainer()

    # Register services
    container.register(Logger, ConsoleLogger)
    container.register(Database, PostgreSQLDatabase)
    container.register(UserService, UserService)
    container.register(EmailService, EmailService)

    # Alternative: register singleton
    file_logger = FileLogger("custom.log")
    container.register_singleton(Logger, file_logger)

    print("=== Basic Dependency Injection ===")
    user_service = container.get(UserService)
    user_service.create_user("alice")
    user_service.get_user(1)

    print("\n=== Email Service ===")
    email_service = container.get(EmailService)
    email_service.send_email("alice@example.com", "Welcome", "Welcome to our service!")

    print("\n=== Factory Registration ===")
    # Register a factory for custom configuration
    container.register_factory(
        Database,
        lambda: PostgreSQLDatabase("postgresql://production/db")
    )

    # Now get a new database instance from factory
    db = container.get(Database)
    print(db.query("SELECT version()"))

    print("\n=== Decorator-based Injection ===")

    @inject(container)
    def process_user_data(user_id: int, logger: Logger, database: Database):
        logger.log(f"Processing user data for {user_id}")
        return database.query(f"SELECT * FROM user_data WHERE user_id = {user_id}")

    result = process_user_data(123)
    print(f"Result: {result}")


if __name__ == "__main__":
    demonstrate_dependency_injection()