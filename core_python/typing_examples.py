"""Advanced Typing Demo"""

from typing import (
    Dict, List, Optional, Union, Callable, TypeVar, Generic,
    Protocol, Literal, Final, ClassVar, Any, Type, cast,
    overload, get_type_hints
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools


T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


@dataclass
class User:
    """User dataclass with type annotations."""
    id: int
    name: str
    email: Optional[str] = None
    is_active: bool = True
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class Drawable(Protocol):
    """Protocol defining drawable objects."""

    def draw(self) -> None:
        ...

    @property
    def area(self) -> float:
        ...


class Circle:
    """Circle class implementing Drawable protocol."""

    def __init__(self, radius: float):
        self.radius = radius

    def draw(self) -> None:
        print(f"Drawing circle with radius {self.radius}")

    @property
    def area(self) -> float:
        return 3.14159 * self.radius ** 2


class Rectangle:
    """Rectangle class implementing Drawable protocol."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def draw(self) -> None:
        print(f"Drawing rectangle {self.width}x{self.height}")

    @property
    def area(self) -> float:
        return self.width * self.height


def draw_shapes(shapes: List[Drawable]) -> None:
    """Function that works with any drawable object."""
    for shape in shapes:
        shape.draw()
        print(f"Area: {shape.area}")


class Container(Generic[T]):
    """Generic container class."""

    def __init__(self) -> None:
        self._items: List[T] = []

    def add(self, item: T) -> None:
        self._items.append(item)

    def get(self, index: int) -> T:
        return self._items[index]

    def all(self) -> List[T]:
        return self._items.copy()


class Cache(Generic[K, V]):
    """Generic cache with key-value pairs."""

    def __init__(self) -> None:
        self._data: Dict[K, V] = {}

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

    def get(self, key: K) -> Optional[V]:
        return self._data.get(key)


def process_data(data: Union[str, int, List[str]]) -> str:
    """Function with union types."""
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, int):
        return str(data * 2)
    else:  # List[str]
        return ", ".join(data)


@overload
def get_value(container: Dict[str, int], key: str) -> int:
    ...


@overload
def get_value(container: List[T], key: int) -> T:
    ...


def get_value(container, key):
    """Overloaded function with different return types."""
    return container[key]


class ConfigMeta(type):
    """Metaclass for configuration classes."""

    def __new__(cls, name, bases, namespace):
        # Add type checking for configuration values
        annotations = namespace.get('__annotations__', {})
        for attr_name, attr_type in annotations.items():
            if not attr_name.startswith('_'):
                namespace[f'_{attr_name}_type'] = attr_type
        return super().__new__(cls, name, bases, namespace)


class Config(metaclass=ConfigMeta):
    """Configuration class with type metadata."""

    DEBUG: ClassVar[bool] = False
    MAX_CONNECTIONS: Final[int] = 100
    database_url: str
    retry_count: int = 3


def typed_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that preserves type information."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


@typed_decorator
def calculate_area(radius: float) -> float:
    """Calculate circle area."""
    return 3.14159 * radius ** 2


# Literal types
Status = Literal["pending", "approved", "rejected"]


def process_request(status: Status) -> str:
    """Function using literal types."""
    if status == "pending":
        return "Request is being processed"
    elif status == "approved":
        return "Request has been approved"
    else:  # rejected
        return "Request has been rejected"


def demonstrate_type_checking() -> None:
    """Demonstrate various typing features."""

    # Basic types
    user = User(id=1, name="Alice", email="alice@example.com")
    print(f"User: {user}")

    # Protocol usage
    shapes: List[Drawable] = [Circle(5.0), Rectangle(3.0, 4.0)]
    draw_shapes(shapes)

    # Generics
    string_container: Container[str] = Container()
    string_container.add("hello")
    string_container.add("world")
    print(f"Strings: {string_container.all()}")

    int_container: Container[int] = Container()
    int_container.add(1)
    int_container.add(2)
    print(f"Numbers: {int_container.all()}")

    # Cache
    cache: Cache[str, int] = Cache()
    cache.set("count", 42)
    print(f"Cached value: {cache.get('count')}")

    # Union types
    print(process_data("hello"))
    print(process_data(21))
    print(process_data(["a", "b", "c"]))

    # Overloads
    my_dict = {"key": 42}
    my_list = [1, 2, 3]
    print(f"Dict value: {get_value(my_dict, 'key')}")
    print(f"List value: {get_value(my_list, 1)}")

    # Literal types
    print(process_request("pending"))
    print(process_request("approved"))

    # Decorated function
    area = calculate_area(5.0)
    print(f"Area: {area}")

    # Type hints introspection
    hints = get_type_hints(User)
    print(f"User type hints: {hints}")


if __name__ == "__main__":
    demonstrate_type_checking()