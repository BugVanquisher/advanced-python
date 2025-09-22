"""Descriptors Demo"""


class ValidatedAttribute:
    """A descriptor that validates attribute values."""

    def __init__(self, validator=None, default=None):
        self.validator = validator
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        self.name = f'_{name}'

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.name, self.default)

    def __set__(self, obj, value):
        if self.validator and not self.validator(value):
            raise ValueError(f"Invalid value: {value}")
        setattr(obj, self.name, value)


class PositiveNumber(ValidatedAttribute):
    """Descriptor for positive numbers."""

    def __init__(self):
        super().__init__(validator=lambda x: isinstance(x, (int, float)) and x > 0)


class Person:
    """Example class using descriptors."""

    age = PositiveNumber()

    def __init__(self, name, age):
        self.name = name
        self.age = age


class LazyProperty:
    """A descriptor that computes a value once and caches it."""

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        if not hasattr(obj, f'_{self.name}'):
            setattr(obj, f'_{self.name}', self.func(obj))
        return getattr(obj, f'_{self.name}')


class Calculator:
    """Example class using lazy property."""

    def __init__(self, numbers):
        self.numbers = numbers

    @LazyProperty
    def expensive_calculation(self):
        print("Performing expensive calculation...")
        return sum(x**2 for x in self.numbers)


if __name__ == "__main__":
    # Test ValidatedAttribute
    person = Person("Alice", 25)
    print(f"{person.name} is {person.age} years old")

    try:
        person.age = -5  # This will raise ValueError
    except ValueError as e:
        print(f"Error: {e}")

    # Test LazyProperty
    calc = Calculator([1, 2, 3, 4, 5])
    print("First access:")
    print(calc.expensive_calculation)  # Calculation happens here
    print("Second access:")
    print(calc.expensive_calculation)  # Cached value returned