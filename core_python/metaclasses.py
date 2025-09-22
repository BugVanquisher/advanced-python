"""Metaclasses Demo"""


class SingletonMeta(type):
    """Metaclass that creates singleton instances."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Database(metaclass=SingletonMeta):
    """Database connection using singleton pattern."""

    def __init__(self):
        self.connection = "Database connection established"

    def query(self, sql):
        return f"Executing: {sql}"


class ValidatedMeta(type):
    """Metaclass that adds validation to class attributes."""

    def __new__(cls, name, bases, namespace):
        # Add validation for attributes that start with 'validate_'
        for attr_name, attr_value in namespace.items():
            if attr_name.startswith('validate_') and callable(attr_value):
                field_name = attr_name.replace('validate_', '')
                namespace[f'_validated_{field_name}'] = None

        # Create the class
        new_class = super().__new__(cls, name, bases, namespace)

        # Add property for each validated field
        for attr_name in namespace:
            if attr_name.startswith('validate_'):
                field_name = attr_name.replace('validate_', '')
                cls._add_validated_property(new_class, field_name, namespace[attr_name])

        return new_class

    @staticmethod
    def _add_validated_property(cls, field_name, validator):
        """Add a validated property to the class."""

        def getter(self):
            return getattr(self, f'_validated_{field_name}')

        def setter(self, value):
            if not validator(self, value):
                raise ValueError(f"Invalid value for {field_name}: {value}")
            setattr(self, f'_validated_{field_name}', value)

        setattr(cls, field_name, property(getter, setter))


class Person(metaclass=ValidatedMeta):
    """Person class with validated attributes."""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def validate_name(self, value):
        return isinstance(value, str) and len(value) > 0

    def validate_age(self, value):
        return isinstance(value, int) and 0 <= value <= 150


class AutoPropertyMeta(type):
    """Metaclass that automatically creates properties for private attributes."""

    def __new__(cls, name, bases, namespace):
        # Find private attributes (those starting with underscore)
        private_attrs = [attr for attr in namespace if attr.startswith('_') and not attr.startswith('__')]

        for attr in private_attrs:
            public_name = attr[1:]  # Remove leading underscore
            if public_name not in namespace:  # Don't override existing attributes
                cls._add_property(namespace, public_name, attr)

        return super().__new__(cls, name, bases, namespace)

    @staticmethod
    def _add_property(namespace, public_name, private_name):
        """Add a property that wraps access to a private attribute."""

        def getter(self):
            return getattr(self, private_name)

        def setter(self, value):
            setattr(self, private_name, value)

        namespace[public_name] = property(getter, setter)


class Account(metaclass=AutoPropertyMeta):
    """Account class with auto-generated properties."""

    def __init__(self, balance, owner):
        self._balance = balance
        self._owner = owner

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount

    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount


class RegistryMeta(type):
    """Metaclass that maintains a registry of all created classes."""

    registry = {}

    def __new__(cls, name, bases, namespace):
        new_class = super().__new__(cls, name, bases, namespace)
        cls.registry[name] = new_class
        return new_class

    @classmethod
    def get_class(cls, name):
        return cls.registry.get(name)

    @classmethod
    def list_classes(cls):
        return list(cls.registry.keys())


class Animal(metaclass=RegistryMeta):
    """Base animal class."""

    def __init__(self, name):
        self.name = name


class Dog(Animal):
    """Dog class."""

    def bark(self):
        return f"{self.name} says woof!"


class Cat(Animal):
    """Cat class."""

    def meow(self):
        return f"{self.name} says meow!"


class AttributeTrackerMeta(type):
    """Metaclass that tracks attribute access."""

    def __new__(cls, name, bases, namespace):
        new_class = super().__new__(cls, name, bases, namespace)

        # Wrap __getattribute__ to track access
        original_getattribute = new_class.__getattribute__

        def tracked_getattribute(self, name):
            if not name.startswith('_'):
                if not hasattr(self, '_access_log'):
                    object.__setattr__(self, '_access_log', [])
                self._access_log.append(name)
            return original_getattribute(self, name)

        new_class.__getattribute__ = tracked_getattribute
        return new_class


class TrackedClass(metaclass=AttributeTrackerMeta):
    """Class that tracks attribute access."""

    def __init__(self):
        self.value1 = "Hello"
        self.value2 = "World"

    def get_access_log(self):
        return getattr(self, '_access_log', [])


def demonstrate_metaclasses():
    """Demonstrate various metaclass patterns."""

    print("=== Singleton Pattern ===")
    db1 = Database()
    db2 = Database()
    print(f"Same instance: {db1 is db2}")
    print(db1.query("SELECT * FROM users"))

    print("\n=== Validated Attributes ===")
    person = Person("Alice", 25)
    print(f"Person: {person.name}, age {person.age}")

    try:
        person.age = -5  # This will raise ValueError
    except ValueError as e:
        print(f"Validation error: {e}")

    print("\n=== Auto Properties ===")
    account = Account(1000, "John")
    print(f"Balance: {account.balance}, Owner: {account.owner}")
    account.deposit(500)
    print(f"After deposit: {account.balance}")

    print("\n=== Class Registry ===")
    dog = Dog("Buddy")
    cat = Cat("Whiskers")

    print(f"Registered classes: {RegistryMeta.list_classes()}")
    dog_class = RegistryMeta.get_class("Dog")
    print(f"Retrieved Dog class: {dog_class}")

    print("\n=== Attribute Tracking ===")
    tracked = TrackedClass()
    print(f"Value1: {tracked.value1}")
    print(f"Value2: {tracked.value2}")
    print(f"Access log: {tracked.get_access_log()}")


if __name__ == "__main__":
    demonstrate_metaclasses()