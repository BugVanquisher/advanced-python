"""
Metaclasses Demo
----------------
Example showing how to create and use metaclasses in Python.
"""

class MetaLogger(type):
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name} with MetaLogger")
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=MetaLogger):
    pass
