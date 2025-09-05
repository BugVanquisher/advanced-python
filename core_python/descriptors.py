"""
Descriptors Demo
----------------
Shows how Python descriptors work.
"""

class Descriptor:
    def __get__(self, instance, owner):
        return f"Accessed from {owner}"

class MyClass:
    attr = Descriptor()

if __name__ == "__main__":
    print(MyClass().attr)
