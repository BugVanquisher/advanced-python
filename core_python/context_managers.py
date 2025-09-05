"""
Context Managers Demo
---------------------
Custom context manager using __enter__ and __exit__.
"""

class MyContext:
    def __enter__(self):
        print("Entering context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting context")

if __name__ == "__main__":
    with MyContext():
        print("Inside context")
