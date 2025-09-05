"""
Decorators Demo
---------------
Shows simple decorator usage.
"""

def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logger
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("World"))
