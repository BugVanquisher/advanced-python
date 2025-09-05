"""
Pipeline Pattern
----------------
Chaining operations.
"""

class Pipeline:
    def __init__(self, funcs):
        self.funcs = funcs

    def run(self, value):
        for f in self.funcs:
            value = f(value)
        return value

if __name__ == "__main__":
    funcs = [lambda x: x+1, lambda x: x*2]
    p = Pipeline(funcs)
    print(p.run(3))  # -> 8
