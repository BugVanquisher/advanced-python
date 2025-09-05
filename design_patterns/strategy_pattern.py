"""
Strategy Pattern
----------------
Different algorithms encapsulated.
"""

class Strategy:
    def execute(self, data):
        raise NotImplementedError

class StrategyA(Strategy):
    def execute(self, data):
        return f"A processed {data}"

class StrategyB(Strategy):
    def execute(self, data):
        return f"B processed {data}"

if __name__ == "__main__":
    s = StrategyA()
    print(s.execute("input"))
    s = StrategyB()
    print(s.execute("input"))
