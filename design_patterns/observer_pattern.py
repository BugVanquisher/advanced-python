"""
Observer Pattern
----------------
Classic observer implementation in Python.
"""

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def notify(self, message):
        for obs in self._observers:
            obs.update(message)

class Observer:
    def update(self, message):
        print(f"Received: {message}")

if __name__ == "__main__":
    subject = Subject()
    subject.attach(Observer())
    subject.notify("Event triggered!")
