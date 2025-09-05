"""
GIL Demo
--------
Illustrates Global Interpreter Lock behavior.
"""

import threading

x = 0

def increment():
    global x
    for _ in range(10**6):
        x += 1

if __name__ == "__main__":
    t1 = threading.Thread(target=increment)
    t2 = threading.Thread(target=increment)
    t1.start(); t2.start()
    t1.join(); t2.join()
    print("Final x:", x)
