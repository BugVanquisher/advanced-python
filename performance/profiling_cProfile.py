"""
Profiling with cProfile
-----------------------
Simple profiling demo.
"""

import cProfile

def compute():
    return sum(i*i for i in range(10**6))

if __name__ == "__main__":
    cProfile.run("compute()")
