"""
Memory Optimization Demo
------------------------
Using memoryview and numpy for efficient handling.
"""

import numpy as np

if __name__ == "__main__":
    arr = np.arange(10**6, dtype=np.int32)
    mv = memoryview(arr)
    print("Memoryview length:", len(mv))
