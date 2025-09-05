"""
Multiprocessing vs Threading
----------------------------
Demonstrates difference in parallelism.
"""

import time
import threading
import multiprocessing

def worker(n):
    s = sum(i*i for i in range(10**6))
    print(f"Done {n}")

if __name__ == "__main__":
    start = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print("Threads took:", time.time()-start)

    start = time.time()
    procs = [multiprocessing.Process(target=worker, args=(i,)) for i in range(4)]
    [p.start() for p in procs]
    [p.join() for p in procs]
    print("Procs took:", time.time()-start)
