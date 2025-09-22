# 🕸️ Concurrency in Python

This guide explains Python concurrency in a clear and practical way, with runnable examples.  
It covers **threads, processes, asyncio, and high-level abstractions**.

---

## 1. What is Concurrency?
- **Concurrency**: Many tasks in progress (not necessarily at the same time).  
- **Parallelism**: Tasks actually running simultaneously (needs multiple CPU cores).  

In Python:
- **Threads** → Concurrency, limited by the Global Interpreter Lock (GIL).  
- **Processes** → True parallelism (each process has its own interpreter).  
- **AsyncIO** → Cooperative multitasking (non-blocking I/O).  

---

## 2. Threads (`threading`)
Good for I/O-bound tasks (waiting on files, APIs).  

```python
import threading, time

def worker(n):
    print(f"Start {n}")
    time.sleep(2)  # Simulate I/O
    print(f"Done {n}")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()
```

---

3. Processes (multiprocessing)

Bypasses the GIL → good for CPU-heavy tasks.
```Python
from multiprocessing import Process
import os

def compute(n):
    print(f"Process {n} (PID {os.getpid()})")
    total = sum(range(10_000_000))
    print(f"Done {n} total={total}")

procs = [Process(target=compute, args=(i,)) for i in range(3)]
for p in procs: p.start()
for p in procs: p.join()
```

---

4. AsyncIO (asyncio)

Perfect for high-concurrency I/O tasks (e.g., many network calls).
```Python
import asyncio, random

async def worker(n):
    print(f"Start {n}")
    await asyncio.sleep(random.uniform(1, 3))  # non-blocking
    print(f"Done {n}")

async def main():
    tasks = [asyncio.create_task(worker(i)) for i in range(3)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

---

5. Unified Abstraction – concurrent.futures

Swap easily between threads and processes.
```Python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * n

with ThreadPoolExecutor(max_workers=3) as executor:
    print("Thread results:", list(executor.map(task, range(5))))

with ProcessPoolExecutor(max_workers=3) as executor:
    print("Process results:", list(executor.map(task, range(5))))
```

---

6. When to Use What
	•	I/O-bound tasks → asyncio or threading
	•	CPU-bound tasks → multiprocessing
	•	Mixed workloads → hybrid approach

---

7. Mental Model
	•	Threading → Many workers in one kitchen, but only one cooks at a time.
	•	Multiprocessing → Many kitchens, each with its own cook.
	•	AsyncIO → One cook, but excellent at switching dishes while waiting.

---

✅ With these tools:
	•	Speed up I/O-heavy apps with threads or asyncio.
	•	Scale CPU-heavy apps with multiprocessing.
	•	Mix them for real-world systems.
