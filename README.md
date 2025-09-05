# 📘 Advanced Python Learning & Practice

## 🔥 Purpose

This repository is my personal playground for learning, practicing, and refreshing advanced Python skills.
It’s structured to cover deeper parts of the language, performance tuning, concurrency, internals, and design patterns — with plenty of runnable examples and notes.

---

## 🗂 Repository Structure
```
advanced-python/
│
├── core_python/         # Advanced language features (metaclasses, descriptors, etc.)
├── performance/         # Profiling, optimization, async & parallelism
├── design_patterns/     # Implementations of common & Pythonic patterns
├── internals/           # Bytecode, GIL, garbage collection, CPython internals
├── testing_debugging/   # Property-based testing, pytest, debugging techniques
└── notebooks/           # Jupyter explorations for interactive learning
```

---

## 🚀 Learning Roadmap

Here’s the suggested progression:

1. Core Python
* Iterators, generators, coroutines
* Descriptors, __slots__, data classes
* Metaclasses & class creation
* Type hints, generics, typing & mypy
2.	Concurrency & Parallelism
* asyncio, concurrent.futures, multiprocessing
* Event loops, tasks, async/await patterns
* Building an async mini-project (e.g., scraper or chat server)
3.	Performance & Memory
* Profiling (cProfile, line_profiler)
* Memory optimization (gc, objgraph, memory_profiler)
* Vectorization with NumPy, Cython / Numba basics
4.	Internals Deep Dive
* Python bytecode (dis)
* GIL effects & workarounds
* Import system customization
* CPython internals overview
5.	Testing & Debugging
* Unit vs property-based testing (hypothesis)
* Pytest fixtures and parametrization
* Debugging tricks (pdb, ipdb, breakpoint())
6.	Design Patterns in Python
* Strategy, Observer, Singleton, Pipeline
* Functional patterns (higher-order funcs, decorators)
* Plug-in architectures
7.	Mini Projects
* Async web scraper
* Simple ORM with descriptors & metaclasses
* Task scheduler
* Tiny DSL parser

---

## 📚 References & Resources
### Books
* Fluent Python by Luciano Ramalho
* Effective Python by Brett Slatkin
* Python Cookbook by David Beazley & Brian K. Jones
### Talks & Articles
* David Beazley’s PyCon talks (coroutines, GIL, generators)

* Raymond Hettinger’s talks on Python internals & patterns

* CPython source code (for internals exploration)
### Tools
* pytest
* hypothesis
* objgraph

---

## 🏁 Usage

Clone the repo and explore by running examples:
```
git clone https://github.com/<your-username>/advanced-python.git
cd advanced-python
python core_python/metaclasses.py
```
Or launch the Jupyter notebooks:
```
jupyter notebook notebooks/
```

---

✨ The goal isn’t just to collect snippets, but to build a living reference library that grows as my Python skills grow.