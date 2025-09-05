"""
Bytecode Disassembly
--------------------
Uses Python's `dis` module to inspect bytecode.
"""

import dis

def add(a, b):
    return a + b

if __name__ == "__main__":
    dis.dis(add)
