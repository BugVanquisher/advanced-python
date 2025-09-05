"""
Debugging Tricks
----------------
Using breakpoint and pdb.
"""

def buggy():
    x = 1
    y = 0
    breakpoint()  # Enter debugger
    return x / y

if __name__ == "__main__":
    buggy()
