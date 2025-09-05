"""
Typing & Generics Demo
----------------------
Using typing module for generic functions.
"""

from typing import TypeVar, List

T = TypeVar("T")

def first_element(lst: List[T]) -> T:
    return lst[0]

if __name__ == "__main__":
    print(first_element([1,2,3]))
    print(first_element(["a","b","c"]))
