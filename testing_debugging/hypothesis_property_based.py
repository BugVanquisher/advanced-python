"""
Hypothesis Property-Based Testing
---------------------------------
Quick example with hypothesis.
"""

from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_commutative_addition(x, y):
    assert x + y == y + x
