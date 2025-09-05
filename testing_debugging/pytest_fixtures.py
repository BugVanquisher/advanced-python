"""
Pytest Fixtures Demo
--------------------
Simple pytest fixture usage.
"""

import pytest

@pytest.fixture
def data():
    return [1,2,3]

def test_sum(data):
    assert sum(data) == 6
