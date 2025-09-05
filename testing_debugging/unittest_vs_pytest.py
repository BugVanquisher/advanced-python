"""
Unittest vs Pytest
------------------
Shows basic comparison.
"""

import unittest

def add(a, b): return a + b

class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1,2), 3)

if __name__ == "__main__":
    unittest.main()
