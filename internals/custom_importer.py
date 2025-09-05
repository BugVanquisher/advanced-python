"""
Custom Importer
---------------
Hooks into import system.
"""

import sys

class MyImporter:
    def find_module(self, fullname, path=None):
        print(f"Importing {fullname}")
        return None

if __name__ == "__main__":
    sys.meta_path.insert(0, MyImporter())
    import math
