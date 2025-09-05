"""
AsyncIO Examples
----------------
Basic async/await usage in Python.
"""

import asyncio

async def hello():
    await asyncio.sleep(1)
    print("Hello async world!")

if __name__ == "__main__":
    asyncio.run(hello())
