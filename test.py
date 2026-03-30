import asyncio
import websockets

async def test():
    async with websockets.connect("wss://so2ry8z3sw91zl-8765.proxy.runpod.net") as ws:
        print("✅ Connected!")

asyncio.run(test())