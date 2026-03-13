import httpx
import asyncio
import json

async def test():
    async with httpx.AsyncClient() as client:
        url = 'http://127.0.0.1:8000/chat'
        messages = [{"role": "user", "content": "the password is urbox2026, go to urbox.ai/admin, and post a blog about urbox.ai vs missive"}]
        async with client.stream('POST', url, json={'messages': messages}, timeout=60) as response:
            async for chunk in response.aiter_text():
                print(f"Received chunk of max len {len(chunk)}\n")
                if 'image' in chunk:
                    print(f"FOUND IMAGE in chunk of length {len(chunk)}")
                    # don't print the whole base64
                    pass
                else:
                    print(chunk[:200])

asyncio.run(test())
