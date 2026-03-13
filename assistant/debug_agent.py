import asyncio
from core.agent import agent_chat

async def main():
    messages = [{"role": "user", "content": "the password is urbox2026, go to urbox.ai/admin, and post a blog"}]
    print("Starting agent_chat...")
    async for event in agent_chat(messages):
        if event["type"] == "status":
            print(f"Status Event: {event['content']}")
            if "image" in event:
                print(f"  -> Has image! Length: {len(event['image'])}")
            else:
                print("  -> No image in event.")
        elif event["type"] == "content":
            print(f"Content Event: {event['content'][:100]}...")
            
if __name__ == "__main__":
    asyncio.run(main())
