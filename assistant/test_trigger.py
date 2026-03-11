import httpx
import json

data = {
    "messages": [
        {"role": "user", "content": "search up urbox.ai and tell me what it is"}
    ]
}

with httpx.stream("POST", "http://127.0.0.1:8000/chat", json=data) as response:
    for line in response.iter_lines():
        if line:
            print(line)
