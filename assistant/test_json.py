import json

test_str = '{"type": "status", "content": "_Browsing to: **https://urbox.ai/admin**_ (step 1/20)", "image": "' + 'A' * 10000 + '"}'

# simulate chunked
chunk1 = test_str[:5000]
chunk2 = test_str[5000:]

# simulate decoder
buffer = chunk1
lines = buffer.split('\n')
buffer = lines.pop()
print("lines after chunk1:", lines)
print("buffer after chunk1:", len(buffer))

buffer += chunk2
lines = buffer.split('\n')
buffer = lines.pop()
print("lines after chunk2:", len(lines))
if lines:
    try:
        json.loads(lines[0])
        print("SUCCESSFULLY PARSED")
    except Exception as e:
        print("ERROR:", e)
