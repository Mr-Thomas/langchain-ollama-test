import ollama

stream = ollama.chat(
    model='qwen2:1.5b',
    messages=[
        {
            'role': 'user',
            'content': '你是谁?',
        },
    ], stream=True)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)


