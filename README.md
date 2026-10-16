# Stroncius

Anonymous AI chat on Tor. No accounts, no server-side logs.

- Chat with small local LLMs (Llama 3.2 1B / SmolLM2 1.7B)
- Conversations live in the browser's localStorage only
- Rate limited per IP, mobile-first UI

## Stack

Python stdlib only. Proxies to local llama.cpp servers
(OpenAI-compatible `/v1/chat/completions`).

## Run

```
python3 app.py  # 127.0.0.1:18098
```

Needs a llama.cpp server on 127.0.0.1:18090 (llama) and/or
127.0.0.1:18091 (smollm2). Deploy as a Tor hidden service.
