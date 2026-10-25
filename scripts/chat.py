#!/usr/bin/env python3
"""Interactive CLI chat with the local lm24h model via its vLLM endpoint."""
import os, sys, json, urllib.request

# Port 8010 by default: port 8000 is taken by the ds41-flash serving stack on this box.
BASE_URL = os.environ.get("LM24H_BASE_URL", "http://127.0.0.1:8010")
API_URL = f"{BASE_URL}/v1/chat/completions"
MODEL = os.environ.get("LM24H_MODEL", "lm24h-chat")

def chat():
    print("=" * 60)
    print(f" 🤖 lm24h-chat Interactive Terminal (trained from scratch) @ {BASE_URL}")
    print(" Type your message and press Enter. Type 'exit' to quit.")
    print("=" * 60)
    history = []
    while True:
        try:
            prompt = input("\n\033[1;34mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not prompt or prompt.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        history.append({"role": "user", "content": prompt})
        payload = json.dumps({
            "model": MODEL,
            "messages": history,
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        }).encode("utf-8")

        req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["choices"][0]["message"]["content"]
                print(f"\n\033[1;32mlm24h:\033[0m {reply}")
                history.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"\n[Error querying model: {e}]")

if __name__ == "__main__":
    chat()
