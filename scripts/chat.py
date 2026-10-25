#!/usr/bin/env python3
"""Interactive CLI chat with the local lm24h model via its vLLM endpoint."""
import sys, json, urllib.request

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "lm24h-chat"

def chat():
    print("=" * 60)
    print(" 🤖 lm24h-chat Interactive Terminal (trained from scratch)")
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
