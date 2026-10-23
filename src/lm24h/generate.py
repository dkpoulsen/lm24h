"""Sample from a native checkpoint (naive no-KV-cache decoding; fine for smoke/eval prompts)."""
import argparse, torch
from tokenizers import Tokenizer
from lm24h.model import ModelConfig, Transformer

@torch.no_grad()
def sample(model, ids, max_new, temp, top_p, stop_ids, device):
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(x[:, -model.c.max_seq_len:])[0, -1].float()
        if temp <= 0: nxt = lg.argmax()
        else:
            p = torch.softmax(lg / temp, -1); sp, si = p.sort(descending=True); cs = sp.cumsum(0)
            sp[cs - sp > top_p] = 0; nxt = si[torch.multinomial(sp / sp.sum(), 1)][0]
        x = torch.cat([x, nxt.view(1, 1)], 1)
        if nxt.item() in stop_ids: break
    return x[0].tolist()[len(ids):]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--chat", action="store_true")
    ap.add_argument("--prompts", nargs="*"); ap.add_argument("--max_new", type=int, default=200); ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args(); dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = Transformer(ModelConfig(**ck["cfg"]["model"])); m.load_state_dict(ck["model"]); m = m.to(dev).eval()
    tok = Tokenizer.from_file("/lm24h/data/tokenizer/tokenizer.json"); eos = tok.token_to_id("<|endoftext|>"); end = tok.token_to_id("<|im_end|>")
    prompts = a.prompts or (["What is the capital of France?", "Explain what a transformer neural network is in two sentences.",
                             "Write a Python function that checks whether a number is prime.", "Give me three tips for learning a new language."] if a.chat else
                            ["The theory of evolution by natural selection", "Photosynthesis is the process by which", "def fibonacci(n):", "The French Revolution began in"])
    for p in prompts:
        txt = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n" if a.chat else p
        out = sample(m, tok.encode(txt, add_special_tokens=False).ids, a.max_new, a.temp, 0.9, {end if a.chat else eos}, dev)
        print(f"### {p}\n{tok.decode(out, skip_special_tokens=False)}\n", flush=True)
