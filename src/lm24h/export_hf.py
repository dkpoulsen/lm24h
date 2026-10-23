"""Export a lm24h checkpoint to a HF Qwen3ForCausalLM directory (safetensors + tokenizer) usable by vLLM / llama.cpp."""
import argparse, json, os, shutil, torch
from safetensors.torch import save_file
from lm24h.model import ModelConfig

def hf_config(c: ModelConfig, dtype="bfloat16"):
    return {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": c.vocab_size, "hidden_size": c.d_model,
            "intermediate_size": c.ffn_dim, "num_hidden_layers": c.n_layer, "num_attention_heads": c.n_head,
            "num_key_value_heads": c.n_kv_head, "head_dim": c.head_dim, "hidden_act": "silu", "max_position_embeddings": c.max_seq_len,
            "rms_norm_eps": c.norm_eps, "rope_theta": c.rope_theta, "tie_word_embeddings": c.tie_embeddings, "attention_bias": False,
            "use_sliding_window": False, "torch_dtype": dtype, "bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 3}

def convert_state(sd, c: ModelConfig, dtype=torch.bfloat16):
    out = {}
    for k, v in sd.items():
        if k.startswith("rope_"): continue
        if k == "lm_head.weight" and c.tie_embeddings: continue
        nk = "model." + k if not k.startswith("lm_head") else k
        out[nk] = v.detach().to(dtype).contiguous().cpu()
    return out

CHATML = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")

def export(ckpt_path, out_dir, tokenizer_dir="/lm24h/data/tokenizer", chat=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ModelConfig(**ck["cfg"]["model"]); os.makedirs(out_dir, exist_ok=True)
    save_file(convert_state(ck["model"], c), f"{out_dir}/model.safetensors", metadata={"format": "pt"})
    json.dump(hf_config(c), open(f"{out_dir}/config.json", "w"), indent=1)
    hc = hf_config(c); hc["eos_token_id"] = 2 if chat else 0; json.dump(hc, open(f"{out_dir}/config.json", "w"), indent=1)
    json.dump({"do_sample": True, "temperature": 0.7, "top_p": 0.9, "eos_token_id": 2 if chat else 0}, open(f"{out_dir}/generation_config.json", "w"))
    shutil.copy(f"{tokenizer_dir}/tokenizer.json", out_dir)
    tcfg = {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "<|im_end|>" if chat else "<|endoftext|>", "bos_token": None,
            "pad_token": "<|pad|>", "model_max_length": c.max_seq_len, "clean_up_tokenization_spaces": False}
    if chat: tcfg["chat_template"] = CHATML
    json.dump(tcfg, open(f"{out_dir}/tokenizer_config.json", "w"), indent=1)
    print("exported", out_dir, "step", ck.get("step"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--chat", action="store_true"); a = ap.parse_args(); export(a.ckpt, a.out, chat=a.chat)
