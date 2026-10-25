# lm24h: 500M-Parameter Modern Chat Transformer in 24h

A modern production pretraining and instruction-tuning setup designed to train a ~500M parameter decoder-only transformer from scratch in ~21 hours on 5× NVIDIA CMP 170HX mining GPUs (64 GiB HBM2e each).

Exports 1:1 to Hugging Face `Qwen3ForCausalLM` and boots natively in upstream **vLLM** and **llama.cpp**.

📖 **Read the full scientific and engineering report in [FINDINGS.md](FINDINGS.md).**

---

## Model Specifications (`lm24h-500m`)

| Architecture | Qwen3-style Causal Transformer (GQA + QK-norm + SwiGLU + RoPE + RMSNorm) |
|---|---|
| **Parameters** | 494.2M (tied embeddings) |
| **Context Length** | 2,048 tokens |
| **Hidden Size ($d_{model}$)** | 1,280 |
| **Layers** | 24 |
| **Attention Heads** | 20 query heads / 5 KV heads (4:1 GQA, head dim 64) |
| **Intermediate Size** | 3,840 (SwiGLU) |
| **Vocabulary** | 32,768 (custom byte-level BPE trained on FineWeb-Edu) |
| **RoPE Theta ($\theta$)** | 100,000.0 |
| **QK Normalization** | RMSNorm on Q and K heads prior to RoPE |

---

## Training Pipeline Overview

### Phase 1: Pretraining
* **Data:** 9.04 Billion tokens from `HuggingFaceFW/fineweb-edu` (sample-10BT)
* **Optimization:** Fused AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.1)
* **Schedule:** Warmup-Stable-Decay (WSD): 400-step linear warmup to $8 \times 10^{-4}$, flat for 80%, linear cooldown over last 20% to $4 \times 10^{-5}$
* **Throughput:** ~136,000–146,000 tokens/sec across 5 GPUs (91–98 TFLOPS/GPU, ~55–60% MFU)
* **Pretrain Time:** ~17.5 hours
* **Final Validation Loss:** **2.6805** (Perplexity: **14.59**)

### Phase 2: Supervised Fine-Tuning (SFT)
* **Data:** 579,028 dialogues from `HuggingFaceTB/smoltalk` (1.19B total tokens, 659M assistant tokens)
* **Format:** ChatML (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`)
* **Masking:** Assistant-only loss mask ($y=-100$ on prompt tokens)
* **SFT Time:** ~2.5 hours (1 epoch)
* **Final Validation Loss:** **1.0293** (Perplexity: **2.80**)

---

## Hardware & Engineering Breakthrough

Training on 5× CMP 170HX mining cards presents a severe architectural bottleneck:
* **Compute & Memory:** 162.3 TFLOPS bf16 GEMM and 1,577 GB/s HBM2e memory bandwidth per card (A100 silicon).
* **Interconnect:** Physically limited to **PCIe Gen2 x4 electrical** (~2 GB/s unidirectional, no P2P, no NVLink). NCCL all-reduce algorithm bandwidth is only **0.75 GiB/s**.

### How we solved it:
1. **Large Global Batches:** 983,040 tokens/step (16 micro-batch × 6 grad accum × 5 GPUs × 2048 ctx). Forward/backward compute takes ~6.8s while interconnect sync takes ~0.35s ($< 5\%$ overhead).
2. **Gradient Compression:** DDP was wired with `bf16_compress_hook`, transmitting bf16 gradients across NCCL and cutting transfer latency in half.
3. **Kernel Compilation:** PyTorch 2 `torch.compile` fuses norms, activations, and residual additions for maximum hardware saturation.

---

## Quickstart & Usage

### 1. Interactive Terminal Chat
Chat with the locally hosted model via standard CLI:
```bash
python3 scripts/chat.py
```

### 2. Serving via vLLM
Serve the exported weights with upstream vLLM:
```bash
vllm serve /lm24h/export/chat \
  --served-model-name lm24h-chat \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.25
```

### 3. OpenAI-Compatible API Call
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lm24h-chat",
    "messages": [
      {"role": "user", "content": "Write a python function to compute prime numbers."}
    ],
    "temperature": 0.7
  }'
```

---

## Repository Structure

```
├── configs/
│   ├── pretrain_500m.toml       # Pretraining hyperparameters & schedule
│   └── sft_500m.toml            # SFT hyperparameters & chat config
├── src/lm24h/
│   ├── model.py                 # Qwen3 causal architecture (GQA, QK-norm, SwiGLU)
│   ├── data.py                  # Packed shards memory-mapped dataloader
│   ├── train.py                 # DDP distributed trainer with WSD schedule
│   ├── export_hf.py             # 1:1 Hugging Face exporter + ChatML template
│   └── generate.py              # Sampling and evaluation script
├── scripts/
│   ├── bench_hw.py              # Single-GPU hardware benchmarking
│   ├── bench_comm.py            # Multi-GPU NCCL interconnect benchmark
│   ├── bench_train.py           # Model sizing throughput comparison
│   ├── train_tokenizer.py       # Custom 32k BPE tokenizer training
│   ├── prep_data.py             # Parallel uint16 tokenization for FineWeb
│   ├── prep_sft.py              # SmolTalk ChatML tokenization & masking
│   ├── launch_stage.sh          # Fault-tolerant auto-restarting trainer
│   ├── after_pretrain.sh        # Unattended end-to-end post-pretrain pipeline
│   └── chat.py                  # Interactive CLI chat client
├── tests/
│   └── test_model.py            # Unit tests & exact HF Qwen3 logit parity
├── results/
│   ├── metrics.json             # Key numerical metrics & loss values
│   ├── samples_base.txt         # Generations from raw base model
│   └── samples_chat.txt         # Generations from fine-tuned chat model
└── FINDINGS.md                  # Comprehensive scientific and technical report
```

---

## License
MIT License. Created by dkpoulsen.
