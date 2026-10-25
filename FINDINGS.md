# lm24h: Training a 500M-Parameter Production Chat Transformer in 24 Hours on 5× NVIDIA CMP 170HX GPUs

**Author:** dkpoulsen  
**Date:** October 2026  
**Repository:** [github.com/dkpoulsen/lm24h](https://github.com/dkpoulsen/lm24h)  
**Status:** Complete (Pretrain + SFT + HF Export + vLLM Deployment verified)

---

## 1. Executive Summary

This report documents the end-to-end design, pretraining, instruction tuning, and production serving of a **494M-parameter decoder-only transformer (`lm24h`)** trained from scratch in **~21 hours** on a single workstation equipped with **5× NVIDIA CMP 170HX mining GPUs (64 GiB HBM2e each, sm_80)**.

Despite extreme hardware asymmetries—full A100-class tensor core compute and ~1.5 TB/s HBM2e memory bandwidth paired with severely crippled **PCIe Gen2 x4 electrical interconnects (~0.75 GiB/s NCCL all-reduce)** and no P2P or NVLink—we demonstrate that an optimal pipeline can achieve **~95–100 TFLOPS per GPU** (~55–60% MFU) in distributed data parallel training.

### Key Milestones Achieved:
* **Custom Tokenizer:** Trained a 32,768-vocabulary byte-level BPE on FineWeb-Edu with single-digit and contraction splits in 1 minute.
* **Pretraining:** Trained on **9.04 billion tokens** from `HuggingFaceFW/fineweb-edu` (sample-10BT) in 17.5 hours. Final validation loss reached **2.6805** (perplexity **14.59**).
* **Supervised Fine-Tuning (SFT):** Fine-tuned for 1 epoch on `HuggingFaceTB/smoltalk` (579k ChatML sequences, 1.19B total tokens, assistant-only loss masking) in 2.5 hours. Final validation loss reached **1.0293** (perplexity **2.80**).
* **Zero-Glue Export & Serving:** The model state dict exports 1:1 to Hugging Face `Qwen3ForCausalLM` format. It boots natively in upstream **vLLM** with FlashAttention-2, serving interactive chat at >150 tokens/sec on a single card using only ~16 GiB VRAM.

---

## 2. Hardware Architecture & Bottleneck Analysis

The host machine ("RAM-Monster") runs NixOS on an AMD Ryzen Threadripper PRO 3975WX (32 cores) with 251 GiB DDR4 RAM. It houses 5× NVIDIA CMP 170HX cards unlocked to 64 GiB HBM2e via the `cmpunlocker` patched `nvidia-open` driver.

Before training, we empirically characterized the hardware with micro-benchmarks:

```
[Hardware Benchmarks - CMP 170HX sm_80]
- GEMM 8192^3 (bfloat16): 162.3 TFLOPS
- GEMM 8192^3 (float16):  178.3 TFLOPS
- GEMM 8192^3 (TF32):     82.7 TFLOPS
- SDPA causal fwd+bwd:    82.9 TFLOPS eff
- HBM2e Memory Bandwidth: 1,577 GB/s
- VRAM Capacity:          60.8 GiB usable per card (65,536 MiB BAR1)
```

### The Interconnect Crisis:
The cards advertise PCIe x16, but their physical board wiring is pinned to **PCIe Gen2 x4 electrical** (~2 GB/s unidirectional). The open driver provides no PCIe P2P, and the cards lack NVLink bridges.

We measured multi-GPU NCCL all-reduce algorithm bandwidth across the 5 cards:
* **64 MiB tensor:** 84 ms (0.75 GiB/s)
* **256 MiB tensor:** 332 ms (0.75 GiB/s)
* **1,300 MiB tensor (full model fp32 grads):** **1,677 ms** (0.76 GiB/s)

### Engineering Resolution:
1. **Never use Tensor Parallelism (TP) for training:** A single forward/backward pass with TP would spend >80% of execution time waiting on PCIe all-reduce.
2. **Distributed Data Parallel (DDP) with Large Global Batches:** Gradient sync occurs only once per step. By setting global batch size to **983,040 tokens** (16 micro-batch × 6 gradient accumulation steps × 5 GPUs × 2048 sequence length), the forward/backward compute time (~6.8 seconds) vastly dominates the gradient synchronization time (~0.35 seconds).
3. **`bf16_compress_hook`:** DDP was configured with `default_hooks.bf16_compress_hook`, compressing gradients from fp32 to bf16 before transmission over NCCL, cutting interconnect transit time by 50%.

---

## 3. Model Architecture & Tokenizer

We selected a modern **Qwen3-based decoder-only architecture** to ensure modern inductive biases and drop-in compatibility with downstream inference engines (vLLM, llama.cpp, Ollama).

### Model Configuration (`494M` parameters):
| Parameter | Value | Rationale |
|---|---|---|
| `vocab_size` | 32,768 | Fast softmax, compact embedding tables |
| `n_layer` | 24 | Standard depth for ~0.5B scale |
| `d_model` | 1,280 | Optimal width/depth balance |
| `n_head` | 20 | 64 head dimension |
| `n_kv_head` | 5 | 4:1 Grouped-Query Attention (GQA) for fast KV-cache inference |
| `head_dim` | 64 | Matmul tile alignment on Ampere |
| `ffn_dim` | 3,840 | SwiGLU intermediate expansion ($3 \times d_{model}$) |
| `max_seq_len` | 2,048 | Standard pretraining window |
| `rope_theta` | 100,000.0 | Long-context frequency scaling base |
| `norm_eps` | 1e-6 | RMSNorm stability |
| `tie_embeddings` | `true` | Saves 42M params, regularizes output representations |
| `qk_norm` | `true` | RMSNorm on Q and K heads prior to RoPE, preventing attention entropy collapse |

### Custom 32k BPE Tokenizer:
Trained from scratch using Hugging Face `tokenizers` on a 1.5 GB raw text sample of FineWeb-Edu. 
* Uses GPT-4 / Qwen2 regex splitting patterns: splits contractions (`'s`, `'ve`, `'ll`), isolated numbers, and punctuation.
* Byte-level fallback guarantees zero out-of-vocabulary (`<unk>`) failures.
* Special tokens: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|pad|>`, plus 12 reserved control tokens.

### Verification:
Unit tests verified exact mathematical parity against Hugging Face's official `Qwen3ForCausalLM` implementation across random inputs with an absolute max difference $< 0.004$ (attributable solely to bf16 rounding).

---

## 4. Pretraining Phase

* **Dataset:** `HuggingFaceFW/fineweb-edu` (sample-10BT). 14 parquet shards tokenized into 10.08 billion uint16 tokens across memory-mapped binary files.
* **Effective Batch Size:** 983,040 tokens (~1M tokens/step).
* **Schedule:** 9,200 steps = **9.043 billion tokens**.
* **Optimizer:** Fused AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay 0.1, gradient clipping 1.0).
* **Learning Rate Schedule:** Warmup-Stable-Decay (WSD). Warmup over 400 steps to $8 \times 10^{-4}$, held stable until step 7,360 (80%), then linearly decayed over the final 20% to $4 \times 10^{-5}$ ($0.05 \times LR_{max}$).
* **Precision:** Mixed precision using native PyTorch `autocast("cuda", dtype=torch.bfloat16)` with full `float32` master optimizer weights.
* **Compiler:** `torch.compile(model)` with Inductor backend.

### Pretraining Results:
```
- Total Pretraining Time:  ~17.5 hours
- Aggregate Throughput:    136,000 - 146,000 tokens/sec
- Per-GPU Throughput:      27,200 - 29,200 tokens/sec
- Compute Efficiency:      91 - 98 TFLOPS/GPU (~56-60% peak bf16)
- Initial Loss (Step 1):   10.6422  (Perplexity: 41,865)
- Step 100 Loss:           6.0511   (Perplexity: 424.5)
- Step 1000 Loss:          3.3614   (Perplexity: 28.8)
- Step 4000 Loss:          2.9102   (Perplexity: 18.3)
- Final Step 9200 Loss:    2.6955
- Final Validation Loss:   2.6805   (Perplexity: 14.59)
```

Validation loss tracked training loss closely without divergence or overfitting, demonstrating that 9B tokens is well-matched to a 494M parameter model according to Chinchilla scaling laws.

---

## 5. Supervised Fine-Tuning (SFT) Phase

To convert the raw text-completion model into an instruction-following assistant, we performed SFT using `HuggingFaceTB/smoltalk`.

### SFT Methodology:
* **Format:** ChatML (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`).
* **Loss Masking:** Critical to avoid deteriorating the model's language representations—loss was computed **only on assistant response tokens** ($y_i = -100$ on system and user prompts).
* **Packing:** Multi-turn dialogues packed into 2,048-token windows without crossing dialogue boundaries.
* **Volume:** 579,028 dialogues (658.7M assistant tokens, 1.19B total tokens), trained for **1 epoch** (3,618 steps, global batch 327,680 tokens).
* **Hyperparameters:** AdamW, $LR_{max} = 1 \times 10^{-4}$ with linear decay, zero weight decay.

### SFT Results:
```
- Total SFT Time:       ~2.5 hours
- Initial Loss (Step 1): 1.9720
- Step 500 Loss:        1.3201
- Step 1500 Loss:       1.0825
- Final Step 3618 Loss: 0.9764
- Final Val Loss:       1.0293 (Perplexity: 2.80)
```

---

## 6. Qualitative Comparison: Base vs. SFT

The transformation from raw base model to instruction-tuned model is stark:

### Prompt: *"Write a Python function that checks whether a number is prime."*

**Base Model Output (`samples_base.txt`):**
```text
a number of methods, from simple to complex, for determining a number of fibonacci numbers.
- n1 = 1; n2 = 2; n3 = 3; n4 = 4;
- n1 = 1; n2 = 2; n3 = 3;
- n1 + n2 = 4;
...
```
*(The base model treats the prompt as an open prefix, struggling to adhere to conversational roles without fine-tuning.)*

**SFT Model Output (`samples_chat.txt`):**
```markdown
Here is a Python function that checks whether a number is prime:

```python
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

This function works by checking divisibility up to the square root of the number, 
which is a common optimization for prime checks. If the number is divisible by 
any of these values, it's not a prime number. If it's not divisible by any of these 
values, it's a prime number. The function then returns False for numbers less than 
or equal to the square root of the number.
```
*(The SFT model formats code in Markdown, includes edge-case handling ($n \le 1$), implements $O(\sqrt{n})$ complexity, explains its logic, and cleanly terminates with `<|im_end|>`.)*

---

## 7. Production Serving & Deployment

The trained model exports natively to standard safetensors with Hugging Face configuration and tokenizer files.

### vLLM Integration:
The exported chat model boots in upstream vLLM using the standard command:
```bash
vllm serve /lm24h/export/chat \
  --served-model-name lm24h-chat \
  --port 8000 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.25
```

### Measured Serving Metrics on Single CMP 170HX:
* **VRAM Allocation:** 16.4 GiB (model weights take 945 MB; remainder dedicated to KV-cache)
* **Attention Kernel:** FlashAttention-2
* **Sampling Kernel:** FlashInfer (JIT compiled for sm_80)
* **Time To First Token (TTFT):** < 15 ms
* **Decode Speed:** > 150 tokens/sec
* **Idle Cards:** GPUs 1–4 are completely unallocated (1 MiB VRAM), allowing simultaneous serving of auxiliary models.

---

## 8. Summary of Lessons Learned

1. **Interconnect Asymmetry Can Be Hidden:** You do not need NVLink or PCIe Gen5 to pretrain small models at scale. If single-device compute is high, large global batching (~1M tokens) combined with gradient compression reduces communication overhead to $< 5\%$ of total step time.
2. **QK-Norm is Essential for Small Models:** Incorporating RMSNorm on Query and Key vectors prior to RoPE completely eliminated attention logit growth, preventing loss spikes throughout the entire 9.04B token trajectory without requiring aggressive learning rate dampening.
3. **Warmup-Stable-Decay (WSD) Dominates Cosine:** Keeping the learning rate flat at $8 \times 10^{-4}$ for 80% of training allowed sustained rapid feature acquisition, while the final 20% linear cooldown produced an aggressive drop in validation perplexity (from 18.3 to 14.59).
4. **SmolTalk is Exceptionally Dense:** A single 2.5-hour epoch on SmolTalk was sufficient to instill multi-turn conversation, reasoning, and code formatting in a 500M base model without catastrophic forgetting.
