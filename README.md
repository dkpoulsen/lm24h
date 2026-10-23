# lm24h — 494M Qwen3-architecture LM, pretrain + SFT in ~21h on 5x CMP 170HX

Model: 24L, d=1280, 20 heads / 5 KV (GQA), head_dim 64, SwiGLU 3840, QK-norm, RoPE θ=1e5, RMSNorm, tied emb, 32k BPE, ctx 2048. Exports 1:1 to HF `Qwen3ForCausalLM`.
Pretrain: FineWeb-Edu sample-10BT (10.08B tok), 9200 steps x 983k tok = 9.04B tok, AdamW (0.9/0.95, wd 0.1), lr 8e-4 WSD, bf16 autocast + fp32 master, DDP w/ bf16 grad compression (PCIe Gen2 x4, no P2P: allreduce ≈0.75 GiB/s).
SFT: SmolTalk (579k packed rows, ChatML, assistant-only loss), 1 epoch, lr 1e-4.

## Layout
- `src/lm24h/{model,data,train,export_hf,generate}.py`, `configs/*.toml`, `tests/` (pytest incl. HF Qwen3 logit parity)
- `scripts/run.sh <gpus> <python args>` — podman run (image localhost/glm53-engine:1.8.0; extra pip deps in /home/dkp/lm24h/pylib via PYTHONPATH)
- `scripts/launch_stage.sh <cfg> <log>` — auto-restart from latest ckpt; `touch /home/dkp/lm24h/STOP` to prevent restart; SIGTERM → saves ckpt + exits
- `scripts/after_pretrain.sh` — unattended: SFT → export → samples. Log `/home/dkp/lm24h/runs/pipeline.log`

## Data/outputs (/home/dkp/lm24h)
data/{raw,tok,sft,tokenizer}, runs/lm24h-500m/{train.log,tb,ckpt}, runs/lm24h-500m-sft, export/{base,chat}, runs/samples_{base,chat}.txt

## Monitor
`grep -E "^step|eval" /home/dkp/lm24h/runs/lm24h-500m/train.log | tail`; TensorBoard: `tensorboard --logdir /home/dkp/lm24h/runs`
Tests: `CNAME=t scripts/run.sh 4 -m pytest -q tests`

## Notes
- Pretrain uses all 5 GPUs; ds41-flash/assist/glm services must stay stopped until the pipeline logs ALL DONE.
- Documents are packed without cross-doc attention masking (standard for this scale).
