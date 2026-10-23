"""Single-GPU throughput benchmark of candidate configs (picks the model size)."""
import sys, time, torch
from lm24h.model import ModelConfig, Transformer
S=2048
cands = {
 "300M": dict(n_layer=24,d_model=1024,n_head=16,n_kv_head=4,head_dim=64,ffn_dim=3072),
 "470M": dict(n_layer=24,d_model=1280,n_head=20,n_kv_head=5,head_dim=64,ffn_dim=3840),
 "600M": dict(n_layer=28,d_model=1408,n_head=22,n_kv_head=2,head_dim=64,ffn_dim=4224),
}
names = sys.argv[1:] or list(cands)
for name in names:
    c = ModelConfig(max_seq_len=S, **cands[name])
    m = Transformer(c).cuda()  # fp32 master weights
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    mc = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    for mb in (16, 8, 4):
        try:
            x = torch.randint(0, c.vocab_size, (mb, S), device="cuda")
            def step():
                with ac: loss = mc(x, x)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            for _ in range(4): step()
            torch.cuda.synchronize(); t=time.time(); n=10
            for _ in range(n): step()
            torch.cuda.synchronize(); dt=(time.time()-t)/n
            tps = mb*S/dt; tf = tps*c.flops_per_token(S)/1e12
            print(f"{name} params={c.n_params()/1e6:.0f}M mb={mb} {tps:,.0f} tok/s {tf:.1f} TFLOPS mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(name, "OOM mb", mb)
    del m, mc, opt; torch.cuda.empty_cache()
