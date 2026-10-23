import torch, time
import torch.nn.functional as F
d='cuda'
def t(f,n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); s=time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time()-s)/n
print(torch.cuda.get_device_name(0), torch.__version__)
for dt in (torch.bfloat16, torch.float16, torch.float32):
    a=torch.randn(8192,8192,device=d,dtype=dt); b=torch.randn(8192,8192,device=d,dtype=dt)
    dtm=t(lambda: a@b); print(f"matmul 8192^3 {dt}: {2*8192**3/dtm/1e12:.1f} TFLOPS")
torch.backends.cuda.matmul.allow_tf32=True
a=torch.randn(8192,8192,device=d); b=torch.randn(8192,8192,device=d)
dtm=t(lambda: a@b); print(f"matmul fp32+tf32: {2*8192**3/dtm/1e12:.1f} TFLOPS")
B,H,S,D=8,16,2048,64
q,k,v=[torch.randn(B,H,S,D,device=d,dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
def fa(): F.scaled_dot_product_attention(q,k,v,is_causal=True).sum().backward()
dtm=t(fa,10); fl=4*B*H*S*S*D*0.5*3.5; print(f"sdpa causal fwd+bwd bf16: {fl/dtm/1e12:.1f} TFLOPS eff")
x=torch.randn(1<<28,device=d); dtm=t(lambda: x.mul_(1.0001)); print(f"mem bw: {2*x.numel()*4/dtm/1e9:.0f} GB/s")
print("mem free GiB", torch.cuda.mem_get_info()[0]/2**30)
