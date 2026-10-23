import os, time, torch, torch.distributed as dist
dist.init_process_group("nccl"); r=dist.get_rank(); w=dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for mb in (64, 256, 650, 1300):  # bf16 grads of ~32M..650M params
    x=torch.randn(mb*2**20//2, device="cuda", dtype=torch.bfloat16)
    for _ in range(2): dist.all_reduce(x)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(5): dist.all_reduce(x)
    torch.cuda.synchronize(); dt=(time.time()-t)/5
    if r==0: print(f"allreduce {mb} MiB bf16 x{w}: {dt*1000:.0f} ms  ({mb/1024/dt:.2f} GiB/s algbw)", flush=True)
dist.destroy_process_group()
