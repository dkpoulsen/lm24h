"""Pretraining: DDP + bf16 autocast (fp32 master) + torch.compile + WSD schedule + atomic resumable ckpts."""
import argparse, os, sys, time, json, math, signal, glob, tomllib, torch, torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP
from lm24h.model import ModelConfig, Transformer
from lm24h.data import PackedShards, SFTRows, Prefetcher

def lr_at(step, c):
    if step < c["warmup_steps"]: return c["lr"] * (step + 1) / c["warmup_steps"]
    decay_start = int(c["max_steps"] * (1 - c["decay_frac"]))
    if step < decay_start: return c["lr"]
    t = (step - decay_start) / max(1, c["max_steps"] - decay_start)
    return c["lr"] * (1 - t * (1 - c["min_lr_ratio"]))

def build_optimizer(model, c):
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    nodecay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed" not in n)]
    return torch.optim.AdamW([{"params": decay, "weight_decay": c["weight_decay"]},
                              {"params": nodecay, "weight_decay": 0.0}],
                             lr=c["lr"], betas=tuple(c["betas"]), eps=1e-8, fused=True)

def save_ckpt(path_dir, step, model, opt, cfg, extra):
    os.makedirs(path_dir, exist_ok=True); tmp = f"{path_dir}/.tmp_step{step}.pt"; fin = f"{path_dir}/step{step:07d}.pt"
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, **extra}, tmp)
    os.replace(tmp, fin); return fin

def latest_ckpt(path_dir):
    fs = sorted(glob.glob(f"{path_dir}/step*.pt")); return fs[-1] if fs else None

@torch.no_grad()
def evaluate(model, val, c, rank, world, device):
    model.eval(); tot = torch.zeros(2, device=device); mb = c["micro_batch"]
    for i in range(c["eval_batches"]):
        x, y = val.batch((i * world + rank) * mb, mb); x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16): l = model(x, y)
        tot += torch.stack([l.detach() * x.numel(), torch.tensor(float(x.numel()), device=device)])
    dist.all_reduce(tot); model.train(); return (tot[0] / tot[1]).item()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out", default="/lm24h/runs")
    ap.add_argument("--max_steps", type=int); ap.add_argument("--no_compile", action="store_true"); ap.add_argument("--init", help="override resume")
    a = ap.parse_args()
    cfg = tomllib.load(open(a.config, "rb")); mc, tc = cfg["model"], cfg["train"]
    if a.max_steps: tc["max_steps"] = a.max_steps
    if a.no_compile: tc["compile"] = False
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    lrank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lrank); device = f"cuda:{lrank}"
    torch.manual_seed(tc["seed"]); torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = f"{a.out}/{tc['run_name']}"; ck_dir = f"{run_dir}/ckpt"; S = mc["max_seq_len"]
    model_cfg = ModelConfig(**mc); model = Transformer(model_cfg).to(device)
    opt = build_optimizer(model, tc); step = 0
    ck = latest_ckpt(ck_dir)
    if ck:
        sd = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); step = sd["step"]; del sd
        if rank == 0: print(f"[resume] {ck} step={step}", flush=True)
    elif tc.get("init_from"):   # stage-2 start: weights only, fresh optimizer/schedule
        sd = torch.load(tc["init_from"], map_location=device, weights_only=False); model.load_state_dict(sd["model"]); del sd
        if rank == 0: print(f"[init_from] {tc['init_from']}", flush=True)
    ddp = DDP(model, device_ids=[lrank], bucket_cap_mb=256, gradient_as_bucket_view=True)
    ddp.register_comm_hook(None, default_hooks.bf16_compress_hook)   # halves PCIe traffic (x4 Gen2, no P2P)
    fwd = torch.compile(ddp) if tc["compile"] else ddp
    mb, ga = tc["micro_batch"], tc["grad_accum"]; gb_tokens = mb * ga * world * S
    if tc.get("sft"):
        train = SFTRows(tc["data_train"], S, tc["seed"]); val = SFTRows(tc["data_val"], S, 7)
        if tc.get("epochs"): tc["max_steps"] = int(tc["epochs"] * train.n / (mb * ga * world))
    else:
        train = PackedShards(tc["data_train"], S, tc["seed"]); val = PackedShards(tc["data_val"], S, 7)
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(f"{run_dir}/tb"); os.makedirs(run_dir, exist_ok=True)
        json.dump({"model": mc, "train": tc, "world": world, "params": model_cfg.n_params()}, open(f"{run_dir}/config.json", "w"), indent=1)
        print(f"params={model_cfg.n_params()/1e6:.1f}M global_batch={gb_tokens:,} tok train_tokens={train.tokens/1e9:.2f}B "
              f"total_steps={tc['max_steps']} (={tc['max_steps']*gb_tokens/1e9:.2f}B tok)", flush=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True)); signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    starts = [((s * ga + m) * world + rank) * mb for s in range(step, tc["max_steps"]) for m in range(ga)]
    it = Prefetcher(train, starts, mb, device); fpt = model_cfg.flops_per_token(S)
    model.train(); t_last = time.time(); toks_win = 0; skipped = 0
    while step < tc["max_steps"]:
        lr = lr_at(step, tc)
        for g in opt.param_groups: g["lr"] = lr
        loss_acc = torch.zeros((), device=device)
        for m in range(ga):
            x, y = next(it)
            ctx = ddp.no_sync() if m < ga - 1 else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16): loss = fwd(x, y)
                (loss / ga).backward()
            loss_acc += loss.detach() / ga
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        if torch.isfinite(gn): opt.step()
        else: skipped += 1
        opt.zero_grad(set_to_none=True)
        step += 1; toks_win += gb_tokens
        dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
        if step % 10 == 0 or step == 1:
            torch.cuda.synchronize(); dt = time.time() - t_last; tps = toks_win / dt; t_last = time.time(); toks_win = 0
            if rank == 0:
                lv = loss_acc.item(); mfu = tps * fpt / world / 1e12
                eta = (tc["max_steps"] - step) * gb_tokens / tps / 3600
                print(f"step {step}/{tc['max_steps']} loss {lv:.4f} lr {lr:.2e} gn {gn.item():.2f} {tps:,.0f} tok/s "
                      f"{mfu:.0f} TFLOPS/gpu mem {torch.cuda.max_memory_allocated()/2**30:.0f}G skipped {skipped} eta {eta:.1f}h", flush=True)
                tb.add_scalar("train/loss", lv, step); tb.add_scalar("train/lr", lr, step); tb.add_scalar("train/grad_norm", gn.item(), step)
                tb.add_scalar("perf/tok_s", tps, step); tb.add_scalar("perf/tflops_gpu", mfu, step)
            if not math.isfinite(loss_acc.item()): print("non-finite loss; aborting for restart from ckpt", flush=True); sys.exit(3)
        if step % tc["eval_every"] == 0 or step == tc["max_steps"]:
            vl = evaluate(model, val, tc, rank, world, device)
            if rank == 0: print(f"[eval] step {step} val_loss {vl:.4f} ppl {math.exp(vl):.2f}", flush=True); tb.add_scalar("val/loss", vl, step)
        if step % tc["ckpt_every"] == 0 or step == tc["max_steps"] or stop["flag"]:
            if rank == 0:
                f = save_ckpt(ck_dir, step, model, opt, cfg, {}); print(f"[ckpt] {f}", flush=True)
                for old in sorted(glob.glob(f"{ck_dir}/step*.pt"))[:-tc["keep_ckpts"]]: os.remove(old)
            dist.barrier()
        flag = torch.tensor(float(stop["flag"]), device=device); dist.all_reduce(flag)
        if flag.item() > 0: break
    if rank == 0 and step >= tc["max_steps"]:
        lnk = f"{ck_dir}/final.pt"
        if os.path.lexists(lnk): os.remove(lnk)
        os.symlink(f"step{step:07d}.pt", lnk)
    if rank == 0: print("[done]", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__": main()
