"""SmolTalk -> ChatML rows of (S+1) tokens with assistant-only loss mask. Convs never straddle rows; overlong convs are truncated."""
import glob, numpy as np, pyarrow.parquet as pq
from multiprocessing import Pool
from tokenizers import Tokenizer
S=2048; RAW="/lm24h/data/sft_raw"; OUT="/lm24h/data/sft"
import os; os.makedirs(OUT, exist_ok=True)
SYS_DEFAULT=None
def render(tok, msgs, ids):
    toks, mask = [], []
    for m in msgs:
        role=m["role"]; head=tok.encode(f"<|im_start|>{role}\n", add_special_tokens=False).ids
        body=tok.encode(m["content"].strip()+"<|im_end|>\n", add_special_tokens=False).ids
        toks+=head+body; mask+=[0]*len(head)+([1]*len(body) if role=="assistant" else [0]*len(body))
    return toks, mask
def work(f):
    tok=Tokenizer.from_file("/lm24h/data/tokenizer/tokenizer.json"); pad=tok.token_to_id("<|pad|>")
    rows_t, rows_m = [], []; cur_t, cur_m = [], []
    def flush():
        nonlocal cur_t, cur_m
        if cur_t:
            n=S+1-len(cur_t); rows_t.append(cur_t+[pad]*n); rows_m.append(cur_m+[0]*n); cur_t, cur_m = [], []
    for b in pq.ParquetFile(f).iter_batches(batch_size=500, columns=["messages"]):
        for msgs in b.column(0).to_pylist():
            if not msgs or not any(m["role"]=="assistant" for m in msgs): continue
            t,m=render(tok,msgs,None); t,m=t[:S+1],m[:S+1]
            if len(cur_t)+len(t)>S+1: flush()
            cur_t+=t; cur_m+=m
    flush()
    return np.array(rows_t,dtype=np.uint16), np.array(rows_m,dtype=np.uint8)
if __name__=="__main__":
    for split,pat in (("val",f"{RAW}/test-*.parquet"),("train",f"{RAW}/train-*.parquet")):
        files=sorted(glob.glob(pat))
        with Pool(min(10,len(files))) as p: r=p.map(work,files)
        T=np.concatenate([x[0] for x in r]); M=np.concatenate([x[1] for x in r])
        T.tofile(f"{OUT}/{split}.tok"); M.tofile(f"{OUT}/{split}.mask")
        print(split,"rows",len(T),"assistant tokens",int(M.sum()),"fill",float((T!=3).mean()),flush=True)
