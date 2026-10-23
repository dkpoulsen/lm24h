"""Tokenize FineWeb-Edu parquet -> uint16 shards (docs joined with <|endoftext|>). Holds out first 20k docs of last shard as val."""
import glob, os, sys, numpy as np, pyarrow.parquet as pq
from multiprocessing import Pool
from tokenizers import Tokenizer
RAW="/lm24h/data/raw"; OUT="/lm24h/data/tok"; os.makedirs(OUT, exist_ok=True)
VAL_DOCS=20000
def work(f):
    tok=Tokenizer.from_file("/lm24h/data/tokenizer/tokenizer.json"); eos=tok.token_to_id("<|endoftext|>")
    name=os.path.basename(f).replace(".parquet",""); last = f==sorted(glob.glob(f"{RAW}/*.parquet"))[-1]
    outs={"train":[], "val":[]}; ndoc=0
    for b in pq.ParquetFile(f).iter_batches(batch_size=1000, columns=["text"]):
        enc=tok.encode_batch(b.column(0).to_pylist())
        for e in enc:
            split="val" if (last and ndoc<VAL_DOCS) else "train"; ndoc+=1
            outs[split].append(np.array(e.ids+[eos],dtype=np.uint16))
    res={}
    for split,arrs in outs.items():
        if not arrs: continue
        a=np.concatenate(arrs); a.tofile(f"{OUT}/{split}_{name}.bin"); res[split]=len(a)
    print(name,res,flush=True); return res
if __name__=="__main__":
    files=sorted(glob.glob(f"{RAW}/*.parquet"))
    with Pool(14) as p: r=p.map(work,files,chunksize=1)
    tr=sum(x.get("train",0) for x in r); va=sum(x.get("val",0) for x in r); print("TOTAL train",tr,"val",va)
