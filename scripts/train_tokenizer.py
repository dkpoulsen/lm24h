"""Train a 32k byte-level BPE (Qwen2-style pre-tokenizer: single digits, contractions) on a FineWeb-Edu sample."""
import glob, sys, pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, Regex
RAW="/lm24h/data/raw"; OUT="/lm24h/data/tokenizer/tokenizer.json"
SPECIAL=["<|endoftext|>","<|im_start|>","<|im_end|>","<|pad|>"]+[f"<|reserved_{i}|>" for i in range(12)]
PAT=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
def texts(per_file=40000):
    for f in sorted(glob.glob(f"{RAW}/*.parquet")):
        n=0
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text"]):
            for t in b.column(0).to_pylist():
                yield t; n+=1
                if n>=per_file: break
            if n>=per_file: break
tok=Tokenizer(models.BPE())
tok.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PAT),behavior="isolated",invert=False),
                                           pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)])
tok.decoder=decoders.ByteLevel()
tr=trainers.BpeTrainer(vocab_size=32768,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                       min_frequency=2,show_progress=False)
tok.train_from_iterator(texts(),tr)
tok.save(OUT)
print("vocab",tok.get_vocab_size())
t=tok.encode("Hello world! 12345 The quick brown fox — naïve café. def f(x): return x**2\n")
print(t.tokens); print(tok.decode(t.ids))
