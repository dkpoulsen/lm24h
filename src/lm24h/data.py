"""Deterministic, resumable packed-token loader over uint16 .bin shards (no state needed to resume)."""
import glob, threading, queue, numpy as np, torch


class PackedShards:
    def __init__(self, pattern: str, seq_len: int, seed: int = 1234):
        self.S, self.seed = seq_len, seed
        self.files = sorted(glob.glob(pattern)); assert self.files, f"no shards match {pattern}"
        self.mm = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        nw = [(len(m) - 1) // seq_len for m in self.mm]          # windows per shard
        self.cum = np.cumsum([0] + nw); self.n = int(self.cum[-1]); self._perm = (None, None)
        self.tokens = int(sum(len(m) for m in self.mm))

    def _perm_for(self, epoch):
        if self._perm[0] != epoch:
            self._perm = (epoch, np.random.default_rng((self.seed, epoch)).permutation(self.n))
        return self._perm[1]

    def window(self, sample_idx: int):
        j = int(self._perm_for(sample_idx // self.n)[sample_idx % self.n])
        k = int(np.searchsorted(self.cum, j, side="right") - 1); off = (j - self.cum[k]) * self.S
        a = self.mm[k][off: off + self.S + 1]
        return a

    def batch(self, first_idx: int, count: int):
        arr = np.stack([self.window(first_idx + i) for i in range(count)]).astype(np.int64)
        t = torch.from_numpy(arr)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


class Prefetcher:
    """Background thread yielding (x, y) cuda tensors for a fixed list of sample-index starts."""
    def __init__(self, ds, starts, count, device, depth=6):
        self.q = queue.Queue(depth); self.dev = device
        def run():
            for s in starts:
                x, y = ds.batch(s, count); self.q.put((x.pin_memory(), y.pin_memory()))
            self.q.put(None)
        threading.Thread(target=run, daemon=True).start()
    def __iter__(self): return self
    def __next__(self):
        it = self.q.get()
        if it is None: raise StopIteration
        return it[0].to(self.dev, non_blocking=True), it[1].to(self.dev, non_blocking=True)


class SFTRows(PackedShards):
    """Rows of (S+1) tokens + aligned assistant-loss mask; y = -100 where mask is 0."""
    def __init__(self, prefix: str, seq_len: int, seed: int = 1234):
        self.S, self.seed = seq_len, seed
        self.tok = np.memmap(prefix + ".tok", dtype=np.uint16, mode="r").reshape(-1, seq_len + 1)
        self.msk = np.memmap(prefix + ".mask", dtype=np.uint8, mode="r").reshape(-1, seq_len + 1)
        self.n = len(self.tok); self._perm = (None, None); self.tokens = int(self.msk.sum())
    def batch(self, first_idx: int, count: int):
        js = [int(self._perm_for((first_idx + i) // self.n)[(first_idx + i) % self.n]) for i in range(count)]
        t = torch.from_numpy(self.tok[js].astype(np.int64)); m = torch.from_numpy(self.msk[js].astype(bool))
        x, y = t[:, :-1].contiguous(), t[:, 1:].clone(); y[~m[:, 1:]] = -100
        return x, y
