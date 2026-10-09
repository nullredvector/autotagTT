"""Sharded, resumable embedding store: float16 npy + ids json per shard."""
import json, os
import numpy as np
from .archive import OUT, write_json_atomic


class EmbStore:
    def __init__(self, model, feature="cover"):
        slug = model.replace("/", "_").replace(":", "_")
        self.dir = OUT / "emb" / slug / feature
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _shards(self):
        return sorted(self.dir.glob("shard_*.json"))

    def done_ids(self):
        s = set()
        for p in self._shards():
            s.update(json.loads(p.read_text()))
        return s

    def add(self, ids, vecs):
        n = len(self._shards())
        name = f"shard_{n:05d}"
        np.save(self.dir / f"{name}.tmp.npy", vecs.astype(np.float16))
        os.replace(self.dir / f"{name}.tmp.npy", self.dir / f"{name}.npy")
        write_json_atomic(self.dir / f"{name}.json", ids)  # json last = commit marker

    def load(self):
        ids, mats = [], []
        for p in self._shards():
            ids += json.loads(p.read_text())
            mats.append(np.load(p.with_suffix(".npy")))
        if not mats:
            return [], np.zeros((0, 0), np.float16)
        return ids, np.concatenate(mats)
