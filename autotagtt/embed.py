import time
import numpy as np
from PIL import Image
from .archive import find_covers, DEFAULT_MODEL, DEFAULT_PRETRAINED
from .store import EmbStore


class CoverDS:
    def __init__(self, items, preprocess):
        self.items, self.pre = items, preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, path = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return self.pre(Image.new('RGB', (64, 64))), vid, False
        return self.pre(img), vid, True


_MODEL = {}


def load_model(model=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED):
    """Loaded once per process; the covers and frames steps share it."""
    key = (model, pretrained)
    if key not in _MODEL:
        import open_clip
        m, _, pre = open_clip.create_model_and_transforms(model, pretrained=pretrained, precision="fp16", device="cuda")
        m.eval()
        _MODEL[key] = (m, pre, None)
    return _MODEL[key]


class ImageEmbedder:
    """encode(list of PIL images) -> (n, d) float32, L2-normalised. Used for video frames and face crops."""

    def __init__(self, model=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED, batch=64):
        self.m, self.pre, _ = load_model(model, pretrained)
        self.batch = batch

    def encode(self, images):
        import torch
        out = []
        with torch.no_grad():
            for s in range(0, len(images), self.batch):
                x = torch.stack([self.pre(im) for im in images[s:s + self.batch]]).cuda().half()
                f = torch.nn.functional.normalize(self.m.encode_image(x).float(), dim=-1)
                out.append(f.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


class TextEmbedder:
    """encode(list of captions) -> (n, d) float32, L2-normalised, in the same space as the image embeddings (SigLIP text tower)."""

    def __init__(self, model=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED, batch=256):
        import open_clip
        self.m, _, _ = load_model(model, pretrained)
        self.tok = open_clip.get_tokenizer(model)
        self.batch = batch

    def encode(self, texts):
        import torch
        out = []
        with torch.no_grad():
            for s in range(0, len(texts), self.batch):
                t = self.tok([x[:300] for x in texts[s:s + self.batch]]).cuda()
                f = torch.nn.functional.normalize(self.m.encode_text(t).float(), dim=-1)
                out.append(f.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def embed_covers(model=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED, batch=256, shard=4096, workers=8, limit=None):
    store = EmbStore(model, "cover")
    covers = find_covers()
    done = store.done_ids()
    todo = sorted((v, p) for v, p in covers.items() if v not in done)
    if limit:
        todo = todo[:limit]
    print(f"covers: {len(covers)} total, {len(done)} embedded, {len(todo)} to do", flush=True)
    if not todo:
        return store
    import torch
    m, pre, _ = load_model(model, pretrained)
    t0, n = time.time(), 0
    for s in range(0, len(todo), shard):
        chunk = todo[s:s + shard]
        dl = torch.utils.data.DataLoader(CoverDS(chunk, pre), batch_size=batch, num_workers=workers)
        ids, vecs = [], []
        with torch.no_grad():
            for x, vid, ok in dl:
                f = m.encode_image(x.cuda().half())
                f = torch.nn.functional.normalize(f.float(), dim=-1)
                keep = [i for i, o in enumerate(ok) if o]
                ids += [vid[i] for i in keep]
                vecs.append(f[keep].cpu().numpy())
        if ids:
            store.add(ids, np.concatenate(vecs))
        n += len(chunk)
        r = n / (time.time() - t0)
        print(f"  {n}/{len(todo)}  {r:.0f} img/s  eta {(len(todo)-n)/r/60:.1f} min  vram {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
    return store
