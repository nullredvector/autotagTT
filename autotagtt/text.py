"""Captions as an extra signal. Each caption is embedded locally with the model's text tower (same space as the images) and
cached like the cover embeddings. Raw caption text is never stored, logged or sent anywhere; only the derived vectors are kept
(owner-only, under data/.appdata/autotag/emb/)."""
import base64, gzip, json, re
import numpy as np
from .archive import APPDATA, DEFAULT_MODEL, DEFAULT_PRETRAINED
from .store import EmbStore

TEXT_READY = 0.5            # captions are used for a tag only when at least this share of videos has one
MIN_CHARS = 3


def load_captions():
    """{video_id: caption} from the viewer's db_texts.js ({} if missing/unreadable). Same format the viewer reads."""
    try:
        txt = (APPDATA / "db_texts.js").read_text()
        m = re.search(r'_base64\s*=\s*"([A-Za-z0-9+/=]+)"', txt) or re.search(r'"([A-Za-z0-9+/=]{40,})"', txt)
        data = json.loads(gzip.decompress(base64.b64decode(m.group(1))))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v.strip() for k, v in data.items() if isinstance(v, str) and len(v.strip()) >= MIN_CHARS}


def embed_texts(embedder, ids, model=DEFAULT_MODEL, batch=2048, log=print):
    """Embed captions for ids not yet in the cache. Returns the number added."""
    store = EmbStore(model, "text")
    caps = load_captions()
    done = store.done_ids()
    todo = [v for v in ids if v in caps and v not in done]
    log(f"captions: {len(caps)} available, {len(done)} embedded, {len(todo)} to do")
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        store.add(chunk, embedder.encode([caps[v] for v in chunk]))
    return len(todo)


class LazyTextEmbedder:
    """Loads the model only if there is something to embed (a quiet night never touches the GPU)."""

    def __init__(self, model=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED):
        self.model, self.pretrained, self._e = model, pretrained, None

    def encode(self, texts):
        if self._e is None:
            from .embed import TextEmbedder
            self._e = TextEmbedder(self.model, self.pretrained)
        return self._e.encode(texts)
