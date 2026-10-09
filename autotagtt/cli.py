import argparse, json, random
import numpy as np
from .archive import OUT, load_groups
from .embed import DEFAULT_MODEL, DEFAULT_PRETRAINED, embed_covers
from .store import EmbStore
from .train import train_tag, write_suggestions


def main():
    ap = argparse.ArgumentParser(prog="autotagtt")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("embed", "train", "run", "montage"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
        p.add_argument("--tags", default="braces", help="comma separated group names")
        p.add_argument("--limit", type=int, help="embed at most N new covers (smoke test)")
        p.add_argument("--top", type=int, default=300)
    a = ap.parse_args()
    cmd = a.cmd or "run"
    tags = [t.strip().lower() for t in a.tags.split(",") if t.strip()]

    if cmd in ("embed", "run"):
        embed_covers(a.model, a.pretrained, limit=getattr(a, "limit", None))
        if cmd == "embed":
            return
    store = EmbStore(a.model, "cover")
    ids, X = store.load()
    if cmd in ("train", "run"):
        res = {t: train_tag(t, ids, X, top=a.top) for t in tags}
        write_suggestions(res, a.model)
    if cmd == "montage":
        from .montage import make_montage
        sugg = json.loads((OUT / "autotag.json").read_text())
        g = load_groups()
        for t in tags:
            s = sugg[t][:100]
            make_montage([x["id"] for x in s], OUT / f"montage_{t}_top.jpg", [x["score"] for x in s])
            pos = sorted(g[t]); random.Random(0).shuffle(pos)
            make_montage(pos[:40], OUT / f"montage_{t}_known.jpg")
            print("wrote montages for", t)


if __name__ == "__main__":
    main()
