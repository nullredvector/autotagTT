import argparse, json, os, random
from .archive import OUT, DEFAULT_MODEL, DEFAULT_PRETRAINED, load_groups, enabled_tags, label
from .store import EmbStore
from .train import train_tag, write_suggestions


def main():
    os.umask(0o077)   # everything we create is owner-only
    ap = argparse.ArgumentParser(prog="autotagtt")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("embed", "train", "run", "montage"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
        p.add_argument("--tags", default="", help="comma separated group names (default: the tags enabled in Starplayer)")
        p.add_argument("--limit", type=int, help="embed at most N new covers (smoke test)")
        p.add_argument("--top", type=int, default=300)
    a = ap.parse_args()
    cmd = a.cmd or "run"
    tags = [t.strip().lower() for t in a.tags.split(",") if t.strip()] or enabled_tags()

    if cmd in ("embed", "run"):
        from .embed import embed_covers
        embed_covers(a.model, a.pretrained, limit=getattr(a, "limit", None))
        if cmd == "embed":
            return
    store = EmbStore(a.model, "cover")
    ids, X = store.load()
    if cmd in ("train", "run"):
        res = {t: r for t in tags if (r := train_tag(t, ids, X, top=a.top)) is not None}
        if not res:
            raise SystemExit("no tag could be trained")
        write_suggestions(res, a.model, a.pretrained)
        from .train import write_ai_tags
        write_ai_tags(res, a.model)
    if cmd == "montage":
        from .montage import make_montage
        sugg = json.loads((OUT / "autotag.json").read_text())
        g = load_groups()
        for t in tags:
            s = sugg[t][:100]
            make_montage([x["id"] for x in s], OUT / f"montage_{label(t)}_top.jpg", [x["score"] for x in s])
            pos = sorted(g[t]); random.Random(0).shuffle(pos)
            make_montage(pos[:40], OUT / f"montage_{label(t)}_known.jpg")
            print("wrote montages for", label(t))


if __name__ == "__main__":
    main()
