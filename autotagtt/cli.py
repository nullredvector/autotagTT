import argparse, json, os, random, time
from .archive import OUT, DEFAULT_MODEL, DEFAULT_PRETRAINED, load_groups, enabled_tags, label, write_json_atomic
from .store import EmbStore


def main():
    os.umask(0o077)   # everything we create is owner-only
    ap = argparse.ArgumentParser(prog="autotagtt")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("embed", "train", "run", "montage", "worker", "notify-test"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
        p.add_argument("--tags", default="", help="comma separated group names (default: the tags enabled in kashibox)")
        p.add_argument("--limit", type=int, help="embed at most N new covers (smoke test)")
        p.add_argument("--top", type=int, default=300)
    a = ap.parse_args()
    cmd = a.cmd or "run"
    if cmd == "notify-test":
        from . import notify
        print("sent" if notify.notify("test", "autotagTT: test notification", "If you can read this, notifications are working.", tags=["bell"]) else "not sent (is NTFY_URL set?)")
        return
    if cmd == "worker":
        from .worker import main as worker_main
        return worker_main()
    tags = [t.strip().lower() for t in a.tags.split(",") if t.strip()] or enabled_tags()

    if cmd in ("embed", "run"):
        from .embed import embed_covers
        n_before = len(EmbStore(a.model, "cover").done_ids())
        embed_covers(a.model, a.pretrained, limit=getattr(a, "limit", None))
        if cmd == "embed":
            return
    store = EmbStore(a.model, "cover")
    ids, X = store.load()
    if cmd in ("train", "run"):
        _train_and_write(a, tags, ids, X, new_covers=(len(ids) - n_before) if cmd == "run" else 0, with_frames=(cmd == "run"))
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


class _Lazy:
    """Duplicate groups are expensive and identical for every tag: compute them once, only if needed."""
    def __init__(self, X):
        self.X, self.val = X, None

    def __call__(self):
        if self.val is None:
            from .train import dup_groups
            self.val = dup_groups(self.X)
            print(f"near-duplicate groups: {len(set(self.val.tolist()))} for {len(self.val)} images", flush=True)
        return self.val


def _frames_step(a, ids, X, tags, process=True):
    """Frame analysis as configured in kashibox: off | trial | full. Returns FrameFeatures or None."""
    import time
    from .archive import find_videos
    from .frames import FrameStore, frames_config, embed_videos, coverage, write_stats
    from .train import FrameFeatures, select_trial, trial_report
    cfg = frames_config()
    fstore = FrameStore(a.model)
    if process and cfg["mode"] != "off":
        videos = find_videos()
        have = fstore.done_ids() | fstore.skipped_ids()
        if cfg["mode"] == "trial":
            want = select_trial([v for v in ids if v in videos], tags)
            todo = [(v, videos[v]) for v in sorted(want, reverse=True) if v not in have]
        else:
            todo = [(v, videos[v]) for v in sorted((v for v in ids if v in videos), reverse=True) if v not in have]   # newest first
        print(f"frames: mode={cfg['mode']} mouth={cfg['mouth']} {len(todo)} videos to do", flush=True)
        if todo:
            from .embed import ImageEmbedder
            from .frames import embed_videos
            threads = int(os.environ.get("AUTOTAGTT_THREADS", "8"))
            st = embed_videos(ImageEmbedder(a.model, a.pretrained), todo, fstore, mouth=cfg["mouth"],
                              budget_s=cfg["budget_min"] * 60, threads=threads)
            done, total = coverage(ids, fstore, videos)
            write_stats(done, total, st["done"] / max(st["seconds"], 1e-6), cfg["mode"])
        else:
            done, total = coverage(ids, fstore, videos)
            write_stats(done, total, 0, cfg["mode"])
    fids, F = fstore.load_pooled()
    if not len(fids):
        return None
    videos = find_videos()
    done, total = coverage(ids, fstore, videos)
    ff = FrameFeatures(fids, F, ids, done / max(total, 1))
    if process and cfg["mode"] == "trial":
        trial_report(tags, ids, X, ff, set(json.loads((OUT / "frames_trial_ids.json").read_text())))
    return ff


def _train_and_write(a, tags, ids, X, new_covers, with_frames):
    from .train import train_tag, write_suggestions, write_ai_tags
    t0 = time.time()
    ff = _frames_step(a, ids, X, tags, process=with_frames)
    lazy = _Lazy(X)
    res, modes = {}, {}
    for t in tags:
        r = train_tag(t, ids, X, top=a.top, grp=lazy, ff=ff, model=a.model)
        if r is not None:
            res[t] = r
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    if not res:
        raise SystemExit("no tag could be trained")
    write_suggestions(res, a.model, a.pretrained)
    write_ai_tags(res, a.model)
    changed = bool(new_covers) or modes.get("trained", 0) > 0 or modes.get("incremental", 0) > 0
    write_json_atomic(OUT / "run_summary.json", {"changed": changed, "new_covers": new_covers, "modes": modes,
                                                 "frames_used": bool(ff and ff.ready), "seconds": round(time.time() - t0)})
    print(f"run summary: new_covers={new_covers} " + " ".join(f"{k}={v}" for k, v in modes.items()), flush=True)


if __name__ == "__main__":
    main()
