"""Stage 2: video frames. Samples a few frames per video with FFmpeg, embeds them with the same model as the covers,
and caches the frame-level embeddings (float16, memory-mapped on load). Optionally also embeds a crop of the lower face
(detected with OpenCV's bundled Haar cascade) to catch small dental detail.

Privacy: frames live in a private temp dir (0700) only while a batch is processed, then are deleted. Only embeddings are kept."""
import json, os, re, shutil, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from .archive import OUT, DEFAULT_MODEL, find_videos, write_json_atomic

K = 5                                   # frames sampled per video
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
MAX_ATTEMPTS = 2                        # a video that yields no frames this many times is skipped from then on
COVERAGE_READY = 0.95                   # frame features are used for training once this share of videos is done
STATS = OUT / "frames_stats.json"
PROGRESS = OUT / "frames_progress.json"      # live, updated after every batch; Starplayer shows it on every device
TRIAL_IDS = OUT / "frames_trial_ids.json"
TRIAL_REPORT = OUT / "frames_trial.json"
CONFIG = OUT / "frames.json"            # {"mode": "off|trial|full", "mouth": bool, "budget_min": int}
DEFAULT_CONFIG = {"mode": "off", "mouth": False, "budget_min": 120}


def frames_config():
    try:
        c = json.loads(CONFIG.read_text())
    except Exception:
        c = {}
    mode = c.get("mode") if c.get("mode") in ("off", "trial", "full") else "off"
    b = c.get("budget_min")
    return {"mode": mode, "mouth": c.get("mouth") is True,
            "budget_min": b if isinstance(b, int) and 10 <= b <= 600 else DEFAULT_CONFIG["budget_min"]}


# ───────────── FFmpeg ─────────────
def video_duration(path):
    """Seconds, parsed from `ffmpeg -i` (so ffprobe is not needed); None if unreadable."""
    try:
        r = subprocess.run([FFMPEG, "-nostdin", "-i", str(path)], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
    if not m:
        return None
    d = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return d if d > 0 else None


def extract_frames(path, outdir, k=K):
    """k frames evenly spaced through the video (seeking before decoding, so only a few MB are read)."""
    dur = video_duration(path)
    if not dur:
        return []
    outdir = Path(outdir)
    paths = []
    for i in range(k):
        t = dur * (i + 0.5) / k
        out = outdir / f"f{i}.jpg"
        try:
            subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(path),
                            "-frames:v", "1", "-vf", "scale=384:-2", "-q:v", "3", str(out)],
                           capture_output=True, timeout=60)
        except Exception:
            continue
        if out.exists() and out.stat().st_size > 0:
            paths.append(out)
    return paths


_cascade = None


def lower_face_crop(img):
    """Crop of the lower half of the largest detected face (mouth and chin), or None. Needs opencv (optional)."""
    global _cascade
    try:
        import cv2
    except Exception:
        return None
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    arr = np.asarray(img.convert("L"))
    faces = _cascade.detectMultiScale(arr, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    W, H = img.size
    box = (max(0, int(x - 0.1 * w)), max(0, int(y + 0.5 * h)), min(W, int(x + 1.1 * w)), min(H, int(y + 1.15 * h)))
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return None
    return img.crop(box)


# ───────────── cache ─────────────
class FrameStore:
    """Per-shard files: ids json (commit marker, written last), frames (n,K,d) f16, mask (n,K), mouth (n,d) f16, hasmouth (n,)."""

    def __init__(self, model=DEFAULT_MODEL):
        slug = model.replace("/", "_").replace(":", "_")
        self.base = OUT / "emb" / slug
        self.dir = self.base / "frames"
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.failed_path = self.base / "frames_failed.json"

    def _shards(self):
        return sorted(self.dir.glob("shard_*.json"))

    def done_ids(self):
        s = set()
        for p in self._shards():
            s.update(json.loads(p.read_text()))
        return s

    def failed(self):
        try:
            return json.loads(self.failed_path.read_text())
        except Exception:
            return {}

    def mark_failed(self, ids):
        f = self.failed()
        for v in ids:
            f[v] = f.get(v, 0) + 1
        write_json_atomic(self.failed_path, f)

    def skipped_ids(self):
        return {v for v, n in self.failed().items() if n >= MAX_ATTEMPTS}

    def add(self, ids, frames, mask, mouth, hasmouth):
        name = f"shard_{len(self._shards()):05d}"
        for key, arr, dt in (("frames", frames, np.float16), ("mask", mask, np.bool_), ("mouth", mouth, np.float16), ("hasmouth", hasmouth, np.bool_)):
            tmp = self.dir / f"{name}.{key}.tmp.npy"
            np.save(tmp, np.asarray(arr).astype(dt))
            os.replace(tmp, self.dir / f"{name}.{key}.npy")
        write_json_atomic(self.dir / f"{name}.json", ids)

    def load_pooled(self):
        """(ids, F) with F float16 (n, 3d+1) = [mean over frames, max over frames, mean lower-face crop, has_crop]."""
        ids, rows = [], []
        for p in self._shards():
            base = p.with_suffix("")
            fr = np.load(f"{base}.frames.npy", mmap_mode="r")
            mk = np.load(f"{base}.mask.npy")
            mo = np.load(f"{base}.mouth.npy", mmap_mode="r")
            hm = np.load(f"{base}.hasmouth.npy")
            f32 = np.asarray(fr, dtype=np.float32)
            cnt = np.maximum(mk.sum(1, keepdims=True), 1)
            mean = (f32 * mk[:, :, None]).sum(1) / cnt
            mx = np.where(mk[:, :, None], f32, -np.inf).max(1)
            mx[~mk.any(1)] = 0
            rows.append(np.concatenate([mean, mx, np.asarray(mo, dtype=np.float32), hm[:, None].astype(np.float32)], axis=1).astype(np.float16))
            ids += json.loads(p.read_text())
        if not rows:
            return [], np.zeros((0, 0), np.float16)
        return ids, np.concatenate(rows)


# ───────────── processing ─────────────
def _private_tmp():
    base = Path(tempfile.gettempdir())
    return tempfile.mkdtemp(prefix="atframes_", dir=base)       # mkdtemp is 0700


def embed_videos(embedder, todo, store, mouth=False, budget_s=None, threads=8, batch=32, log=print, on_progress=None):
    """todo: [(video_id, path)]. Resumable (finished batches are committed); stops after budget_s seconds."""
    t0 = time.time()
    done = failed = 0
    d = None
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for s in range(0, len(todo), batch):
            if budget_s is not None and time.time() - t0 > budget_s:
                log(f"frames: time budget reached after {done} videos; the rest continues next run")
                break
            chunk = todo[s:s + batch]
            tmp = _private_tmp()
            try:
                def one(item):
                    vid, path = item
                    sub = Path(tmp) / vid
                    sub.mkdir()
                    return vid, extract_frames(path, sub)
                extracted = list(pool.map(one, chunk))
                images, owner, mouths, mowner, good = [], [], [], [], []
                for vid, paths in extracted:
                    if not paths:
                        continue
                    vi = len(good)
                    good.append(vid)
                    for fi, p in enumerate(paths):
                        with Image.open(p) as im:
                            img = im.convert("RGB")
                        images.append(img)
                        owner.append((vi, fi))
                        if mouth:
                            crop = lower_face_crop(img)
                            if crop is not None:
                                mouths.append(crop)
                                mowner.append(vi)
                bad = [vid for vid, paths in extracted if not paths]
                if bad:
                    store.mark_failed(bad)
                    failed += len(bad)
                if not good:
                    continue
                emb = embedder.encode(images)
                d = emb.shape[1]
                frames = np.zeros((len(good), K, d), np.float32)
                mask = np.zeros((len(good), K), bool)
                for (vi, fi), e in zip(owner, emb):
                    frames[vi, fi] = e
                    mask[vi, fi] = True
                mo = np.zeros((len(good), d), np.float32)
                hm = np.zeros(len(good), bool)
                if mouths:
                    me = embedder.encode(mouths)
                    for vi in sorted(set(mowner)):
                        sel = [j for j, o in enumerate(mowner) if o == vi]
                        v = me[sel].mean(0)
                        mo[vi] = v / (np.linalg.norm(v) + 1e-9)
                        hm[vi] = True
                store.add(good, frames, mask, mo, hm)
                done += len(good)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)         # frames never outlive their batch
            el = max(time.time() - t0, 1e-6)
            if on_progress:
                try:
                    on_progress({"session_done": done, "session_failed": failed, "session_target": len(todo), "elapsed": el})
                except Exception:
                    pass                                        # progress reporting must never stop the scan
            log(f"  frames {done}/{len(todo)}  {done / el:.1f} videos/s  eta {(len(todo) - done - failed) / max(done / el, 1e-6) / 60:.1f} min")
    return {"done": done, "failed": failed, "seconds": time.time() - t0}


def coverage(cover_ids, store=None, videos=None):
    """(done, total): videos with frames (or given up on) over videos that exist, among the embedded covers."""
    store = store or FrameStore()
    videos = videos if videos is not None else find_videos()
    ids = [v for v in cover_ids if v in videos]
    have = store.done_ids() | store.skipped_ids()
    return sum(1 for v in ids if v in have), len(ids)


def write_progress(mode, total, covered, session_done=0, session_target=0, rate=0.0, running=True):
    """Small atomic file the Starplayer server reads, so any device (and any page reload) shows the same live progress."""
    write_json_atomic(PROGRESS, {"mode": mode, "total": total, "covered": covered, "session_done": session_done,
                                 "session_target": session_target, "rate": round(rate, 4), "running": running, "ts": time.time()})


def write_stats(done, total, rate, mode):
    write_json_atomic(STATS, {"done": done, "total": total, "rate": rate, "mode": mode, "ts": time.time()})
