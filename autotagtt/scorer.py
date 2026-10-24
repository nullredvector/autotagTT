"""The scorer's model files and maths. `train` saves, per usable tag, a cover-only classifier (so a bare cover image can be
scored) with its calibration and weight under data/.appdata/autotag/models/. `serve` loads them and answers
"how well does this cover fit the tags I taught you?".

score = 1 - prod(1 - w_t * p_t)            (noisy OR: one strong match is enough, several add up)
p_t   = calibrated probability for tag t   (Platt on out-of-fold values vs the user's accepts/rejects when there are enough,
                                            otherwise the class-balanced logistic output; both are optimistic because the
                                            negatives are provisional, see the README caveat)
w_t   = the tag's reviewed precision (or its cross-validated precision when there are few reviews); tags below 0.3 are dropped
Private files: owner-only, and they contain tag names, so never print or commit them."""
import hashlib, json, os, threading, time
import numpy as np
from .archive import OUT, write_json_atomic

MODELS = OUT / "models"
MANIFEST = "manifest.json"
MIN_WEIGHT = 0.3
CAL_MIN = 20            # reviewed positives AND reviewed negatives needed to calibrate a tag
REVIEWED_MIN = 30       # reviewed decisions needed to weight a tag by its reviewed precision
TOP_TAGS = 3


def tag_key(tag):
    return hashlib.sha1(tag.encode()).hexdigest()[:12]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def platt_fit(d_pos, d_neg):
    """(a, c) with p = sigmoid(a*d + c), fitted on out-of-fold decision values of the reviewed items; None if not monotone."""
    from sklearn.linear_model import LogisticRegression
    d = np.concatenate([np.asarray(d_pos, float), np.asarray(d_neg, float)]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(d_pos)), np.zeros(len(d_neg))])
    lr = LogisticRegression(C=1e4, max_iter=500).fit(d, y)
    a, c = float(lr.coef_[0][0]), float(lr.intercept_[0])
    return {"a": a, "c": c} if a > 0 else None


def tag_weight(metrics, n_acc, n_rej):
    """(weight, source): reviewed precision with enough reviews, else the cross-validated precision at the top."""
    n = n_acc + n_rej
    if n >= REVIEWED_MIN:
        return float(n_acc / n), "reviewed"
    cv = metrics.get("known_precision@100", metrics.get("known_precision@50", metrics.get("ap", 0.0)))
    return float(min(max(cv, 0.0), 1.0)), "cv"


# ───────────── writing (called by train) ─────────────
def has_model(tag, models_dir=None):
    d = models_dir or MODELS
    k = tag_key(tag)
    return (d / f"{k}.npz").exists() and (d / f"{k}.json").exists()


def save_tag_model(tag, w, b, calib, weight, source, info=None, models_dir=None):
    d = models_dir or MODELS
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    k = tag_key(tag)
    tmp = d / f"{k}.tmp.npz"
    np.savez(tmp, w=np.asarray(w, np.float32), b=np.float64(b))
    os.replace(tmp, d / f"{k}.npz")
    os.chmod(d / f"{k}.npz", 0o600)
    write_json_atomic(d / f"{k}.json", {"tag": tag, "weight": round(float(weight), 4), "weight_source": source, "calib": calib,
                                        "info": info or {}, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")})


def write_manifest(tags, models_dir=None):
    """Lists the tags the service should use. Written last (commit marker). `model` changes only when the content changes."""
    d = models_dir or MODELS
    entries = []
    for tag in sorted(tags):
        k = tag_key(tag)
        try:
            meta = json.loads((d / f"{k}.json").read_text())
            blob = (d / f"{k}.npz").read_bytes()
        except Exception:
            continue
        if meta.get("weight", 0) < MIN_WEIGHT:
            continue
        entries.append({"key": k, "tag": tag, "weight": meta["weight"], "calib": meta.get("calib"),
                        "sha": hashlib.sha1(blob).hexdigest()[:10]})
    h = hashlib.sha1(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:6]
    model_id = None
    try:
        old = json.loads((d / MANIFEST).read_text())
        if old.get("hash") == h:
            model_id = old.get("model")                   # same content: keep the id (and the date in it)
    except Exception:
        pass
    model_id = model_id or time.strftime("%Y-%m-%d") + "-" + h
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json_atomic(d / MANIFEST, {"model": model_id, "hash": h, "tags": entries})
    return model_id, len(entries)


# ───────────── reading (the service) ─────────────
class ModelSet:
    """Loads the manifest and every listed tag's classifier; reloads itself when the manifest changes (no restart needed)."""

    def __init__(self, models_dir=None):
        self.dir = models_dir or MODELS
        self.lock = threading.Lock()
        self.tags, self.model_id, self._stamp = [], None, None

    def _stat(self):
        try:
            s = (self.dir / MANIFEST).stat()
            return (s.st_mtime_ns, s.st_size)
        except OSError:
            return None

    def load(self):
        stamp = self._stat()
        if stamp is None:
            with self.lock:
                self.tags, self.model_id, self._stamp = [], None, None
            return False
        man = json.loads((self.dir / MANIFEST).read_text())
        tags = []
        for e in man.get("tags", []):
            try:
                z = np.load(self.dir / f"{e['key']}.npz")
                tags.append({"tag": e["tag"], "weight": float(e["weight"]), "calib": e.get("calib"),
                             "w": z["w"].astype(np.float32), "b": float(z["b"])})
            except Exception:
                continue                                   # a half-written tag is skipped; the next manifest brings it back
        with self.lock:
            self.tags, self.model_id, self._stamp = tags, man.get("model"), stamp
        return True

    def maybe_reload(self):
        if self._stat() != self._stamp:
            try:
                self.load()
            except Exception:
                pass                                       # keep serving the previous model

    def score(self, emb):
        """(score 0..1, [[tag, w*p], ... best three], {tag: p}) for one L2-normalised embedding."""
        emb = np.asarray(emb, np.float32).reshape(-1)
        with self.lock:
            tags = list(self.tags)
        contrib, probs = [], {}
        for t in tags:
            if t["w"].shape[0] != emb.shape[0]:
                continue
            d = float(emb @ t["w"] + t["b"])
            cal = t["calib"]
            p = float(sigmoid(cal["a"] * d + cal["c"])) if cal else float(sigmoid(d))
            probs[t["tag"]] = round(p, 4)
            contrib.append((t["tag"], t["weight"] * p))
        score = 1.0 - float(np.prod([1.0 - c for _, c in contrib])) if contrib else 0.0
        best = sorted(contrib, key=lambda x: -x[1])[:TOP_TAGS]
        return round(score, 4), [[n, round(c, 4)] for n, c in best], probs
