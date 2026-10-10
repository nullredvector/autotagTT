import hashlib, json, os, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from .archive import OUT, APPDATA, load_groups, write_json_atomic, label

FEEDBACK = APPDATA / "autotag_feedback.json"   # written by kashibox: {tag: {accepted:[], rejected:[]}}
MIN_POSITIVES = 30                              # fewer reviewed/known positives than this: no model for the tag
MIN_AUC = 0.80                                  # weaker tags are reported but never shown or auto-tagged
AI_PRECISION = 0.80                             # AI tags are applied only above the score where reviewed precision >= this
AI_MIN_REVIEWED = 30
AI_WINDOW = 50                                  # precision is judged on the most recent N reviewed videos going down the ranking
AI_UNVALIDATED_TOP = 100                        # no usable reviews yet: only the top N per tag
CACHE = OUT / "cache"                           # per-tag scores + weights (private, hashed names)
CODE_VERSION = 3                                # bump when scoring logic changes: invalidates every cache
MAX_AGE_S = 14 * 86400                          # retrain at least every two weeks even if nothing changed
NEW_FRACTION = 0.05                             # more new videos than this share of the scored set: retrain instead
MIN_FEATURE_GAIN = 0.01                         # AP a richer feature set must add before it replaces cover-only
MIN_MOUTH_SHARE = 0.05                          # share of videos needing a lower-face crop before the 'mouth' set is considered
COVERAGE_READY = 0.95                           # frame features are used once this share of videos has them
TRIAL_IDS = OUT / "frames_trial_ids.json"
TRIAL_REPORT = OUT / "frames_trial.json"
DUP_SIM = 0.97                                  # cosine similarity treated as the same image


def load_feedback(tag):
    """(accepted, rejected, unsure). Unsure = reviewed but undecided: left out of training entirely."""
    try:
        fb = json.loads(FEEDBACK.read_text()).get(tag, {})
    except Exception:
        fb = {}
    return set(fb.get("accepted", [])), set(fb.get("rejected", [])), set(fb.get("unsure", []))


def dup_groups(X, thr=DUP_SIM, chunk=2048):
    """Cluster near-duplicate embeddings (union-find) so CV folds never split them."""
    n = len(X)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    Xf = X.astype(np.float32)
    for s in range(0, n, chunk):
        sims = Xf[s:s + chunk] @ Xf.T
        for i, j in zip(*np.nonzero(sims > thr)):
            i += s
            if i < j:
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    return np.array([find(i) for i in range(n)])


def _fit(X, y, w, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=300)
    clf.fit(sc.transform(X), y, sample_weight=w)
    return sc, clf


def _score(sc, clf, X):
    return clf.decision_function(sc.transform(X))


def _boot_ci(ypos_sorted, fn, reps=200, seed=0):
    """95% percentile bootstrap CI over positives/negatives resampled with replacement."""
    rng = np.random.default_rng(seed)
    n = len(ypos_sorted)
    vals = []
    for _ in range(reps):
        idx = np.sort(rng.integers(0, n, n))   # resampling keeps rank order
        vals.append(fn(ypos_sorted[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


class FrameFeatures:
    """Pooled frame features aligned by video id. Videos without frames get zeros (and a 0 flag)."""

    def __init__(self, ids, F, cover_ids, coverage):
        self.index = {v: i for i, v in enumerate(ids)}
        self.F = F
        self.d = (F.shape[1] - 1) // 3 if len(ids) else 0
        self.coverage = coverage
        self.ready = len(ids) > 0 and coverage >= COVERAGE_READY
        self.mouth_ready = self.ready and float(F[:, -1].astype(np.float32).mean()) >= MIN_MOUTH_SHARE

    def block(self, ids, kind):
        cols = 2 * self.d if kind == "frames" else self.F.shape[1]
        out = np.zeros((len(ids), cols), np.float32)
        for r, v in enumerate(ids):
            i = self.index.get(v)
            if i is not None:
                out[r] = self.F[i, :cols]
        return out

    def kinds(self):
        return (["frames"] if self.ready else []) + (["mouth"] if self.mouth_ready else [])

    def signature(self):
        return "cover" + ("+frames" if self.ready else "") + ("+mouth" if self.mouth_ready else "")


def _features(kind, ids, X, ff):
    return X if kind == "cover" else np.hstack([X, ff.block(ids, kind)])


def _cv(Xf, y, w, splits, C):
    oof = np.zeros(len(y))
    for tr, te in splits:
        sc, clf = _fit(Xf[tr], y[tr], w[tr], C)
        oof[te] = _score(sc, clf, Xf[te])
    return oof


def _linear(sc, clf):
    """Fold the scaler into the weights: score = X @ w + b, so new videos can be scored without refitting."""
    w = clf.coef_[0] / sc.scale_
    return w.astype(np.float32), float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())


# ───────────── per-tag cache (scores + weights), so unchanged tags are not retrained ─────────────
def _tagdir(tag):
    return CACHE / hashlib.sha1(tag.encode()).hexdigest()[:12]


def _fingerprint(model, signature, pos_ids, rejected, unsure):
    h = hashlib.sha1()
    h.update(f"{CODE_VERSION}|{model}|{signature}".encode())
    for name, group in (("p", pos_ids), ("r", rejected), ("u", unsure)):
        h.update(name.encode() + ",".join(sorted(group)).encode())
    return h.hexdigest()


def _cache_save(tag, meta, ids=None, scores=None, w=None, b=None):
    d = _tagdir(tag)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    if ids is not None:
        write_json_atomic(d / "ids.json", ids)
        for name, arr in (("scores", scores.astype(np.float32)), ("w", w)):
            tmp = d / f"{name}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, d / f"{name}.npy")
    write_json_atomic(d / "meta.json", dict(meta, b=b))     # meta last = commit marker


def _cache_load(tag):
    d = _tagdir(tag)
    try:
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("status") != "model":
            return meta, None
        return meta, {"ids": json.loads((d / "ids.json").read_text()), "scores": np.load(d / "scores.npy"), "w": np.load(d / "w.npy")}
    except Exception:
        return None, None


def _outputs(tag, ids, scores, pos_ids, accepted, rejected, unsure, metrics, top):
    """Suggestions + AI tags from scores and the CURRENT labels (cheap; identical for a fresh or a reused model)."""
    y = np.array([1 if v in pos_ids else 0 for v in ids])
    order = np.argsort(-scores)
    sugg, seen = [], set()
    for i in order:
        v = ids[i]
        if y[i] or v in rejected or v in unsure or v in seen:
            continue
        seen.add(v)
        sugg.append({"id": v, "score": round(float(scores[i]), 4), "status": "pending"})
        if len(sugg) >= top:
            break
    if metrics["gate"] == "weak":
        sugg = []
        ai = {"cutoff": None, "mode": "off", "items": {}}
    else:
        ai = ai_tags(ids, scores, y, accepted, rejected, unsure)
    metrics["ai_mode"], metrics["ai_cutoff"], metrics["ai_count"] = ai["mode"], ai["cutoff"], len(ai["items"])
    print(f"  ai tags: mode={ai['mode']} cutoff={ai['cutoff']} count={len(ai['items'])}", flush=True)
    return {"suggestions": sugg, "metrics": metrics, "ai": ai["items"]}


def train_tag(tag, ids, X, top=300, folds=5, cs=(0.001, 0.01, 0.1), seed=0, grp=None, ff=None, model="", force=False):
    """Train (or reuse) one tag. Returns None (skipped) or {"suggestions","metrics","ai","mode"} where mode is
    trained | incremental | reused. ff: FrameFeatures or None. grp: duplicate groups aligned with ids (computed once per run)."""
    if len(ids) == 0:
        raise SystemExit("no embeddings; run `embed` first")
    groups = load_groups()
    if tag not in groups:
        print(f"[{label(tag)}] no such group in stars.json; skipped", flush=True)
        return None
    accepted, rejected, unsure = load_feedback(tag)
    unsure -= groups[tag] | accepted | rejected      # an explicit label always wins over "not sure"
    pos_ids = groups[tag] | accepted
    signature = ff.signature() if ff is not None else "cover"
    fp = _fingerprint(model, signature, pos_ids, rejected, unsure)

    # unsure videos are left out of training and scoring entirely
    id_set = set(ids)
    keep = [i for i, v in enumerate(ids) if v not in unsure]
    n_unsure = len(unsure & id_set)
    cur_ids = [ids[i] for i in keep]

    meta, cache = (None, None) if force else _cache_load(tag)
    fresh = meta is not None and meta.get("fp") == fp and time.time() - meta.get("trained_at", 0) < MAX_AGE_S
    if fresh and meta.get("status") == "skipped":
        print(f"[{label(tag)}] unchanged; skipped (needs {MIN_POSITIVES} positives)", flush=True)
        return None
    if fresh and cache is not None:
        have = set(cache["ids"])
        new_idx = [i for i, v in enumerate(cur_ids) if v not in have]
        few = len(new_idx) <= max(200, NEW_FRACTION * len(cache["ids"]))
        if meta["metrics"]["gate"] == "weak" or few:
            return _reuse(tag, meta, cache, cur_ids, X[keep] if len(new_idx) else None, ff, new_idx, pos_ids, accepted, rejected, unsure, top)

    # ── full training ──
    X = X[keep].astype(np.float32)
    ids = cur_ids
    idx = {v: i for i, v in enumerate(ids)}
    y = np.array([1 if v in pos_ids else 0 for v in ids])
    if y.sum() < max(folds, MIN_POSITIVES):
        print(f"[{label(tag)}] only {int(y.sum())} embedded positives (need {MIN_POSITIVES}); skipped", flush=True)
        _cache_save(tag, {"status": "skipped", "fp": fp, "trained_at": time.time()})
        return None
    w = np.ones(len(ids))
    for v in rejected:           # reviewed-and-rejected = confirmed negative, count more
        if v in idx and not y[idx[v]]:
            w[idx[v]] = 5.0
    n_rej = len(rejected & set(idx))
    print(f"[{label(tag)}] embedded={len(ids)} positives={y.sum()} reviewed_rejected={n_rej}", flush=True)

    grp = grp() if callable(grp) else grp
    grp = dup_groups(X) if grp is None else grp[keep]
    skf = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y, grp))

    feat, aps = "cover", {}
    if ff is not None and ff.kinds():
        quick = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, grp))
        for kind in ["cover"] + ff.kinds():
            aps[kind] = float(average_precision_score(y, _cv(_features(kind, ids, X, ff), y, w, quick, cs[0])))
        for kind in ff.kinds():                       # a richer feature set must clearly beat the simpler one
            if aps[kind] > aps[feat] + MIN_FEATURE_GAIN:
                feat = kind
        print(f"  features: {feat} (quick AP { {k: round(v, 3) for k, v in aps.items()} })", flush=True)
    Xf = _features(feat, ids, X, ff) if feat != "cover" else X

    best = None
    for C in cs:
        oof = _cv(Xf, y, w, splits, C)
        ap = average_precision_score(y, oof)
        print(f"  C={C}: AP={ap:.3f} AUC={roc_auc_score(y, oof):.3f}", flush=True)
        if best is None or ap > best[0]:
            best = (ap, C, oof)
    ap, C, oof = best

    order = np.argsort(-oof)
    ypos = y[order]
    P = int(y.sum())
    # NOTE: "unlabeled" = not in the group. Many are probably true positives the user hasn't
    # tagged, so every precision figure below is a LOWER bound on real precision.
    metrics = {"C": C, "ap": float(ap), "ap_ci95": _boot_ci(ypos, lambda v: average_precision_score(v, -np.arange(len(v)))) if P else None,
               "auc": float(roc_auc_score(y, oof)), "positives": P, "n": len(ids),
               "reviewed_rejected": n_rej, "excluded_unsure": n_unsure, "reviewed_accepted": len(accepted & set(idx)),
               "features": feat, "feature_aps": {k: round(v, 4) for k, v in aps.items()}}
    for frac in (0.01, 0.02, 0.05, 0.10):
        k = max(1, int(len(ids) * frac))
        metrics[f"recall@top{frac:.0%}"] = float(ypos[:k].sum() / P)
    for k in (50, 100, 200, 500):
        if k <= len(ids):
            metrics[f"known_precision@{k}"] = float(ypos[:k].mean())
    metrics["gate"] = "ok" if metrics["auc"] >= MIN_AUC else "weak"
    print("  " + json.dumps(metrics), flush=True)
    if metrics["gate"] == "weak":
        print(f"  gate: weak (AUC < {MIN_AUC}); no suggestions or AI tags for this tag", flush=True)

    sc, clf = _fit(Xf, y, w, C)
    wt, b = _linear(sc, clf)
    _cache_save(tag, {"status": "model", "fp": fp, "trained_at": time.time(), "features": feat, "metrics": metrics}, ids, oof, wt, b)
    out = _outputs(tag, ids, oof, pos_ids, accepted, rejected, unsure, metrics, top)
    out["mode"] = "trained"
    return out


def _reuse(tag, meta, cache, cur_ids, Xcur, ff, new_idx, pos_ids, accepted, rejected, unsure, top):
    """Same labels as last time: keep the saved scores and only score videos added since."""
    metrics = dict(meta["metrics"])
    present = set(cur_ids)
    ids = [v for v in cache["ids"] if v in present]
    keep_rows = [i for i, v in enumerate(cache["ids"]) if v in present]
    scores = cache["scores"][keep_rows]
    n_new = 0
    if metrics["gate"] != "weak" and new_idx:
        feat = meta["features"]
        Xn = Xcur[new_idx].astype(np.float32)
        nids = [cur_ids[i] for i in new_idx]
        if feat != "cover":
            have = [j for j, v in enumerate(nids) if v in ff.index]        # score them once their frames exist
            nids, Xn = [nids[j] for j in have], Xn[have]
            Xn = _features(feat, nids, Xn, ff) if len(nids) else Xn
        if nids:
            ns = Xn @ cache["w"] + meta["b"]
            ids = ids + nids
            scores = np.concatenate([scores, ns.astype(np.float32)])
            n_new = len(nids)
    metrics["n"] = len(ids)
    metrics["incremental_new"] = n_new
    print(f"[{label(tag)}] reused saved model; {n_new} new videos scored", flush=True)
    out = _outputs(tag, ids, scores, pos_ids, accepted, rejected, unsure, metrics, top)
    out["mode"] = "incremental" if n_new else "reused"
    return out


def ai_tags(ids, oof, y, accepted, rejected, unsure):
    """Videos to label with an AI tag (never in a human group, never already reviewed).
    Walking down the ranking, the cutoff is the last reviewed video before the trailing-window precision
    (last AI_WINDOW decisions) drops below AI_PRECISION; needs AI_MIN_REVIEWED decisions. Without enough
    reviews only the top AI_UNVALIDATED_TOP are tagged."""
    order = np.argsort(-oof)
    recent, seen, cutoff = [], 0, None
    for i in order:
        v = ids[i]
        if v in accepted or v in rejected:
            recent.append(v in accepted)
            recent = recent[-AI_WINDOW:]
            seen += 1
            if seen >= AI_MIN_REVIEWED:
                if sum(recent) / len(recent) < AI_PRECISION:
                    break                      # precision of the latest decisions fell below target: stop here
                cutoff = float(oof[i])
    mode = "reviewed"
    if cutoff is None:
        mode = "unvalidated"
    items, count = {}, 0
    for i in order:
        v = ids[i]
        if y[i] or v in accepted or v in rejected or v in unsure:
            continue
        if mode == "reviewed" and oof[i] < cutoff:
            break
        if mode == "unvalidated" and count >= AI_UNVALIDATED_TOP:
            break
        items[v] = round(float(oof[i]), 4)
        count += 1
    return {"cutoff": None if cutoff is None else round(cutoff, 4), "mode": mode, "items": items}


def write_ai_tags(results, model_name):
    """ai_tags.json: {"videos": {id: [[tag, score], ...]}} sorted best-first. kashibox reads it; it
    is separate from stars.json so human groups are never touched."""
    videos = {}
    for tag, r in results.items():
        for v, sc in r.get("ai", {}).items():
            videos.setdefault(v, []).append([tag, sc])
    for v in videos:
        videos[v].sort(key=lambda t: -t[1])
    out = {"_meta": {"model": model_name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "tags": sorted(t for t, r in results.items() if r.get("ai")),
                     "note": "AI labels, uncalibrated scores; not human tags"},
           "videos": videos}
    write_json_atomic(OUT / "ai_tags.json", out, backup=True)
    print(f"wrote ai_tags.json ({len(videos)} videos)")


def write_suggestions(results, model_name, pretrained=None):
    """score = logistic-regression decision value (log-odds on class-balanced data).
    Higher = more similar to the tagged examples. NOT a calibrated probability."""
    path = OUT / "autotag.json"
    out = {"_meta": {"model": model_name, "pretrained": pretrained,
                     "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "score": "oof logistic decision value; uncalibrated"}}
    for tag, r in results.items():
        out[tag] = r["suggestions"]
        out["_meta"].setdefault("metrics", {})[tag] = r["metrics"]
    write_json_atomic(path, out, backup=True)
    print(f"wrote {path.name}")


def select_trial(video_ids, tags, per_tag=150, random_extra=1500, seed=0):
    """Videos for the frame trial: some positives of every enabled tag plus random others. Reused/extended across runs."""
    import random
    rng = random.Random(seed)
    groups = load_groups()
    vset = set(video_ids)
    chosen = set()
    try:
        chosen |= set(json.loads(TRIAL_IDS.read_text()))
    except Exception:
        pass
    for tag in tags:
        acc, _, _ = load_feedback(tag)
        pos = sorted((groups.get(tag, set()) | acc) & vset)
        rng.shuffle(pos)
        chosen |= set(pos[:per_tag])
    others = sorted(vset - chosen)
    rng.shuffle(others)
    if len(chosen) < per_tag * max(1, len(tags)) + random_extra:
        chosen |= set(others[:random_extra])
    write_json_atomic(TRIAL_IDS, sorted(chosen))
    return chosen


def trial_report(tags, ids, X, ff, trial_ids):
    """Does adding frames (and lower-face crops) help? Compares cross-validated AP on the trial subset only; the real
    outputs are not affected."""
    from sklearn.model_selection import StratifiedKFold
    sel = [i for i, v in enumerate(ids) if v in trial_ids and v in ff.index]
    out = {}
    if len(sel) < 200:
        print(f"trial: only {len(sel)} videos have frames; not enough for a comparison", flush=True)
        write_json_atomic(TRIAL_REPORT, out)
        return out
    sids = [ids[i] for i in sel]
    Xs = X[sel].astype(np.float32)
    groups = load_groups()
    flag = ff.block(sids, "mouth")[:, -1]
    mouth_ok = float(flag.mean()) >= MIN_MOUTH_SHARE
    for tag in tags:
        acc, _, _ = load_feedback(tag)
        y = np.array([1 if v in (groups.get(tag, set()) | acc) else 0 for v in sids])
        if y.sum() < 15 or y.sum() > len(y) - 15:
            continue
        splits = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Xs, y))
        w = np.ones(len(y))
        row = {"n_pos": int(y.sum()), "n": len(y)}
        for kind in ["cover", "frames"] + (["mouth"] if mouth_ok else []):
            Xk = Xs if kind == "cover" else np.hstack([Xs, ff.block(sids, kind)])
            row["ap_" + kind] = round(float(average_precision_score(y, _cv(Xk, y, w, splits, 0.001))), 4)
        out[tag] = row
        print(f"[{label(tag)}] trial AP: " + " ".join(f"{k[3:]}={v}" for k, v in row.items() if k.startswith("ap_")), flush=True)
    write_json_atomic(TRIAL_REPORT, out)
    return out
