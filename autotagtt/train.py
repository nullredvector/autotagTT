import json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from .archive import OUT, APPDATA, load_groups, write_json_atomic, label

FEEDBACK = APPDATA / "autotag_feedback.json"   # written by Starplayer: {tag: {accepted:[], rejected:[]}}
MIN_POSITIVES = 30                              # fewer reviewed/known positives than this: no model for the tag
MIN_AUC = 0.80                                  # weaker tags are reported but never shown or auto-tagged
AI_PRECISION = 0.80                             # AI tags are applied only above the score where reviewed precision >= this
AI_MIN_REVIEWED = 30
AI_WINDOW = 50                                  # precision is judged on the most recent N reviewed videos going down the ranking
AI_UNVALIDATED_TOP = 100                        # no usable reviews yet: only the top N per tag
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


def train_tag(tag, ids, X, top=300, folds=5, cs=(0.001, 0.01, 0.1), seed=0):
    if len(ids) == 0:
        raise SystemExit("no embeddings; run `embed` first")
    X = X.astype(np.float32)
    groups = load_groups()
    if tag not in groups:
        print(f"[{label(tag)}] no such group in stars.json; skipped", flush=True)
        return None
    accepted, rejected, unsure = load_feedback(tag)
    unsure -= groups[tag] | accepted | rejected      # an explicit label always wins over "not sure"
    n_unsure = len(unsure & set(ids))
    if n_unsure:
        keep = [i for i, v in enumerate(ids) if v not in unsure]
        ids, X = [ids[i] for i in keep], X[keep]
    idx = {v: i for i, v in enumerate(ids)}
    pos_ids = groups[tag] | accepted
    y = np.array([1 if v in pos_ids else 0 for v in ids])
    if y.sum() < max(folds, MIN_POSITIVES):
        print(f"[{label(tag)}] only {int(y.sum())} embedded positives (need {MIN_POSITIVES}); skipped", flush=True)
        return None
    w = np.ones(len(ids))
    for v in rejected:           # reviewed-and-rejected = confirmed negative, count more
        if v in idx and not y[idx[v]]:
            w[idx[v]] = 5.0
    n_rej = len(rejected & set(idx))
    print(f"[{label(tag)}] embedded={len(ids)} positives={y.sum()} reviewed_rejected={n_rej}", flush=True)

    grp = dup_groups(X)
    print(f"  near-duplicate groups: {len(np.unique(grp))} for {len(ids)} images", flush=True)
    skf = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y, grp))
    best = None
    for C in cs:
        oof = np.zeros(len(ids))
        for tr, te in splits:
            sc, clf = _fit(X[tr], y[tr], w[tr], C)
            oof[te] = _score(sc, clf, X[te])
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
               "reviewed_rejected": n_rej, "excluded_unsure": n_unsure, "reviewed_accepted": len(accepted & set(idx))}
    for frac in (0.01, 0.02, 0.05, 0.10):
        k = max(1, int(len(ids) * frac))
        metrics[f"recall@top{frac:.0%}"] = float(ypos[:k].sum() / P)
    for k in (50, 100, 200, 500):
        if k <= len(ids):
            metrics[f"known_precision@{k}"] = float(ypos[:k].mean())
    print("  " + json.dumps(metrics), flush=True)

    # Rank suggestions by out-of-fold score: each item is scored by a model that never trained on
    # it (or its near-duplicates), so provisional-negative memorisation does not suppress it.
    sugg, seen = [], set()
    for i in order:
        v = ids[i]
        if y[i] or v in rejected or v in seen:
            continue
        seen.add(v)
        sugg.append({"id": v, "score": round(float(oof[i]), 4), "status": "pending"})
        if len(sugg) >= top:
            break
    metrics["gate"] = "ok" if metrics["auc"] >= MIN_AUC else "weak"
    if metrics["gate"] == "weak":
        print(f"  gate: weak (AUC < {MIN_AUC}); no suggestions or AI tags for this tag", flush=True)
        sugg = []
    sc, clf = _fit(X, y, w, C)
    ai = ai_tags(ids, oof, y, accepted, rejected, unsure) if metrics["gate"] == "ok" else {"cutoff": None, "mode": "off", "items": {}}
    metrics["ai_mode"], metrics["ai_cutoff"], metrics["ai_count"] = ai["mode"], ai["cutoff"], len(ai["items"])
    print(f"  ai tags: mode={ai['mode']} cutoff={ai['cutoff']} count={len(ai['items'])}", flush=True)
    return {"suggestions": sugg, "metrics": metrics, "model": (sc, clf), "ai": ai["items"]}


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
    """ai_tags.json: {"videos": {id: [[tag, score], ...]}} sorted best-first. Starplayer reads it; it
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
