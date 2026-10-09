import json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from .archive import OUT, APPDATA, load_groups, write_json_atomic

FEEDBACK = APPDATA / "autotag_feedback.json"   # written by Starplayer: {tag: {accepted:[], rejected:[]}}


def load_feedback(tag):
    try:
        fb = json.loads(FEEDBACK.read_text()).get(tag, {})
    except Exception:
        fb = {}
    return set(fb.get("accepted", [])), set(fb.get("rejected", []))


def _fit(X, y, w, C):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=300)
    clf.fit(sc.transform(X), y, sample_weight=w)
    return sc, clf


def _score(sc, clf, X):
    return clf.decision_function(sc.transform(X))


def train_tag(tag, ids, X, top=300, folds=5, cs=(0.001, 0.01, 0.1), seed=0):
    X = X.astype(np.float32)
    groups = load_groups()
    if tag not in groups:
        raise SystemExit(f"no group named {tag!r}; have {sorted(groups)[:20]}")
    accepted, rejected = load_feedback(tag)
    pos_ids = groups[tag] | accepted
    idx = {v: i for i, v in enumerate(ids)}
    y = np.array([1 if v in pos_ids else 0 for v in ids])
    w = np.ones(len(ids))
    for v in rejected:           # reviewed-and-rejected = confirmed negative, count more
        if v in idx:
            w[idx[v]] = 5.0
    print(f"[{tag}] embedded={len(ids)} positives={y.sum()} rejected={len(rejected & set(idx))}", flush=True)

    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    best = None
    for C in cs:
        oof = np.zeros(len(ids))
        for tr, te in skf.split(X, y):
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
    metrics = {"C": C, "ap": float(ap), "auc": float(roc_auc_score(y, oof)), "positives": P, "n": len(ids)}
    # recall of known positives when reviewing only the top x% / top K of everything
    for frac in (0.01, 0.02, 0.05, 0.10):
        k = int(len(ids) * frac)
        metrics[f"recall@top{frac:.0%}"] = float(ypos[:k].sum() / P)
    for k in (50, 100, 200, 500):
        metrics[f"labeled_precision@{k}"] = float(ypos[:k].mean())
    print("  " + json.dumps(metrics), flush=True)

    sc, clf = _fit(X, y, w, C)
    # Unlabeled items were training negatives in the final fit, so rank suggestions by
    # out-of-fold score (each scored by a model that never saw it as a negative).
    sugg = []
    for i in order:
        v = ids[i]
        if y[i] or v in rejected:
            continue
        sugg.append({"id": v, "score": round(float(oof[i]), 4), "status": "pending"})
        if len(sugg) >= top:
            break
    return {"suggestions": sugg, "metrics": metrics, "model": (sc, clf)}


def write_suggestions(results, model_name):
    path = OUT / "autotag.json"
    out = {"_meta": {"model": model_name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    for tag, r in results.items():
        out[tag] = r["suggestions"]
        out["_meta"].setdefault("metrics", {})[tag] = r["metrics"]
    write_json_atomic(path, out)
    print(f"wrote {path}")
