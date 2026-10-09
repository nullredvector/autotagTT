"""Synthetic-data tests (random vectors, fake IDs). No real archive, no GPU."""
import base64, gzip, importlib, json, os, stat
import numpy as np
import pytest


@pytest.fixture
def arch(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path))
    (tmp_path / "data/.appdata").mkdir(parents=True)
    for m in ("autotagtt.archive", "autotagtt.store", "autotagtt.train"):
        importlib.reload(importlib.import_module(m))
    return tmp_path


def make_data(n=600, npos=60, d=32, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"v{i:05d}" for i in range(n)]
    X = rng.normal(size=(n, d)).astype(np.float32)
    direction = rng.normal(size=d)
    X[:npos] += 1.5 * direction
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return ids, X, set(ids[:npos])


def write_stars(root, pos, name="Braces"):
    (root / "data/.appdata/stars.json").write_text(json.dumps(
        {"stars": [], "levels": [], "groups": [{"id": "g1", "name": name, "videoIds": sorted(pos)}]}))


def test_read_db_roundtrip(arch):
    from autotagtt import archive
    blob = base64.b64encode(gzip.compress(json.dumps({"a": 1}).encode())).decode()
    (arch / "data/.appdata/db_x.js").write_text(f'window.db_x_base64="{blob}";')
    assert archive.read_db("x") == {"a": 1}


def test_group_match_case_insensitive_and_never_writes_stars(arch):
    from autotagtt import archive
    write_stars(arch, {"a", "b"})
    p = arch / "data/.appdata/stars.json"
    before = p.read_bytes()
    assert archive.load_groups()["braces"] == {"a", "b"}
    assert p.read_bytes() == before


def test_train_ranks_hidden_positives_high_and_excludes_known(arch):
    from autotagtt import train
    ids, X, pos = make_data()
    known = set(sorted(pos)[:45])           # 15 true positives left "unlabeled"
    write_stars(arch, known)
    r = train.train_tag("braces", ids, X, top=50)
    s = r["suggestions"]
    assert all(x["id"] not in known and x["status"] == "pending" for x in s)
    assert len({x["id"] for x in s}) == len(s)
    hidden = pos - known
    assert len([x for x in s[:30] if x["id"] in hidden]) >= 8     # hidden positives surface at top
    assert r["metrics"]["ap"] > 0.5 and r["metrics"]["ap_ci95"][0] <= r["metrics"]["ap"] <= r["metrics"]["ap_ci95"][1] + 1e-9


def test_rejected_excluded_and_accepted_become_positive(arch):
    from autotagtt import train
    ids, X, pos = make_data()
    known = set(sorted(pos)[:45])
    write_stars(arch, known)
    first = train.train_tag("braces", ids, X, top=20)["suggestions"]
    rej, acc = first[0]["id"], first[1]["id"]
    (arch / "data/.appdata/autotag_feedback.json").write_text(json.dumps({"braces": {"accepted": [acc], "rejected": [rej]}}))
    r = train.train_tag("braces", ids, X, top=300)
    got = {x["id"] for x in r["suggestions"]}
    assert rej not in got and acc not in got
    assert r["metrics"]["reviewed_rejected"] == 1 and r["metrics"]["positives"] == 46


def test_duplicates_never_split_across_folds(arch):
    from autotagtt.train import dup_groups
    ids, X, _ = make_data(n=100)
    X2 = np.concatenate([X, X[:10] + 1e-4])      # 10 near-duplicates
    g = dup_groups(X2)
    assert all(g[i] == g[100 + i] for i in range(10))
    assert len(np.unique(g)) == 100


def test_errors_are_graceful(arch):
    from autotagtt import train
    ids, X, pos = make_data()
    write_stars(arch, pos)
    with pytest.raises(SystemExit):
        train.train_tag("nonexistent", ids, X)
    with pytest.raises(SystemExit):
        train.train_tag("braces", [], np.zeros((0, 0)))
    write_stars(arch, {"v00000"})
    with pytest.raises(SystemExit):
        train.train_tag("braces", ids, X)   # too few positives


def test_store_resume_and_private_perms(arch):
    from autotagtt.store import EmbStore
    os.umask(0o077)
    s = EmbStore("m/x", "cover")
    s.add(["a", "b"], np.ones((2, 4), np.float32))
    s.add(["c"], np.ones((1, 4), np.float32))
    s2 = EmbStore("m/x", "cover")
    assert s2.done_ids() == {"a", "b", "c"}
    ids, X = s2.load()
    assert X.shape == (3, 4) and X.dtype == np.float16
    assert stat.S_IMODE(os.stat(s.dir).st_mode) == 0o700
    # a crashed write (npy present, json commit marker missing) is ignored
    np.save(s.dir / "shard_00002.npy", np.ones((1, 4), np.float16))
    assert EmbStore("m/x", "cover").done_ids() == {"a", "b", "c"}


def test_suggestions_written_atomically_with_backup(arch):
    from autotagtt import train
    from autotagtt.archive import OUT
    ids, X, pos = make_data()
    write_stars(arch, pos)
    for _ in range(2):
        train.write_suggestions({"braces": train.train_tag("braces", ids, X, top=10)}, "m")
    out = OUT / "autotag.json"
    assert (OUT / "autotag.json.bak").exists()
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    d = json.loads(out.read_text())
    assert len(d["braces"]) == 10 and "score" in d["_meta"]
