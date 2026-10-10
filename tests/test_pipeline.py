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
    assert train.train_tag("nonexistent", ids, X) is None
    with pytest.raises(SystemExit):
        train.train_tag("braces", [], np.zeros((0, 0)))
    write_stars(arch, {"v00000"})
    assert train.train_tag("braces", ids, X) is None   # too few positives


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


def test_unsure_is_excluded_from_training_and_suggestions(arch):
    from autotagtt import train
    ids, X, pos = make_data()
    known = set(sorted(pos)[:45])
    write_stars(arch, known)
    first = train.train_tag("braces", ids, X, top=20)["suggestions"]
    uns = [first[0]["id"], first[1]["id"]]
    (arch / "data/.appdata/autotag_feedback.json").write_text(json.dumps({"braces": {"unsure": uns}}))
    r = train.train_tag("braces", ids, X, top=300)
    assert r["metrics"]["excluded_unsure"] == 2 and r["metrics"]["n"] == len(ids) - 2
    assert not ({x["id"] for x in r["suggestions"]} & set(uns))


def write_multi(root, groups):
    (root / "data/.appdata/stars.json").write_text(json.dumps({"stars": [], "levels": [], "groups": [
        {"id": f"g{i}", "name": n, "videoIds": sorted(v)} for i, (n, v) in enumerate(groups.items())]}))


def test_multi_tag_config_labels_gate_and_ai_tags(arch):
    from autotagtt import archive, train
    importlib.reload(archive)
    ids, X, pos = make_data(n=800, npos=80)
    rng = np.random.default_rng(5)
    noise = set(rng.choice(ids, 60, replace=False).tolist())          # a tag with no visual signal
    write_multi(arch, {"braces": set(sorted(pos)[:60]), "Secret Tag": noise, "tiny": {"v00001", "v00002"}})
    (arch / "data/.appdata/autotag_tags.json").write_text(json.dumps({"enabled": ["Braces", "secret tag", "tiny"]}))
    assert archive.enabled_tags() == ["braces", "secret tag", "tiny"]
    assert archive.label("braces") == "braces" and "secret" not in archive.label("secret tag")
    res = {t: train.train_tag(t, ids, X, top=30) for t in archive.enabled_tags()}
    assert res["tiny"] is None                                        # too few positives
    assert res["braces"]["metrics"]["gate"] == "ok" and res["secret tag"]["metrics"]["gate"] == "weak"
    assert res["secret tag"]["suggestions"] == [] and res["secret tag"]["ai"] == {}
    ai = res["braces"]["ai"]
    assert ai and not (set(ai) & set(sorted(pos)[:60]))               # never tags videos already in the human group
    assert res["braces"]["metrics"]["ai_mode"] == "unvalidated" and len(ai) <= train.AI_UNVALIDATED_TOP


def test_ai_cutoff_uses_reviewed_precision(arch):
    from autotagtt.train import ai_tags
    ids = [f"v{i}" for i in range(200)]
    oof = np.linspace(5, -5, 200)                                     # rank order == id order
    y = np.zeros(200, int)
    accepted = {f"v{i}" for i in range(0, 60)} | {f"v{i}" for i in range(60, 100, 2)}   # 60 sure hits, then 50/50
    rejected = {f"v{i}" for i in range(61, 100, 2)}
    r = ai_tags(ids, oof, y, accepted, rejected, set())
    assert r["mode"] == "reviewed" and r["cutoff"] is not None
    kept = set(r["items"])
    assert not (kept & (accepted | rejected))                         # reviewed items excluded
    assert all(oof[int(v[1:])] >= r["cutoff"] for v in kept)
    assert r["cutoff"] > oof[99]                                      # stops before the 50% precision zone


# ───────────── worker (fake job, no GPU) ─────────────
@pytest.fixture
def wk(arch, monkeypatch, tmp_path):
    job = tmp_path / "fakejob.py"
    job.write_text("import sys,json,os\nprint('covers: 10 total, 0 embedded, 10 to do')\nprint('  10/10  50 img/s  eta 0.0 min  vram 1.0 GB')\n"
                   "print('[tag-aaaaaa] embedded=10 positives=3 reviewed_rejected=0')\n"
                   "if os.environ.get('FAKE_FAIL'): sys.exit(3)\n")
    monkeypatch.setenv("AUTOTAGTT_TEST_JOB", str(job))
    for m in ("autotagtt.archive", "autotagtt.worker"):
        importlib.reload(importlib.import_module(m))
    from autotagtt import worker
    return worker


def test_worker_runs_a_request_once_and_reports(wk):
    from autotagtt.archive import write_json_atomic
    write_json_atomic(wk.REQUEST, {"id": "r1", "ts": 1})
    st = wk.load_status()
    assert wk.tick(st) is True
    s = wk.load_status()
    assert s["state"] == "idle" and s["phase"] == "done" and s["last_request_id"] == "r1" and s["error"] is None
    assert wk.tick(wk.load_status()) is False                    # same request id is never run twice
    write_json_atomic(wk.REQUEST, {"id": "r2", "ts": 2})
    assert wk.tick(wk.load_status()) is True
    assert stat.S_IMODE(os.stat(wk.STATUS).st_mode) == 0o600


def test_worker_failure_is_reported_without_details(wk, monkeypatch):
    from autotagtt.archive import write_json_atomic
    monkeypatch.setenv("FAKE_FAIL", "1")
    write_json_atomic(wk.REQUEST, {"id": "f1", "ts": 1})
    wk.tick(wk.load_status())
    s = wk.load_status()
    assert s["phase"] == "failed" and s["state"] == "idle" and "/" not in s["error"]


def test_worker_ignores_malformed_requests(wk):
    wk.REQUEST.parent.mkdir(parents=True, exist_ok=True)
    for bad in ("{nope", "[]", '{"id": 5}', ""):
        wk.REQUEST.write_text(bad)
        assert wk.tick(wk.load_status()) is False


def test_worker_nightly_schedule_runs_once_per_day(wk):
    from autotagtt.archive import write_json_atomic
    import time as _t
    now = _t.mktime((2026, 10, 12, 3, 30, 0, 0, 0, -1))
    st = {}
    assert wk.tick(st, now) is False                              # no schedule file
    write_json_atomic(wk.SCHEDULE, {"enabled": True, "hour": 3})
    assert wk.tick(st, now - 3600) is False                       # 02:30, not yet
    assert wk.tick(st, now) is True                               # 03:30 runs
    assert wk.tick(st, now + 600) is False                        # not again the same day
    assert wk.tick(st, now + 86400) is True                       # next day
    write_json_atomic(wk.SCHEDULE, {"enabled": False, "hour": 3})
    assert wk.tick(st, now + 2 * 86400) is False
    write_json_atomic(wk.SCHEDULE, {"enabled": True, "hour": 99})
    assert wk.tick(st, now + 3 * 86400) is False                  # invalid hour ignored


def test_worker_progress_counter_and_heartbeat_file(wk):
    st = {"phase": "training"}
    for line in ["[tag-a] embedded=10 positives=3", "ai tags: mode=off cutoff=None count=0",
                 "[tag-b] embedded=10 positives=3"]:
        wk.track_progress(st, line, 11)
    assert st["detail"] == "tag 2 of 11" and st["_finished"] == 1
    st2 = {"phase": "embedding"}
    wk.track_progress(st2, "  10/10  50 img/s", 11)
    assert "detail" not in st2
    wk.HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    import threading
    threading.Thread(target=wk.heartbeat_loop, args=(0.05,), daemon=True).start()
    import time as _t; _t.sleep(0.3)
    assert json.loads(wk.HEARTBEAT.read_text())["t"] > _t.time() - 5


# ───────────── notifications (no network: urlopen is faked) ─────────────
class _Resp:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.fixture
def nt(monkeypatch):
    sent = []
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: (sent.append((req.full_url, req.headers, json.loads(req.data))), _Resp())[1])
    from autotagtt import notify
    importlib.reload(notify)
    return notify, sent


def test_notify_disabled_without_url(nt, monkeypatch):
    notify, sent = nt
    monkeypatch.delenv("NTFY_URL", raising=False)
    assert notify.enabled() is False and notify.notify("done", "t", "m") is False and sent == []


def test_notify_done_message_has_counts_only_and_respects_events(nt, monkeypatch):
    notify, sent = nt
    monkeypatch.setenv("NTFY_URL", "https://ntfy.example.com/secret-topic-123")
    monkeypatch.setenv("NTFY_TOKEN", "tk")
    st = {"phase": "done", "reason": "nightly", "result": {
        "Very Private Tag": {"gate": "ok", "pending": 1200, "ai_count": 300},
        "another secret": {"gate": "weak", "pending": 0, "ai_count": 0}}}
    notify.run_finished(st)
    assert len(sent) == 1
    url, headers, body = sent[0]
    assert url == "https://ntfy.example.com/" and body["topic"] == "secret-topic-123"
    assert headers["Authorization"] == "Bearer tk"
    blob = json.dumps(body).lower()
    assert "private" not in blob and "secret tag" not in blob and "another secret" not in blob
    assert "1 tag usable" in body["message"] and "1 too weak" in body["message"] and "1,200" in body["message"] and "Nightly" in body["message"]
    monkeypatch.setenv("NOTIFY_EVENTS", "failed")
    notify.run_finished(st)
    assert len(sent) == 1                                          # 'done' filtered out


def test_notify_failure_message_and_dedupe(nt, monkeypatch):
    notify, sent = nt
    monkeypatch.setenv("NTFY_URL", "https://ntfy.example.com/t")
    notify.run_finished({"phase": "failed", "error": "job failed (exit code 1); see /archive/x"})
    notify.run_finished({"phase": "failed"})
    assert len(sent) == 1 and "/archive" not in json.dumps(sent[0][2])   # no paths; repeat failures deduped


def test_notify_never_raises_or_leaks_url(nt, monkeypatch, capsys):
    notify, sent = nt
    import urllib.request
    def boom(req, timeout=0): raise OSError("secret-topic-123 unreachable")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setenv("NTFY_URL", "https://ntfy.example.com/secret-topic-123")
    assert notify.notify("test", "t", "m") is False
    assert "secret-topic-123" not in capsys.readouterr().out
    monkeypatch.setenv("NTFY_URL", "https://ntfy.example.com/")          # no topic
    assert notify.notify("test", "t", "m") is False


def test_worker_sends_test_notification_once_and_reports(wk, nt, monkeypatch):
    notify, sent = nt
    from autotagtt import worker
    importlib.reload(worker)
    monkeypatch.setenv("NTFY_URL", "https://ntfy.example.com/t")
    from autotagtt.archive import write_json_atomic
    write_json_atomic(worker.NOTIFY_TEST, {"id": "n1", "ts": 1})
    st = {}
    worker.tick(st); worker.tick(st)
    assert len(sent) == 1 and st["notify_test"]["ok"] is True and st["notifications"] is True
