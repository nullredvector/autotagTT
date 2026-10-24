"""Synthetic-data tests (random vectors, fake IDs). No real archive, no GPU."""
import base64, gzip, importlib, json, os, stat
import numpy as np
import pytest


@pytest.fixture
def arch(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHIVE_DIR", str(tmp_path))
    (tmp_path / "data/.appdata").mkdir(parents=True)
    for m in ("autotagtt.archive", "autotagtt.store", "autotagtt.scorer", "autotagtt.train", "autotagtt.serve"):
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
    noise = set(rng.choice(ids[100:], 60, replace=False).tolist())      # a tag with no visual signal (kept clear of the 80 signal videos)
    write_multi(arch, {"braces": set(sorted(pos)[:60]), "Secret Tag": noise, "tiny": {"v00001", "v00002"}})
    (arch / "data/.appdata/autotag_tags.json").write_text(json.dumps({"enabled": ["Braces", "secret tag", "tiny"]}))
    assert archive.enabled_tags() == ["braces", "secret tag", "tiny"]
    assert archive.label("braces") == "braces" and "secret" not in archive.label("secret tag")
    res = {t: train.train_tag(t, ids, X, top=30) for t in archive.enabled_tags()}
    assert res["tiny"] is None                                        # too few positives
    assert res["braces"]["metrics"]["gate"] == "ok" and res["secret tag"]["metrics"]["gate"] == "weak"
    assert res["secret tag"]["suggestions"] == [] and res["secret tag"]["ai"] == {} and res["secret tag"]["metrics"]["verdict"] == "hopeless"
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


# ───────────── M4: caching + incremental scoring ─────────────
def _setup_tag(arch, n=600, npos=60, known=45, seed=0):
    ids, X, pos = make_data(n=n, npos=npos, seed=seed)
    write_stars(arch, set(sorted(pos)[:known]))
    return ids, X, pos


@pytest.fixture
def cvspy(monkeypatch):
    from autotagtt import train
    calls = {"n": 0}
    real = train._cv
    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(train, "_cv", spy)
    return calls


def test_unchanged_tag_is_reused_without_retraining(arch, cvspy):
    from autotagtt import train
    ids, X, _ = _setup_tag(arch)
    a = train.train_tag("braces", ids, X, top=40)
    n1 = cvspy["n"]
    assert a["mode"] == "trained" and n1 > 0
    b = train.train_tag("braces", ids, X, top=40)
    assert b["mode"] == "reused" and cvspy["n"] == n1                     # no cross-validation the second time
    assert [s["id"] for s in a["suggestions"]] == [s["id"] for s in b["suggestions"]]
    assert a["ai"] == b["ai"]


def test_new_videos_are_scored_with_the_saved_model(arch, cvspy):
    from autotagtt import train
    ids, X, pos = _setup_tag(arch)
    train.train_tag("braces", ids, X, top=40)
    n1 = cvspy["n"]
    rng = np.random.default_rng(9)
    d = X.shape[1]
    direction = np.linalg.lstsq(X[:60], np.ones(60), rcond=None)[0]       # roughly the positive direction
    newX = rng.normal(size=(10, d)).astype(np.float32)
    newX[:5] += 2.5 * (X[:45].mean(0) / np.linalg.norm(X[:45].mean(0))) * 6   # 5 that look like the positives
    newX /= np.linalg.norm(newX, axis=1, keepdims=True)
    nids = [f"new{i}" for i in range(10)]
    r = train.train_tag("braces", ids + nids, np.vstack([X, newX]), top=800)
    assert r["mode"] == "incremental" and cvspy["n"] == n1 and r["metrics"]["incremental_new"] == 10
    sc = {s["id"]: s["score"] for s in r["suggestions"]}
    assert all(f"new{i}" in sc for i in range(10))
    assert np.mean([sc[f"new{i}"] for i in range(5)]) > np.mean([sc[f"new{i}"] for i in range(5, 10)])


def test_label_change_or_many_new_videos_force_a_retrain(arch, cvspy, monkeypatch):
    from autotagtt import train
    ids, X, _ = _setup_tag(arch)
    first = train.train_tag("braces", ids, X, top=40)
    (arch / "data/.appdata/autotag_feedback.json").write_text(json.dumps({"braces": {"rejected": [first["suggestions"][0]["id"]]}}))
    assert train.train_tag("braces", ids, X, top=40)["mode"] == "trained"      # a new review changes the labels
    assert train.train_tag("braces", ids, X, top=40)["mode"] == "reused"
    rng = np.random.default_rng(3)
    bigX = rng.normal(size=(300, X.shape[1])).astype(np.float32)               # 300 new > max(200, 5%)
    bigX /= np.linalg.norm(bigX, axis=1, keepdims=True)
    assert train.train_tag("braces", ids + [f"b{i}" for i in range(300)], np.vstack([X, bigX]), top=40)["mode"] == "trained"
    monkeypatch.setattr(train, "MAX_AGE_S", -1)                                # a stale cache is never trusted
    assert train.train_tag("braces", ids, X, top=40)["mode"] == "trained"


def test_group_change_invalidates_cache_and_unsure_stays_out(arch, cvspy):
    from autotagtt import train
    ids, X, pos = _setup_tag(arch)
    train.train_tag("braces", ids, X, top=40)
    write_stars(arch, set(sorted(pos)[:46]))                                   # user tagged one more video
    r = train.train_tag("braces", ids, X, top=40)
    assert r["mode"] == "trained"
    (arch / "data/.appdata/autotag_feedback.json").write_text(json.dumps({"braces": {"unsure": ["v00300"]}}))
    r2 = train.train_tag("braces", ids, X, top=800)
    assert "v00300" not in {s["id"] for s in r2["suggestions"]} and "v00300" not in r2["ai"]


def test_skipped_tag_verdict_is_cached_and_broken_cache_is_ignored(arch, cvspy):
    from autotagtt import train
    ids, X, pos = _setup_tag(arch, known=10)
    write_stars(arch, {"v00000", "v00001"})
    assert train.train_tag("braces", ids, X) is None
    assert train.train_tag("braces", ids, X) is None                           # cached verdict, no work
    assert cvspy["n"] == 0
    write_stars(arch, set(sorted(pos)[:45]))
    r = train.train_tag("braces", ids, X, top=20)
    assert r["mode"] == "trained"
    (train._tagdir("braces") / "meta.json").unlink()                           # an interrupted save leaves no commit marker
    assert train.train_tag("braces", ids, X, top=20)["mode"] == "trained"


def test_cache_files_are_private_and_hash_named(arch):
    from autotagtt import train
    ids, X, _ = _setup_tag(arch)
    train.train_tag("braces", ids, X, top=20)
    d = train._tagdir("braces")
    assert "braces" not in d.name and stat.S_IMODE(os.stat(d).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(d / "meta.json").st_mode) == 0o600


# ───────────── M5: frames ─────────────
class FakeEmbedder:
    d = 8

    def encode(self, images):
        out = []
        for im in images:
            a = np.asarray(im.resize((8, 8))).astype(np.float32).reshape(-1)[: self.d]
            out.append(a / (np.linalg.norm(a) + 1e-9))
        return np.array(out, np.float32).reshape(len(images), self.d)


@pytest.fixture
def ff_mod(arch, monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    import imageio_ffmpeg
    monkeypatch.setenv("FFMPEG", imageio_ffmpeg.get_ffmpeg_exe())
    importlib.reload(importlib.import_module("autotagtt.frames"))
    from autotagtt import frames
    return frames


def _make_video(frames, path, seconds=4):
    import subprocess
    subprocess.run([frames.FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=160x284:rate=10:duration={seconds}",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)], check=True)


def test_extract_frames_and_duration(ff_mod, tmp_path):
    v = tmp_path / "v.mp4"
    _make_video(ff_mod, v, 4)
    assert abs(ff_mod.video_duration(v) - 4.0) < 0.5
    out = tmp_path / "o"; out.mkdir()
    paths = ff_mod.extract_frames(v, out)
    assert len(paths) == ff_mod.K and all(p.stat().st_size > 0 for p in paths)
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"not a video")
    assert ff_mod.video_duration(bad) is None and ff_mod.extract_frames(bad, out) == []


def test_embed_videos_stores_cleans_up_and_resumes(ff_mod, arch, tmp_path):
    import tempfile, glob
    vids = []
    for i in range(5):
        p = tmp_path / f"v{i}.mp4"; _make_video(ff_mod, p, 3); vids.append((f"v{i}", p))
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"junk"); vids.append(("bad", bad))
    store = ff_mod.FrameStore("m")
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "atframes_*")))
    st = ff_mod.embed_videos(FakeEmbedder(), vids, store, mouth=False, threads=2, batch=2, log=lambda *_: None)
    assert st["done"] == 5 and st["failed"] == 1
    assert set(glob.glob(os.path.join(tempfile.gettempdir(), "atframes_*"))) == before      # frames never outlive their batch
    assert store.done_ids() == {f"v{i}" for i in range(5)}
    ids, F = store.load_pooled()
    assert F.shape == (5, 3 * 8 + 1) and F.dtype == np.float16 and not np.isnan(F.astype(np.float32)).any()
    # resume: finished videos are not redone; a video that failed twice is given up on
    done = store.done_ids()
    todo = [x for x in vids if x[0] not in done and x[0] not in store.skipped_ids()]
    assert [t[0] for t in todo] == ["bad"]
    ff_mod.embed_videos(FakeEmbedder(), todo, store, log=lambda *_: None)
    assert "bad" in store.skipped_ids() and store.failed()["bad"] == 2
    n, total = ff_mod.coverage(["v0", "v1", "bad"], store, {"v0": 1, "v1": 1, "bad": 1})
    assert (n, total) == (3, 3)


def test_embed_videos_respects_time_budget(ff_mod, arch, tmp_path):
    vids = []
    for i in range(4):
        p = tmp_path / f"w{i}.mp4"; _make_video(ff_mod, p, 2); vids.append((f"w{i}", p))
    store = ff_mod.FrameStore("m2")
    st = ff_mod.embed_videos(FakeEmbedder(), vids, store, budget_s=0, batch=1, log=lambda *_: None)
    assert st["done"] == 0                                              # budget exhausted before the first batch


def test_lower_face_crop_without_faces_is_none(ff_mod):
    from PIL import Image
    assert ff_mod.lower_face_crop(Image.new("RGB", (200, 300), "white")) is None


def _frame_data(n=600, npos=60, d=8, cover_signal=False, frame_signal=True, seed=2):
    ids, X, pos = make_data(n=n, npos=npos, seed=seed)
    if not cover_signal:                                              # cover embeddings carry no information about the tag
        rng = np.random.default_rng(seed + 1)
        X = rng.normal(size=X.shape).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    rng = np.random.default_rng(seed + 2)
    F = rng.normal(size=(n, 3 * d + 1)).astype(np.float32)
    F[:, -1] = 0
    if frame_signal:
        pidx = [i for i, v in enumerate(ids) if v in pos]
        F[pidx, :2 * d] += 2.5
    return ids, X, pos, F


def test_frames_chosen_only_when_they_clearly_help(arch):
    from autotagtt import train
    ids, X, pos, F = _frame_data(frame_signal=True)
    write_stars(arch, set(sorted(pos)[:45]))
    ff = train.FrameFeatures(ids, F.astype(np.float16), ids, coverage=1.0)
    r = train.train_tag("braces", ids, X, top=30, ff=ff, force=True)
    assert r["metrics"]["features"] == "frames" and r["metrics"]["ap"] > 0.5
    base = train.train_tag("braces", ids, X, top=30, ff=None, force=True)
    assert r["metrics"]["ap"] > base["metrics"]["ap"] + 0.2 and base["metrics"]["features"] == "cover"
    ids2, X2, pos2, F2 = _frame_data(frame_signal=False)
    write_stars(arch, set(sorted(pos2)[:45]))
    ff2 = train.FrameFeatures(ids2, F2.astype(np.float16), ids2, coverage=1.0)
    assert train.train_tag("braces", ids2, X2, top=30, ff=ff2, force=True)["metrics"]["features"] == "cover"   # noise never wins


def test_frame_features_ignored_until_coverage_is_high(arch):
    from autotagtt import train
    ids, X, pos, F = _frame_data(frame_signal=True)
    write_stars(arch, set(sorted(pos)[:45]))
    ff = train.FrameFeatures(ids[:300], F[:300].astype(np.float16), ids, coverage=0.5)
    assert ff.ready is False and ff.kinds() == []
    assert train.train_tag("braces", ids, X, top=20, ff=ff, force=True)["metrics"]["features"] == "cover"


def test_incremental_scoring_with_frame_features_waits_for_frames(arch, cvspy):
    from autotagtt import train
    ids, X, pos, F = _frame_data(frame_signal=True)
    write_stars(arch, set(sorted(pos)[:45]))
    ff = train.FrameFeatures(ids, F.astype(np.float16), ids, coverage=1.0)
    assert train.train_tag("braces", ids, X, top=20, ff=ff)["metrics"]["features"] == "frames"
    n1 = cvspy["n"]
    rng = np.random.default_rng(4)
    nx = rng.normal(size=(6, X.shape[1])).astype(np.float32)
    nids = [f"n{i}" for i in range(6)]
    nF = rng.normal(size=(6, F.shape[1])).astype(np.float32); nF[:, -1] = 0
    ff2 = train.FrameFeatures(ids + nids[:3], np.vstack([F, nF[:3]]).astype(np.float16), ids + nids, coverage=1.0)   # only 3 of 6 have frames
    r = train.train_tag("braces", ids + nids, np.vstack([X, nx]), top=900, ff=ff2)
    assert r["mode"] == "incremental" and cvspy["n"] == n1 and r["metrics"]["incremental_new"] == 3


def test_trial_selection_and_report(arch):
    from autotagtt import train
    ids, X, pos, F = _frame_data(n=900, npos=90, frame_signal=True)
    write_stars(arch, set(sorted(pos)[:80]))
    chosen = train.select_trial(ids, ["braces"], per_tag=60, random_extra=300)
    assert len(chosen & set(sorted(pos)[:80])) >= 60 and len(chosen) >= 300
    assert json.loads(train.TRIAL_IDS.read_text()) == sorted(chosen)
    ff = train.FrameFeatures(ids, F.astype(np.float16), ids, coverage=0.2)
    rep = train.trial_report(["braces"], ids, X, ff, chosen)
    row = rep["braces"]
    assert row["ap_frames"] > row["ap_cover"] + 0.2 and "ap_mouth" not in row
    assert json.loads(train.TRIAL_REPORT.read_text())["braces"]["n_pos"] == row["n_pos"]


def test_frames_config_defaults_and_validation(arch, ff_mod):
    assert ff_mod.frames_config() == {"mode": "off", "mouth": False, "text": True, "budget_min": 120}
    ff_mod.CONFIG.parent.mkdir(parents=True, exist_ok=True)
    ff_mod.CONFIG.write_text(json.dumps({"mode": "full", "mouth": True, "budget_min": 5}))
    assert ff_mod.frames_config() == {"mode": "full", "mouth": True, "text": True, "budget_min": 120}   # out-of-range budget falls back
    ff_mod.CONFIG.write_text(json.dumps({"mode": "off", "text": False}))
    assert ff_mod.frames_config()["text"] is False
    ff_mod.CONFIG.write_text(json.dumps({"mode": "everything"}))
    assert ff_mod.frames_config()["mode"] == "off"


def test_worker_status_carries_frames_info_and_quiet_nights_do_not_notify(wk, nt, monkeypatch):
    from autotagtt import worker
    importlib.reload(worker)
    notify, sent = nt
    monkeypatch.setenv("NTFY_URL", "https://ntfy.example.com/t")
    from autotagtt.archive import write_json_atomic
    OUT = worker.OUT
    write_json_atomic(OUT / "frames_stats.json", {"done": 10, "total": 100, "rate": 3.5, "mode": "trial", "ts": 1})
    write_json_atomic(OUT / "frames_trial.json", {"x": {"n_pos": 20, "n": 200, "ap_cover": 0.2, "ap_frames": 0.5}})
    write_json_atomic(OUT / "run_summary.json", {"changed": False, "new_covers": 0, "modes": {"reused": 3}})
    write_json_atomic(worker.SCHEDULE, {"enabled": True, "hour": 0})
    st = {}
    assert worker.tick(st, now=__import__("time").mktime((2026, 10, 12, 4, 0, 0, 0, 0, -1))) is True
    s = worker.load_status()
    assert s["frames"]["stats"]["done"] == 10 and s["frames"]["trial"]["x"]["ap_frames"] == 0.5
    assert s["run_summary"]["changed"] is False
    assert sent == []                                     # nightly + nothing changed -> silent
    write_json_atomic(worker.REQUEST, {"id": "m1", "ts": 1})
    worker.tick(st)
    assert len(sent) == 1                                 # a manual run always reports


def test_run_orchestration_trial_then_full_then_reuse(ff_mod, arch, monkeypatch, tmp_path):
    from types import SimpleNamespace
    for m in ("autotagtt.archive", "autotagtt.store", "autotagtt.train", "autotagtt.cli"):
        importlib.reload(importlib.import_module(m))
    from autotagtt import cli, train, embed, archive
    from autotagtt.store import EmbStore
    monkeypatch.setattr(embed, "ImageEmbedder", lambda m, p: FakeEmbedder())
    ids, X, pos = make_data(n=90, npos=45, d=32)
    vdir = arch / "data/Likes/videos"; vdir.mkdir(parents=True)
    for v in ids:
        _make_video(ff_mod, vdir / f"{v}.mp4", 1)
    write_stars(arch, set(sorted(pos)[:40]))
    EmbStore("m", "cover").add(ids, X)
    a = SimpleNamespace(model="m", pretrained="p", top=30)
    out = archive.OUT

    ff_mod.CONFIG.parent.mkdir(parents=True, exist_ok=True)
    ff_mod.CONFIG.write_text(json.dumps({"mode": "trial", "mouth": False, "budget_min": 30}))
    cover_ids, Xc = EmbStore("m", "cover").load()
    cli._train_and_write(a, ["braces"], cover_ids, Xc, new_covers=90, with_frames=True)
    summary = json.loads((out / "run_summary.json").read_text())
    assert summary["changed"] is True and summary["modes"] == {"trained": 1}
    assert json.loads((out / "frames_trial.json").read_text()) is not None and (out / "frames_stats.json").exists()
    stats = json.loads((out / "frames_stats.json").read_text())
    assert stats["done"] == 90 and stats["total"] == 90 and stats["mode"] == "trial"

    ff_mod.CONFIG.write_text(json.dumps({"mode": "full", "mouth": False, "budget_min": 30}))
    cli._train_and_write(a, ["braces"], cover_ids, Xc, new_covers=0, with_frames=True)
    assert json.loads((out / "run_summary.json").read_text())["frames_used"] is True    # coverage is complete, frames are used
    prog = json.loads((out / "frames_progress.json").read_text())
    assert prog["running"] is False and prog["covered"] == prog["total"] == 90 and prog["mode"] == "full"
    cli._train_and_write(a, ["braces"], cover_ids, Xc, new_covers=0, with_frames=True)
    s3 = json.loads((out / "run_summary.json").read_text())
    assert s3["changed"] is False and s3["modes"] == {"reused": 1}                       # a quiet night: nothing recomputed
    ai = json.loads((out / "ai_tags.json").read_text())
    assert "videos" in ai and stat.S_IMODE(os.stat(out / "autotag.json").st_mode) == 0o600


def test_frame_progress_is_reported_after_every_batch(ff_mod, arch, tmp_path):
    vids = []
    for i in range(5):
        p = tmp_path / f"p{i}.mp4"; _make_video(ff_mod, p, 2); vids.append((f"p{i}", p))
    seen = []
    store = ff_mod.FrameStore("mp")
    ff_mod.embed_videos(FakeEmbedder(), vids, store, batch=2, log=lambda *_: None, on_progress=seen.append)
    assert [p["session_done"] for p in seen] == [2, 4, 5] and all(p["session_target"] == 5 for p in seen)
    assert seen[-1]["elapsed"] > 0
    # a broken progress callback never stops the scan
    store2 = ff_mod.FrameStore("mp2")
    def boom(_): raise RuntimeError("x")
    st = ff_mod.embed_videos(FakeEmbedder(), vids[:2], store2, batch=1, log=lambda *_: None, on_progress=boom)
    assert st["done"] == 2


def test_progress_file_is_atomic_private_and_complete(ff_mod, arch):
    ff_mod.write_progress("full", 100, 40, 10, 60, 2.5, running=True)
    d = json.loads(ff_mod.PROGRESS.read_text())
    assert d["covered"] == 40 and d["total"] == 100 and d["running"] is True and d["session_target"] == 60 and d["rate"] == 2.5 and d["ts"] > 0
    assert stat.S_IMODE(os.stat(ff_mod.PROGRESS).st_mode) == 0o600
    ff_mod.write_progress("full", 100, 100, 60, 60, 2.5, running=False)
    assert json.loads(ff_mod.PROGRESS.read_text())["running"] is False



# ───────────── weak tags: review-only tier, hopeless flag, fairer feature rule, captions ─────────────
def _weak_data(n=1500, npos=80, shift=0.4, d=32, seed=3):
    rng = np.random.default_rng(seed)
    ids = [f"v{i:05d}" for i in range(n)]
    X = rng.normal(size=(n, d)).astype(np.float32)
    direction = rng.normal(size=d); direction /= np.linalg.norm(direction)
    X[:npos] += shift * direction * 3
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return ids, X, set(ids[:npos])


def test_weak_but_promising_tag_gets_a_review_only_list(arch):
    from autotagtt import train
    ids, X, pos = _weak_data()
    write_stars(arch, pos)
    r = train.train_tag("braces", ids, X, top=300, force=True)
    m = r["metrics"]
    assert m["gate"] == "weak" and m["verdict"] == "review_only" and m["review_only"] is True and 0.65 <= m["auc"] < 0.8
    assert len(r["suggestions"]) == train.REVIEW_ONLY_TOP and r["ai"] == {}            # a short list to review, never AI tags
    (arch / "data/.appdata/autotag_feedback.json").write_text(json.dumps({"braces": {"rejected": [r["suggestions"][0]["id"]]}}))
    r2 = train.train_tag("braces", ids, X, top=300)
    assert r2["suggestions"][0]["id"] != r["suggestions"][0]["id"] and len(r2["suggestions"]) == train.REVIEW_ONLY_TOP
    assert r2["ai"] == {}


def test_tags_with_too_few_positives_or_no_signal_are_flagged_hopeless(arch):
    from autotagtt import train
    ids, X, pos = _weak_data(npos=35)
    write_stars(arch, pos)
    r = train.train_tag("braces", ids, X, top=300, force=True)
    assert r["metrics"]["verdict"] == "hopeless" and r["suggestions"] == [] and r["metrics"]["review_only"] is False


def test_feature_rule_uses_relative_gain_and_auc_for_rare_tags():
    from autotagtt.train import _clearly_better
    assert _clearly_better(0.0095, 0.78, 0.0067, 0.77)            # +42% AP and +0.01 AUC: counts for a rare tag
    assert not _clearly_better(0.0095, 0.77, 0.0067, 0.77)        # relative AP alone, no AUC support: noise
    assert _clearly_better(0.0058, 0.80, 0.0073, 0.77)            # a clear AUC gain is enough by itself
    assert not _clearly_better(0.0058, 0.76, 0.0073, 0.77)        # worse on both
    assert _clearly_better(0.50, 0.90, 0.40, 0.90)                # big absolute AP gain, as before
    assert not _clearly_better(0.405, 0.905, 0.40, 0.90)


def _text_data(n=900, npos=90, d=8, text_signal=True, frame_signal=False, seed=11):
    rng = np.random.default_rng(seed)
    ids = [f"v{i:05d}" for i in range(n)]
    X = rng.normal(size=(n, 32)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)    # covers say nothing
    pidx = list(range(npos))
    T = rng.normal(size=(n, d)).astype(np.float32)
    if text_signal:
        T[pidx] += 2.5
    F = rng.normal(size=(n, 3 * d + 1)).astype(np.float32); F[:, -1] = 0
    if frame_signal:
        F[pidx, :2 * d] += 2.5
    return ids, X, set(ids[:npos]), T.astype(np.float16), F.astype(np.float16)


def test_captions_are_used_only_when_they_clearly_help(arch):
    from autotagtt import train
    ids, X, pos, T, F = _text_data(text_signal=True)
    write_stars(arch, set(sorted(pos)[:70]))
    tf = train.TextFeatures(ids, T, ids)
    assert tf.ready and tf.coverage == 1.0
    r = train.train_tag("braces", ids, X, top=30, tf=tf, force=True)
    assert r["metrics"]["features"] == "text" and r["metrics"]["auc"] > 0.9 and "text" in r["metrics"]["feature_aucs"]
    ids2, X2, pos2, T2, F2 = _text_data(text_signal=False)
    write_stars(arch, set(sorted(pos2)[:70]))
    assert train.train_tag("braces", ids2, X2, top=30, tf=train.TextFeatures(ids2, T2, ids2), force=True)["metrics"]["features"] == "cover"
    # too few videos with captions: not used
    sparse = train.TextFeatures(ids[:100], T[:100], ids)
    assert sparse.ready is False and train._signature(None, sparse) == "cover"


def test_frames_and_captions_together_when_each_helps(arch):
    from autotagtt import train
    ids, X, pos, T, F = _text_data(text_signal=True, frame_signal=True)
    write_stars(arch, set(sorted(pos)[:70]))
    ff = train.FrameFeatures(ids, F, ids, coverage=1.0)
    tf = train.TextFeatures(ids, T, ids)
    r = train.train_tag("braces", ids, X, top=30, ff=ff, tf=tf, force=True)
    assert r["metrics"]["features"] in ("frames+text", "text", "frames") and r["metrics"]["auc"] > 0.95
    assert set(r["metrics"]["feature_aucs"]) >= {"cover", "frames", "text"}


def test_new_videos_are_scored_with_text_features(arch, cvspy):
    from autotagtt import train
    ids, X, pos, T, F = _text_data(text_signal=True)
    write_stars(arch, set(sorted(pos)[:70]))
    tf = train.TextFeatures(ids, T, ids)
    first = train.train_tag("braces", ids, X, top=30, tf=tf)
    assert first["metrics"]["features"] == "text"
    n1 = cvspy["n"]
    nids = ["new1", "new2", "new3"]
    nX = np.random.default_rng(1).normal(size=(3, 32)).astype(np.float32)
    nT = np.vstack([T.astype(np.float32)[:2] * 0 + 3.0, np.zeros((1, T.shape[1]), np.float32)])    # two captioned like positives, one with no caption
    tf2 = train.TextFeatures(ids + nids[:2], np.vstack([T, nT[:2].astype(np.float16)]), ids + nids)
    r = train.train_tag("braces", ids + nids, np.vstack([X, nX]), top=900, tf=tf2)
    assert r["mode"] == "incremental" and cvspy["n"] == n1 and r["metrics"]["incremental_new"] == 3
    sc = {s["id"]: s["score"] for s in r["suggestions"]}
    assert sc["new1"] > sc["new3"] and sc["new2"] > sc["new3"]


def test_load_and_embed_captions_privately(arch):
    from autotagtt import text
    from autotagtt.archive import write_json_atomic
    importlib.reload(text)
    caps = {"v1": "A secret caption about something", "v2": "x", "v3": "another one #tag", "v4": ""}
    blob = base64.b64encode(gzip.compress(json.dumps(caps).encode())).decode()
    (arch / "data/.appdata/db_texts.js").write_text(f'window.dbvd_base64="{blob}";')
    assert text.load_captions() == {"v1": "A secret caption about something", "v3": "another one #tag"}   # too-short captions dropped
    class E:
        calls = 0
        def encode(self, texts):
            E.calls += 1
            return np.ones((len(texts), 4), np.float32)
    assert text.embed_texts(E(), ["v1", "v2", "v3", "v9"], "m", log=lambda *_: None) == 2
    assert text.embed_texts(E(), ["v1", "v3"], "m", log=lambda *_: None) == 0 and E.calls == 1      # cached: nothing re-embedded
    blob_all = b"".join(p.read_bytes() for p in (arch / "data/.appdata/autotag").rglob("*") if p.is_file())
    assert b"secret caption" not in blob_all and b"another one" not in blob_all                   # raw captions are never written
    (arch / "data/.appdata/db_texts.js").write_text("garbage")
    assert text.load_captions() == {}                                                              # unreadable file: quietly no captions


def test_summary_reports_verdict_and_features(wk):
    from autotagtt.archive import write_json_atomic
    write_json_atomic(wk.OUT / "autotag.json", {"_meta": {"metrics": {"t": {"positives": 90, "auc": 0.77, "gate": "weak", "verdict": "review_only",
                       "review_only": True, "features": "cover", "ai_count": 0, "ai_mode": "off"}}}, "t": [{"id": "a"}] * 5})
    s = wk.summarize()["t"]
    assert s["verdict"] == "review_only" and s["review_only"] is True and s["features"] == "cover" and s["pending"] == 5


def test_optional_steps_never_fail_the_run(arch, monkeypatch, capsys):
    from types import SimpleNamespace
    for m in ("autotagtt.archive", "autotagtt.store", "autotagtt.train", "autotagtt.cli"):
        importlib.reload(importlib.import_module(m))
    from autotagtt import cli, text
    from autotagtt.store import EmbStore
    ids, X, pos = make_data(n=300, npos=45, d=32)
    write_stars(arch, set(sorted(pos)[:40]))
    EmbStore("m", "cover").add(ids, X)
    # a missing library in the captions step (the real incident) and a crash in the frames step
    def boom(*a, **k): raise ModuleNotFoundError("No module named 'transformers' /private/path")
    monkeypatch.setattr(text, "embed_texts", boom)
    monkeypatch.setattr(cli, "_frames_step", boom)
    cli._train_and_write(SimpleNamespace(model="m", pretrained="p", top=20), ["braces"], ids, X, new_covers=0, with_frames=True)
    out = capsys.readouterr().out
    assert "captions: skipped (ModuleNotFoundError)" in out and "frames: skipped (ModuleNotFoundError)" in out
    assert "/private/path" not in out and "transformers" not in out                    # error type only, never the message
    from autotagtt.archive import OUT
    assert (OUT / "autotag.json").exists() and json.loads((OUT / "run_summary.json").read_text())["modes"] == {"trained": 1}



# ───────────── scorer: saved classifiers, calibration, noisy OR, HTTP service ─────────────
import http.client, io, threading, time as _time
from http.server import ThreadingHTTPServer
from PIL import Image


def _models(arch, specs):
    """specs: {tag: (w, b, weight, calib)} -> manifest written, ModelSet loaded."""
    from autotagtt import scorer
    for tag, (w, b, weight, calib) in specs.items():
        scorer.save_tag_model(tag, np.array(w, np.float32), b, calib, weight, "test")
    scorer.write_manifest(list(specs))
    ms = scorer.ModelSet()
    ms.load()
    return scorer, ms


def test_noisy_or_with_known_weights(arch):
    scorer, ms = _models(arch, {"a": ([5, 0, 0, 0], 0.0, 0.9, None), "b": ([0, 0, 0, 0], 0.0, 0.5, None)})
    score, tags, probs = ms.score([1, 0, 0, 0])
    pa, pb = 1 / (1 + np.exp(-5.0)), 0.5
    assert score == round(1 - (1 - 0.9 * pa) * (1 - 0.5 * pb), 4)
    assert tags == [["a", round(0.9 * pa, 4)], ["b", 0.25]] and probs == {"a": round(pa, 4), "b": 0.5}
    only_strong, _, _ = ms.score([1, 0, 0, 0])
    assert only_strong > 0.9 * pa                                   # a second tag adds to one strong match


def test_calibrated_tag_uses_its_platt_curve_and_best_three_are_returned(arch):
    scorer, ms = _models(arch, {"cal": ([1, 0, 0, 0], 0.0, 1.0, {"a": 2.0, "c": -1.0}),
                                "t1": ([0, 1, 0, 0], 0.0, 0.8, None), "t2": ([0, 0, 1, 0], 0.0, 0.6, None), "t3": ([0, 0, 0, 1], 0.0, 0.4, None)})
    score, tags, probs = ms.score([1, 0, 0, 0])
    assert probs["cal"] == round(1 / (1 + np.exp(-(2 * 1 - 1))), 4)
    assert len(tags) == 3 and tags[0][0] == "cal"                    # best three by weight * probability


def test_low_weight_tags_are_dropped_from_the_manifest_and_the_model_id_tracks_content(arch):
    scorer, ms = _models(arch, {"keep": ([1, 0, 0, 0], 0.0, 0.9, None), "weak": ([1, 0, 0, 0], 0.0, 0.2, None)})
    man = json.loads((scorer.MODELS / "manifest.json").read_text())
    assert [t["tag"] for t in man["tags"]] == ["keep"]
    id1 = man["model"]
    scorer.write_manifest(["keep", "weak"])
    assert json.loads((scorer.MODELS / "manifest.json").read_text())["model"] == id1       # unchanged content: same id
    scorer.save_tag_model("keep", np.array([2, 0, 0, 0], np.float32), 0.0, None, 0.9, "test")
    scorer.write_manifest(["keep"])
    id2 = json.loads((scorer.MODELS / "manifest.json").read_text())["model"]
    assert id2 != id1 and len(id2.split("-")[-1]) == 6              # new weights: new id


def test_training_saves_models_only_for_usable_tags_with_weight_and_calibration(arch):
    from autotagtt import train, scorer
    from types import SimpleNamespace
    ids, X, pos = make_data(n=900, npos=90, d=32)
    rng = np.random.default_rng(8)
    noise = set(rng.choice(ids[120:], 60, replace=False).tolist())
    write_multi(arch, {"braces": set(sorted(pos)[:60]), "noise": noise})
    hidden = sorted(pos - set(sorted(pos)[:60]))
    fb = {"braces": {"accepted": hidden[:25], "rejected": [v for v in ids[400:425]]}}
    (arch / "data/.appdata/autotag_feedback.json").write_text(json.dumps(fb))
    good = train.train_tag("braces", ids, X, top=30)
    bad = train.train_tag("noise", ids, X, top=30)
    assert good["metrics"]["gate"] == "ok" and bad["metrics"]["gate"] == "weak"
    assert scorer.has_model("braces") and not scorer.has_model("noise")                # weak tags get no scorer model
    meta = json.loads((scorer.MODELS / f"{scorer.tag_key('braces')}.json").read_text())
    assert meta["weight_source"] == "reviewed" and abs(meta["weight"] - 25 / 50) < 1e-6     # 25 accepted of 50 reviewed
    assert meta["calib"] and meta["calib"]["a"] > 0                                    # 25 + 25 reviews: Platt calibration
    z = np.load(scorer.MODELS / f"{scorer.tag_key('braces')}.npz")
    assert z["w"].shape == (32,)
    # few reviews: class-balanced fallback weighted by cross-validated precision
    (arch / "data/.appdata/autotag_feedback.json").write_text(json.dumps({"braces": {"accepted": hidden[:5], "rejected": ids[400:405]}}))
    train.train_tag("braces", ids, X, top=30)
    meta2 = json.loads((scorer.MODELS / f"{scorer.tag_key('braces')}.json").read_text())
    assert meta2["weight_source"] == "cv" and meta2["calib"] is None and 0 < meta2["weight"] <= 1


def test_tag_whose_scorer_model_is_missing_is_retrained_once(arch, cvspy):
    from autotagtt import train, scorer
    ids, X, pos = make_data(n=600, npos=60, d=32)
    write_stars(arch, set(sorted(pos)[:45]))
    train.train_tag("braces", ids, X, top=20)
    assert train.train_tag("braces", ids, X, top=20)["mode"] == "reused"
    (scorer.MODELS / f"{scorer.tag_key('braces')}.npz").unlink()
    assert train.train_tag("braces", ids, X, top=20)["mode"] == "trained" and scorer.has_model("braces")


class _Emb:
    """Fake embedder: the image's mean red/green/blue become the first three components."""
    def encode(self, imgs):
        out = []
        for im in imgs:
            a = np.asarray(im.resize((4, 4))).reshape(-1, 3).mean(0) / 255.0
            v = np.array([a[0], a[1], a[2], 0.0], np.float32)
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(out, np.float32)


def _img_bytes(fmt="JPEG", size=(32, 48), color=(250, 10, 10)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, fmt)
    return b.getvalue()


@pytest.fixture
def svc(arch):
    from autotagtt import serve, scorer
    scorer_, ms = _models(arch, {"red": ([8, -4, -4, 0], 0.0, 0.9, None)})
    service = serve.Service(models=ms, scores_log=serve.ScoresLog(keep=50, slack=10))
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    def call(method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, json.loads(data or b"{}")
    yield service, call
    server.shutdown()


def test_service_health_loading_and_score_shape(svc):
    service, call = svc
    code, h = call("GET", "/health")
    assert code == 200 and h["ok"] is True and h["ready"] is False and h["tags"] == 1 and h["model"]
    assert call("POST", "/score?id=abc", _img_bytes())[0] == 503                     # model still loading
    service.set_embedder(_Emb())
    assert call("GET", "/health")[1]["ready"] is True
    for fmt in ("JPEG", "PNG", "WEBP"):
        code, r = call("POST", "/score?id=vid_1", _img_bytes(fmt))
        assert code == 200 and set(r) == {"score", "tags", "model"} and 0 <= r["score"] <= 1
        assert r["tags"][0][0] == "red" and r["model"] == h["model"]
    red = call("POST", "/score?id=r", _img_bytes(color=(250, 10, 10)))[1]["score"]
    blue = call("POST", "/score?id=b", _img_bytes(color=(10, 10, 250)))[1]["score"]
    assert red > blue                                                                 # a red cover fits the red tag better
    assert call("GET", "/nope")[0] == 404 and call("POST", "/nope", b"x")[0] == 404


def test_service_limits_and_bad_input(svc):
    service, call = svc
    service.set_embedder(_Emb())
    assert call("POST", "/score?id=a", b"not an image")[0] == 400                    # undecodable
    assert call("POST", "/score?id=a", _img_bytes()[:40])[0] == 400                   # truncated file
    assert call("POST", "/score?id=a", b"")[0] == 400
    assert call("POST", "/score?id=../x", _img_bytes())[0] == 400 and call("POST", "/score", _img_bytes())[0] == 400
    gif = io.BytesIO(); Image.new("RGB", (8, 8)).save(gif, "GIF")
    assert call("POST", "/score?id=a", gif.getvalue())[0] == 400                      # only JPEG/PNG/WebP
    big = io.BytesIO(); Image.new("1", (5000, 5000)).save(big, "PNG")                 # tiny file, 25 megapixels
    assert len(big.getvalue()) < 2 * 1024 * 1024 and call("POST", "/score?id=a", big.getvalue())[0] == 413
    code, _ = call("POST", "/score?id=a", b"x" * (2 * 1024 * 1024 + 1))               # over 2 MB
    assert code == 413


def test_service_picks_up_a_new_model_without_restart_and_logs_scores(svc, arch):
    from autotagtt import scorer
    service, call = svc
    service.set_embedder(_Emb())
    r1 = call("POST", "/score?id=first_id", _img_bytes())[1]
    scorer.save_tag_model("red", np.array([8, -4, -4, 0], np.float32), 0.0, None, 0.5, "test")      # retrained: lower weight
    _time.sleep(0.02)
    scorer.write_manifest(["red"])
    r2 = call("POST", "/score?id=second_id", _img_bytes())[1]
    assert r2["model"] != r1["model"] and r2["score"] < r1["score"]
    lines = [json.loads(l) for l in service.log.path.read_text().splitlines()]
    assert [l["id"] for l in lines] == ["first_id", "second_id"]
    assert set(lines[0]) == {"id", "ts", "p", "score"} and "red" in lines[0]["p"] and lines[0]["score"] == r1["score"]
    assert stat.S_IMODE(os.stat(service.log.path).st_mode) == 0o600


def test_scores_log_is_rotated_keeping_the_newest(arch):
    from autotagtt import serve
    log = serve.ScoresLog(keep=50, slack=10)
    for i in range(130):
        log.append({"id": f"id{i}", "ts": i, "p": {}, "score": 0.1})
        assert log.count <= 61 and len(log.path.read_text().splitlines()) <= 61
    ids = [json.loads(l)["id"] for l in log.path.read_text().splitlines()]
    assert ids[-1] == "id129" and "id0" not in ids and len(ids) >= 50
    assert serve.ScoresLog(keep=50, slack=10).count == len(ids)                      # a restart counts what is already there


def test_worker_keeps_the_scorer_service_running(wk, monkeypatch):
    from autotagtt import worker
    importlib.reload(worker)
    class P:
        def __init__(self, *a, **k): P.n += 1; self.alive = True
        def poll(self): return None if self.alive else 1
    P.n = 0
    monkeypatch.setattr(worker.subprocess, "Popen", P)
    worker._serve.update(proc=None, started=0.0)
    worker.supervise_serve()
    assert P.n == 1
    worker.supervise_serve()                                  # still running: nothing new
    assert P.n == 1
    worker._serve["proc"].alive = False
    worker.supervise_serve()                                  # died, but restarts are rate-limited to one per 30 s
    assert P.n == 1
    worker._serve["started"] -= 31
    worker.supervise_serve()
    assert P.n == 2


def test_scorer_artifacts_reproduce_the_ranking_and_load_into_the_service(arch):
    from autotagtt import cli, scorer, train
    from types import SimpleNamespace
    ids, X, pos = make_data(n=900, npos=90, d=32)
    rng = np.random.default_rng(8)
    noise = set(rng.choice(ids[120:], 60, replace=False).tolist())
    write_multi(arch, {"braces": set(sorted(pos)[:60]), "noise": noise, "other": set(sorted(pos)[:50])})
    cli._train_and_write(SimpleNamespace(model="m", pretrained="p", top=20), ["braces", "noise"], ids, X, new_covers=0, with_frames=False)
    man = json.loads((scorer.MODELS / "manifest.json").read_text())
    assert [t["tag"] for t in man["tags"]] == ["braces"]                      # weak "noise" excluded; "other" was not enabled/trained
    ms = scorer.ModelSet(); ms.load()
    assert ms.model_id == man["model"]
    inside = np.mean([ms.score(X[i])[0] for i in range(60, 90)])               # true positives the user has not tagged
    outside = np.mean([ms.score(X[i])[0] for i in range(300, 330)])
    assert 0.3 < inside <= max(t["weight"] for t in man["tags"]) + 1e-6 and inside > outside + 0.25   # a score never exceeds the tag weight


def test_tag_rescued_only_by_frames_gets_a_capped_weight_and_a_cover_only_model(arch):
    from autotagtt import train, scorer
    ids, X, pos, F = _frame_data(frame_signal=True)                            # covers carry no signal, frames do
    write_stars(arch, set(sorted(pos)[:45]))
    ff = train.FrameFeatures(ids, F.astype(np.float16), ids, coverage=1.0)
    r = train.train_tag("braces", ids, X, top=30, ff=ff, force=True)
    assert r["metrics"]["features"] == "frames" and r["metrics"]["gate"] == "ok"
    meta = json.loads((scorer.MODELS / f"{scorer.tag_key('braces')}.json").read_text())
    assert np.load(scorer.MODELS / f"{scorer.tag_key('braces')}.npz")["w"].shape == (X.shape[1],)   # covers only: no frame columns
    assert meta["weight_source"].endswith("+cover-cap") and meta["weight"] < 0.3         # covers alone know almost nothing here
    scorer.write_manifest(["braces"])
    assert json.loads((scorer.MODELS / "manifest.json").read_text())["tags"] == []        # so it is dropped from scoring
