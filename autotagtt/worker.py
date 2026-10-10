"""Always-on worker: waits for a run request from Starplayer (or the nightly schedule), then runs the fixed
job `python -m autotagtt run` (embed new covers, train every enabled tag) in a subprocess and reports status.

Files (all under data/.appdata/autotag/, owner-only):
  run_request.json  written by Starplayer  {"id": str, "ts": epoch}
  schedule.json     written by Starplayer  {"enabled": bool, "hour": 0-23}
  status.json       written here           state/phase/heartbeat/last result (read by Starplayer)
The command is fixed in this file; nothing in the request files can change what is executed."""
import json, os, subprocess, sys, threading, time
from .archive import OUT, write_json_atomic
from . import notify

REQUEST = OUT / "run_request.json"
SCHEDULE = OUT / "schedule.json"
NOTIFY_TEST = OUT / "notify_test.json"   # written by Starplayer's "Send test notification" button
STATUS = OUT / "status.json"
HEARTBEAT = OUT / "heartbeat.json"   # written by its own thread, so a long silent job never looks like a dead worker
LOG = OUT / "last_run.log"
THREADS = os.environ.get("AUTOTAGTT_THREADS", "8")     # CPU threads for the job (the box has other services)
NICE = int(os.environ.get("AUTOTAGTT_NICE", "10"))     # run the job at lower CPU priority
POLL = float(os.environ.get("AUTOTAGTT_POLL", "5"))
JOB_TIMEOUT = float(os.environ.get("AUTOTAGTT_TIMEOUT", str(6 * 3600)))
JOB_CMD = [sys.executable, "-m", "autotagtt", "run"]
if os.environ.get("AUTOTAGTT_TEST_JOB"):            # tests only: swap in a fake job
    JOB_CMD = [sys.executable, os.environ["AUTOTAGTT_TEST_JOB"]]


def _read(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_status():
    s = _read(STATUS)
    return s if isinstance(s, dict) else {}


def save_status(st):
    st["heartbeat"] = time.time()
    write_json_atomic(STATUS, st)


def frames_info():
    """Frame-analysis progress and trial comparison (tag names appear only here, for Starplayer's UI)."""
    out = {}
    stats = _read(OUT / "frames_stats.json")
    if isinstance(stats, dict):
        out["stats"] = {k: stats.get(k) for k in ("done", "total", "rate", "mode", "ts")}
    trial = _read(OUT / "frames_trial.json")
    if isinstance(trial, dict):
        out["trial"] = trial
    return out


def summarize():
    """Per-tag result from autotag.json. Tag names are only for Starplayer's UI; they are never logged."""
    d = _read(OUT / "autotag.json") or {}
    out = {}
    for tag, m in (d.get("_meta", {}).get("metrics") or {}).items():
        out[tag] = {k: m.get(k) for k in ("positives", "auc", "gate", "ai_count", "ai_mode", "reviewed_accepted", "reviewed_rejected")}
        out[tag]["pending"] = len(d.get(tag) or [])
    return out


def heartbeat_loop(interval=10.0):
    while True:
        try:
            write_json_atomic(HEARTBEAT, {"t": time.time()})
        except Exception:
            pass
        time.sleep(interval)


def track_progress(st, line, total):
    """Tag counter from the job's anonymous log lines: 'tag N of M' while training."""
    if "embedded=" in line:
        st["_started"] = st.get("_started", 0) + 1
    if line.startswith("ai tags:") or "skipped" in line:
        st["_finished"] = st.get("_finished", 0) + 1
    if st.get("phase") == "training" and total:
        st["detail"] = "tag %d of %d" % (min(st.get("_started", 1) or 1, total), total)


def phase_from_line(line, current):
    if line.startswith("covers:"):
        return "embedding"
    if line.startswith("frames:"):
        return "frames"
    if "embedded=" in line:
        return "training"
    return current


def run_job(st, reason, request_id=None):
    """Runs the fixed job to completion; always leaves status in a final state."""
    try:
        from .archive import enabled_tags
        total = len(enabled_tags())
    except Exception:
        total = 0
    st.update(state="running", phase="starting", reason=reason, started=time.time(), error=None, detail="", _started=0, _finished=0)
    if request_id:
        st["last_request_id"] = request_id
    save_status(st)
    t0 = time.time()
    code = None
    try:
        with open(LOG, "w") as log:
            os.chmod(LOG, 0o600)
            env = dict(os.environ, OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, AUTOTAGTT_THREADS=THREADS)
            p = subprocess.Popen(JOB_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
                                 preexec_fn=(lambda: os.nice(NICE)) if NICE else None)
            last = 0.0
            for line in p.stdout:
                log.write(line); log.flush()
                line = line.rstrip()
                st["phase"] = phase_from_line(line, st["phase"])
                if "img/s" in line or line.startswith("  frames "):
                    st["detail"] = line.strip()      # progress lines carry no tag names
                track_progress(st, line, total)
                if time.time() - last > 3:
                    save_status(st); last = time.time()
                if time.time() - t0 > JOB_TIMEOUT:
                    p.kill()
                    raise TimeoutError
            code = p.wait()
        if code != 0:
            raise RuntimeError(code)
        st.update(state="idle", phase="done", error=None, last_ok=time.time(), result=summarize(),
                  run_summary=_read(OUT / "run_summary.json"))
    except TimeoutError:
        st.update(state="idle", phase="failed", error="timed out")
    except Exception:
        st.update(state="idle", phase="failed", error="job failed (exit code %s); see last_run.log" % code)
    st.pop("_started", None); st.pop("_finished", None)
    st.update(finished=time.time(), detail="", frames=frames_info())
    save_status(st)
    try:
        notify.run_finished(st)
    except Exception:
        pass


def due_schedule(st, now=None):
    """True once per local calendar day, at/after the configured hour."""
    sc = _read(SCHEDULE)
    if not isinstance(sc, dict) or sc.get("enabled") is not True:
        return False
    hour = sc.get("hour")
    if not isinstance(hour, int) or not 0 <= hour <= 23:
        return False
    lt = time.localtime(now or time.time())
    day = time.strftime("%Y-%m-%d", lt)
    return lt.tm_hour >= hour and st.get("last_scheduled_day") != day


def handle_test_notification(st):
    req = _read(NOTIFY_TEST)
    rid = req.get("id") if isinstance(req, dict) and isinstance(req.get("id"), str) else None
    if rid and rid != (st.get("notify_test") or {}).get("id"):
        ok = notify.notify("test", "autotagTT: test notification", "If you can read this, notifications are working.", tags=["bell"])
        st["notify_test"] = {"id": rid, "ok": bool(ok), "ts": time.time()}
        save_status(st)


def tick(st, now=None):
    """One poll step. Returns True if a job ran."""
    handle_test_notification(st)
    st["notifications"] = notify.enabled()
    req = _read(REQUEST)
    rid = req.get("id") if isinstance(req, dict) and isinstance(req.get("id"), str) else None
    if rid and rid != st.get("last_request_id"):
        run_job(st, "manual", rid)
        return True
    if due_schedule(st, now):
        st["last_scheduled_day"] = time.strftime("%Y-%m-%d", time.localtime(now or time.time()))
        run_job(st, "nightly")
        return True
    return False


def main():
    os.umask(0o077)
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = load_status()
    if st.get("state") == "running":                  # the container died mid-job last time
        st.update(state="idle", phase="failed", error="interrupted (worker restarted)", detail="")
    st.setdefault("state", "idle")
    save_status(st)
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    print("worker ready", flush=True)
    beat = 0.0
    while True:
        try:
            ran = tick(st)
        except Exception as e:                         # never die; report and keep polling
            print("worker error:", type(e).__name__, flush=True)
            ran = False
        if not ran and time.time() - beat > 10:
            save_status(st); beat = time.time()
        time.sleep(POLL)
