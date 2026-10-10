"""Push notifications via ntfy (https://ntfy.sh, or your own server). Off unless NTFY_URL is set. Same settings as ttpull:

  NTFY_URL       e.g. https://ntfy.example.com/autotagtt-<long-random-string>   (the topic is the last path part)
  NTFY_TOKEN     optional access token (self-hosted with auth)
  NOTIFY_EVENTS  comma list of: done, failed, test   (default: done,failed)

Privacy: messages carry counts only, never tag names, video IDs, file names or paths, and failures to send are logged
without the URL (the topic acts as a password)."""
import json, os, time, urllib.request
from urllib.parse import urlparse

_recent = {}


def _cfg():
    return (os.environ.get("NTFY_URL", "").strip(), os.environ.get("NTFY_TOKEN", "").strip(),
            {e.strip() for e in os.environ.get("NOTIFY_EVENTS", "done,failed").split(",") if e.strip()})


def enabled():
    return bool(_cfg()[0])


def notify(event, title, message, priority=3, tags=(), dedupe_key=None, dedupe_s=0):
    """Returns True if sent. Never raises: a notification problem must never affect a run."""
    url, token, events = _cfg()
    if not url or (event != "test" and event not in events):
        return False
    if dedupe_key:
        if time.time() - _recent.get(dedupe_key, 0) < dedupe_s:
            return False
        _recent[dedupe_key] = time.time()
    try:
        u = urlparse(url)
        topic = u.path.strip("/")
        if not u.scheme or not u.netloc or not topic:
            raise ValueError("NTFY_URL needs a topic")
        body = json.dumps({"topic": topic, "title": title, "message": message, "priority": priority, "tags": list(tags)}).encode()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(f"{u.scheme}://{u.netloc}/", data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return 200 <= r.status < 300
    except Exception as e:
        print("notify: could not send (%s)" % type(e).__name__, flush=True)     # no URL, no details
        return False


def run_finished(st):
    """Called when a job ends. Uses only the numbers from the run summary."""
    if st.get("phase") == "failed":
        notify("failed", "autotagTT: run failed", "The AI tagger run did not finish. Check the Run panel in Starplayer.",
               priority=4, tags=["warning"], dedupe_key="failed", dedupe_s=6 * 3600)
        return
    res = st.get("result") or {}
    ok = sum(1 for m in res.values() if m.get("gate") == "ok")
    weak = sum(1 for m in res.values() if m.get("gate") == "weak")
    pending = sum(m.get("pending") or 0 for m in res.values())
    ai = sum(m.get("ai_count") or 0 for m in res.values())
    kind = "Nightly run" if st.get("reason") == "nightly" else "Run"
    notify("done", "autotagTT: run finished",
           f"{kind} complete. {ok} tag{'s' if ok != 1 else ''} usable" + (f", {weak} too weak" if weak else "") +
           f". {pending:,} suggestion{'s' if pending != 1 else ''} to review, {ai:,} AI-tagged.",
           priority=3, tags=["white_check_mark"])
