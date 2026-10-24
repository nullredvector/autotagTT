"""HTTP scorer, inside the compose network only (never published): POST /score?id=<video id> with a raw cover image.
Answers {"score": 0..1, "tags": [[name, value], ...], "model": "<id>"}; GET /health. One request is processed at a time.
Every score is appended to scout_scores.jsonl (rotated) so a later decision can be joined to the features that produced it.
Nothing is sent anywhere; no image, id or tag name is ever logged to the console."""
import io, json, os, re, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import numpy as np
from PIL import Image
from .archive import OUT, DEFAULT_MODEL, DEFAULT_PRETRAINED
from .scorer import ModelSet

MAX_BYTES = 2 * 1024 * 1024
DRAIN_BYTES = 8 * 1024 * 1024          # how much of an oversize body is read and discarded before answering 413
MAX_PIXELS = 16_000_000                  # larger images are refused before decoding
FORMATS = {"JPEG", "PNG", "WEBP"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
LOG_KEEP = 20_000
LOG_SLACK = 1_000                        # rewrite when the file grows this far past LOG_KEEP
SCORES = OUT / "scout_scores.jsonl"
Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2


class ScoresLog:
    """Append-only JSON lines, owner-only, trimmed to the newest `keep` lines."""

    def __init__(self, path=None, keep=LOG_KEEP, slack=LOG_SLACK):
        self.path, self.keep, self.slack = path or SCORES, keep, slack
        self.lock = threading.Lock()
        self.count = self._count()

    def _count(self):
        try:
            with open(self.path, "rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def append(self, rec):
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(line)
            self.count += 1
            if self.count > self.keep + self.slack:
                self._trim()

    def _trim(self):
        with open(self.path, "r") as f:
            lines = f.readlines()[-self.keep:]
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.writelines(lines)
        os.replace(tmp, self.path)
        self.count = len(lines)


class Service:
    def __init__(self, models=None, scores_log=None):
        self.models = models or ModelSet()
        self.log = scores_log or ScoresLog()
        self.embedder = None
        self.ready = False
        self.infer_lock = threading.Lock()          # one request at a time

    def set_embedder(self, embedder):
        self.embedder = embedder
        self.ready = True

    def health(self):
        self.models.maybe_reload()
        return {"ok": True, "ready": self.ready, "model": self.models.model_id, "tags": len(self.models.tags)}

    def score(self, vid, body):
        """-> (http status, json-able dict)"""
        if not self.ready:
            return 503, {"error": "loading"}
        if not vid or not ID_RE.match(vid):
            return 400, {"error": "bad id"}
        if not body:
            return 400, {"error": "empty body"}
        try:
            im = Image.open(io.BytesIO(body))
            if im.format not in FORMATS:
                return 400, {"error": "unsupported image type"}
            if im.size[0] * im.size[1] > MAX_PIXELS:
                return 413, {"error": "image too large"}
            img = im.convert("RGB")                   # decodes the pixels; truncated or corrupt files fail here
        except Exception:
            return 400, {"error": "cannot decode image"}
        self.models.maybe_reload()
        with self.infer_lock:
            emb = np.asarray(self.embedder.encode([img])[0], np.float32)
        score, tags, probs = self.models.score(emb)
        try:
            self.log.append({"id": vid, "ts": int(time.time()), "p": probs, "score": score})
        except Exception:
            pass                                       # logging must never fail a score
        return 200, {"score": score, "tags": tags, "model": self.models.model_id or "none"}


def make_handler(service):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):                     # no access log: it would carry ids
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/health":
                self._send(200, service.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/score":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send(411, {"error": "Content-Length required"})
                return
            if n > MAX_BYTES:
                left = min(n, DRAIN_BYTES)             # read (and discard) a bounded amount so the client gets its 413, then hang up
                while left > 0:
                    chunk = self.rfile.read(min(65536, left))
                    if not chunk:
                        break
                    left -= len(chunk)
                self.close_connection = True
                self._send(413, {"error": "body too large"})
                return
            body = self.rfile.read(n)
            vid = (parse_qs(u.query).get("id") or [""])[0]
            t0 = time.time()
            code, obj = service.score(vid, body)
            print(f"score {code} {int((time.time() - t0) * 1000)} ms", flush=True)
            self._send(code, obj)
    return H


def main(host="0.0.0.0", port=None):
    os.umask(0o077)
    port = int(port or os.environ.get("AUTOTAGTT_SERVE_PORT", "8090"))
    svc = Service()
    try:
        svc.models.load()
    except Exception:
        pass

    def load_model():
        while True:                                    # a failed load (GPU busy, download hiccup) is retried; /score answers 503 meanwhile
            try:
                from .embed import ImageEmbedder
                svc.set_embedder(ImageEmbedder(DEFAULT_MODEL, DEFAULT_PRETRAINED))
                print("scorer ready", flush=True)
                return
            except Exception as e:
                print(f"scorer: model load failed ({type(e).__name__}); retrying in 60 s", flush=True)
                time.sleep(60)

    threading.Thread(target=load_model, daemon=True).start()

    def watch():                                       # pick up a new model after a nightly run, no restart needed
        while True:
            time.sleep(30)
            svc.models.maybe_reload()
    threading.Thread(target=watch, daemon=True).start()
    ThreadingHTTPServer((host, port), make_handler(svc)).serve_forever()
