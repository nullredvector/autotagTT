"""Read-only access to the Starplayer archive layout and user data."""
import base64, gzip, json, os, re
from pathlib import Path

ROOT = Path(os.environ.get("ARCHIVE_DIR", "/archive"))
DATA = ROOT / "data"
APPDATA = DATA / ".appdata"
OUT = APPDATA / "autotag"          # everything we write lives here (never stars.json/sessions.json)


def read_db(name):
    """db_<name>.js is `window.db_<name>_base64="..."` = base64(gzip(json))."""
    txt = (APPDATA / f"db_{name}.js").read_text()
    m = re.search(r'"([A-Za-z0-9+/=]+)"', txt)
    return json.loads(gzip.decompress(base64.b64decode(m.group(1))))


def find_covers():
    """{videoId: cover Path} across Likes, Favorites and Following/<author>."""
    out = {}
    for sub in sorted(DATA.glob("*/covers")) + sorted(DATA.glob("Following/*/covers")):
        for p in sub.iterdir():
            if p.suffix.lower() == ".jpg":
                out.setdefault(p.stem, p)
    return out


def find_videos():
    out = {}
    for sub in sorted(DATA.glob("*/videos")) + sorted(DATA.glob("Following/*/videos")):
        for p in sub.iterdir():
            if p.suffix.lower() == ".mp4":
                out.setdefault(p.stem, p)
    return out


def load_groups():
    """{lowercase name: set(videoIds)} from stars.json (read-only)."""
    sj = json.loads((APPDATA / "stars.json").read_text())
    return {g["name"].strip().lower(): set(g["videoIds"]) for g in sj.get("groups", [])}


def write_json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")))
    os.replace(tmp, path)
