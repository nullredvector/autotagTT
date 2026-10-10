"""Build a synthetic kashibox archive (fake IDs, generated images/videos) for the integration harness.
usage: make_scratch_archive.py <out_dir> <kashibox_checkout>   -- never reads real archive data."""
import base64, gzip, json, subprocess, shutil, sys
from pathlib import Path
from PIL import Image, ImageDraw
import imageio_ffmpeg

out, mtt = Path(sys.argv[1]), Path(sys.argv[2])
app = out / "data/.appdata"
(out / "data/Likes/covers").mkdir(parents=True, exist_ok=True)
(out / "data/Likes/videos").mkdir(parents=True, exist_ok=True)
app.mkdir(parents=True, exist_ok=True)
ids = [str(7000000000000000000 + i) for i in range(1, 41)]
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
for n, v in enumerate(ids):
    hue = (n * 37) % 255
    im = Image.new("RGB", (270, 480), (hue, 90, 255 - hue))
    ImageDraw.Draw(im).text((20, 20), f"fake {n+1}", fill="white")
    im.save(out / f"data/Likes/covers/{v}.jpg")
    if n < 16:  # a few real playable clips; the rest have covers only
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=270x480:rate=15:duration=3",
                        "-pix_fmt", "yuv420p", "-c:v", "libx264", str(out / f"data/Likes/videos/{v}.mp4")], check=True)

def db(name, var, obj):
    b = base64.b64encode(gzip.compress(json.dumps(obj).encode())).decode()
    (app / f"db_{name}.js").write_text(f'window.{var}_base64="{b}";')

db("videos", "dbv", {v: {"authorId": "a1", "createTime": 1700000000 + i, "diggCount": i} for i, v in enumerate(ids)})
db("authors", "dba", {"a1": {"uniqueIds": ["fake_author"], "nicknames": ["Fake Author"]}})
db("texts", "dbvd", {v: f"synthetic caption {i}" for i, v in enumerate(ids)})
db("likes", "db", {"user": {"uid": "fake-user"}, "likes": {"downloaded": ids, "officialList": ids, "lastRun": {"start": 0, "finish": 0}}})
db("bookmarked", "dbb", {"downloaded": [], "officialList": [], "lastRun": {"start": 0, "finish": 0}})
db("following", "dbf", {"authorItems": {}, "officialAuthorList": [], "started": [], "notInterested": [], "lastRun": {}})
(app / "stars.json").write_text(json.dumps({"stars": {v: {"id": v, "coverSrc": f"data/Likes/covers/{v}.jpg", "authorName": "fake_author", "desc": ""} for v in ids[8:22]}, "levels": {}, "groups": [
    {"id": "g-braces", "name": "Braces", "videoIds": ids[:4]},
    {"id": "g-other", "name": "scenery", "videoIds": ids[4:8]}]}))
(app / "sessions.json").write_text(json.dumps({"sessions": []}))
(app / "autotag").mkdir(exist_ok=True)
(app / "autotag/autotag.json").write_text(json.dumps({
    "_meta": {"model": "synthetic", "generated": "2026-01-01T00:00:00"},
    "scenery": [{"id": v, "score": 1.0, "status": "pending"} for v in ids[22:26]],
    "braces": [{"id": v, "score": round(3.0 - i * 0.2, 4), "status": "pending"} for i, v in enumerate(ids[8:22])]}))
# AI tags: videos 9-22 carry braces + scenery labels; id 5 (already in the 'scenery' human group) must be filtered out
(app / "autotag/ai_tags.json").write_text(json.dumps({"_meta": {"generated": "2026-01-01T00:00:00"}, "videos": {
    **{v: [["braces", round(3.0 - i * 0.2, 4)], ["scenery", round(2.0 - i * 0.1, 4)]] for i, v in enumerate(ids[8:22])},
    ids[4]: [["scenery", 1.0]], ids[0]: [["braces", 1.0]]}}))
shutil.copy(mtt / "Archive.html", out / "Archive.html")
shutil.copy(mtt / "app.js", app / "app.js")
shutil.copy(mtt / "starplayer.js", app / "starplayer.js")
for f in list(app.glob("autotag_feedback.json*")) + list(app.glob("autotag_tags.json*")):
    f.unlink()
print("ok", out)
