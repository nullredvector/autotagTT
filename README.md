# autotagTT

> **Privacy:** this repository contains code only. Archive content, covers, embeddings, suggestions, feedback and model
> caches live in the archive's `data/.appdata/autotag/` folder on your own server and must never be committed (see `.gitignore`).
> Nothing is uploaded anywhere; the only network access is the one-time model-weight download.

Local batch job that suggests tags (first: `braces`) for the Starplayer archive from cover images.
All processing is local; nothing is uploaded. It only **reads** `stars.json` / archive files and
**writes** under `data/.appdata/autotag/` (owner-only permissions, atomic writes, `.bak` of previous output).
It never edits groups: users accept/reject in Starplayer.

```
python -m autotagtt embed   [--limit N]   # SigLIP fp16 cover embeddings, sharded + resumable
python -m autotagtt train                 # grouped CV + ranked suggestions -> autotag/autotag.json
python -m autotagtt run                   # embed + train
python -m autotagtt montage               # contact sheets of top suggestions for eyeballing
```

## Reading the numbers (positive-unlabeled caveat)
- Positives = the tag's group (+ user-accepted). Everything else is a *provisional* negative, not a verified one.
  Likely many are real unlabeled positives, so every "precision" here is a **lower bound**, and
  `known_precision@K` only measures how often held-out *known* positives rank in the top K.
- CV is 5-fold `StratifiedGroupKFold`; near-duplicate images (cosine > 0.97) are kept in the same fold.
- The meaningful metric is precision of suggestions among those the user actually reviews (`reviewed_*` counts).
- `score` = out-of-fold logistic-regression decision value (log-odds on class-balanced data). Ranking aid, **not** a probability.
- Rejections are re-trained as confirmed negatives (weight 5); accepts as positives.

## Tests
`pip install numpy scikit-learn pillow pytest && pytest` — synthetic vectors/fake IDs only; no GPU, no archive.
Real SigLIP inference has not been run yet.

## Review decisions
`accepted` → positive label, `rejected` → confirmed negative (weight 5), `unsure` → **no label**: the video is set aside in
Starplayer, training ignores it, and it stays hidden from Pending until moved back.

## Multiple tags and AI tags
- `data/.appdata/autotag_tags.json` (`{"enabled": [group names]}`) lists the groups to learn. Starplayer's "Tags…" picker on the
  review page writes it; default is `braces`. One embedding pass serves every tag; each tag only costs a small classifier.
- Per tag: skipped below 30 positives, and "weak" (no suggestions/AI tags) if cross-validated AUC < 0.80.
- Logs anonymise tag names (`tag-xxxxxx`); only `braces` is printed by name.
- `ai_tags.json` holds model labels **separate from human groups**: `{"videos": {id: [[tag, score], ...]}}`. A video is
  AI-tagged above a per-tag cutoff: the point where precision of the latest 50 reviewed decisions (walking down the ranking) stays
  >= 80%. With under 30 reviews for a tag, only its top 100 are tagged ("unvalidated").
  Starplayer hides an AI tag once the video is in the same-named group or you accepted/rejected it.

## Deploying
CI (`.github/workflows/build.yml`) runs the tests and publishes `ghcr.io/<owner>/autotagtt:latest` on every push to `main`.
New GHCR packages start private: make the package public (Package settings) or `docker login ghcr.io` on the server.
Add the service from `docker-compose.yml` to your stack with `ARCHIVE_PATH` set; Watchtower updates it from `:latest`.
To disable: stop the service (Starplayer then shows "worker offline"; existing suggestions stay). To roll back: pin the
previous `:<sha>` tag.

## Notifications (ntfy)
Same settings as ttpull; off unless `NTFY_URL` is set on the worker (`NTFY_URL=https://<your ntfy>/<long-random-topic>`,
optional `NTFY_TOKEN`, `NOTIFY_EVENTS=done,failed`). A message is sent when a run finishes or fails. Messages contain counts
only (tags usable/too weak, suggestions to review, AI-tagged), never tag names, video IDs or paths. Use your own ntfy server if
you don't want even those counts to reach a public one. Test from the Run panel ("Send test") or
`docker exec <worker> python -m autotagtt notify-test`.

## Operations

### What a nightly run does now
1. Embeds only new covers (and, if frame analysis is on, frames for new videos within the time budget).
2. Duplicate grouping is computed once per run, not once per tag.
3. Per tag, the saved scores + weights are reused when the labels (group, accepted, rejected, not-sure), the model and the feature set are unchanged;
   new videos are scored with the saved weights (no retrain). A tag retrains when its labels change, when more than
   max(200, 5%) videos were added, or when its cache is older than 14 days. A quiet night finishes in about a minute and sends no notification.
4. The job runs at low CPU priority with `AUTOTAGTT_THREADS` threads (default 8), so other services keep their CPU.

Per-tag caches live in `data/.appdata/autotag/cache/<hash>/` (hashed names, owner-only): delete the folder to force full retraining.

### Frame analysis (Milestone 5), controlled from Starplayer's Run panel
- **Off** (default). **Trial** reads frames for a sample (a few thousand videos: positives of each enabled tag plus random others) and shows,
  per tag, cross-validated accuracy with and without frames; real outputs are unchanged. **Full** reads every video, in runs of up to N minutes
  (resumes where it stopped; newest videos first). Switching back to Off stops it; cached frame embeddings are kept.
- Frames are used for a tag only once 95% of videos are covered and only if they beat cover-only by at least 0.01 average precision in cross-validation.
- Optional lower-face crops (OpenCV Haar cascade) are a third candidate feature set.
- Frames are extracted with FFmpeg into a private temp dir per batch and deleted right after embedding. Only float16 embeddings are stored
  (`emb/<model>/frames/`); videos that yield no frames twice are skipped from then on.

### Update, disable, roll back
- **Update:** push to `main`; CI publishes the image and Watchtower restarts the worker (do not push while a run is in progress: the run is lost and restarts from its last cached step).
- **Disable the feature:** stop the worker container (Starplayer shows "worker offline"; existing suggestions and AI tags stay). To remove AI tags, delete `autotag/ai_tags.json`.
- **Roll back:** pin the previous `:<sha>` image tag; `.bak` copies of `autotag.json` and `ai_tags.json` are kept; delete `autotag/cache/` if scores look wrong.
- **Recover from a crash:** nothing needs cleaning up. Writes are atomic, finished tags and frame batches are already cached, and the next run continues from there.
