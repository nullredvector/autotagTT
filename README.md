# autotagTT

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
