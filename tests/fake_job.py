"""Stand-in for `python -m autotagtt run` in the end-to-end harness (no GPU, no archive)."""
import time
print("covers: 100 total, 0 embedded, 100 to do", flush=True)
for n in (25, 50, 75, 100):
    time.sleep(2)
    print(f"  {n}/100  60 img/s  eta 0.0 min  vram 1.0 GB", flush=True)
print("[tag-abc123] embedded=100 positives=40 reviewed_rejected=0", flush=True)
time.sleep(2)
