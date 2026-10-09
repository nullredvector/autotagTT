from PIL import Image, ImageDraw
from .archive import find_covers


def make_montage(ids, out, scores=None, cols=10, w=180):
    covers = find_covers()
    h = int(w * 16 / 9)
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    for n, v in enumerate(ids):
        try:
            im = Image.open(covers[v]).convert("RGB").resize((w, h))
        except Exception:
            continue
        if scores:
            ImageDraw.Draw(im).text((4, 4), f"{n+1}: {scores[n]:.2f}", fill="yellow")
        sheet.paste(im, ((n % cols) * w, (n // cols) * h))
    sheet.save(out, quality=85)
