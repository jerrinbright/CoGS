"""Wipe-slider pairs: hand-picked frames at the models' training resolution (dataset image_scale 0.5).

    python3 tools/build_slider.py /tmp/cogs_render_lr     # renders from: render_cogs.py --image-scale 0.5 --frames 4,9,14,...

Selection (Nov 2026): every 5th frame scored by PSNR and LPIPS (alex), restricted to frames where the person is large,
the object is in view and the input is sharp; the top candidates were then inspected by eye and frames with visible
artifacts on the person or object dropped. GT and CoGS come from the same dataset camera; zero shift is asserted.
"""
import json, re, sys
from pathlib import Path

import numpy as np
from PIL import Image

SITE = Path(__file__).resolve().parents[1]
REN = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cogs_render_lr")
PICKS = {"Backpack": [69, 74, 229], "Tennis": [294, 264], "Suitcase": [249, 254, 99],
         "Playground": [314, 219, 89], "Dance": [374, 369, 69, 364], "Lounge": [284, 279, 114]}
sys.path.insert(0, str(Path(__file__).parent))


def native(seq, kind, f):
    im = Image.open(REN / seq / kind / f"{f:05d}.jpg").convert("RGB")
    return im.crop((0, 0, im.width // 2 * 2, im.height // 2 * 2))


def shift(a, b, r=4):
    A = np.asarray(a.convert("L"), np.float64); B = np.asarray(b.convert("L"), np.float64); m = r + 2; best = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            e = ((A[m:-m, m:-m] - B[m + dy:B.shape[0] - m + dy, m + dx:B.shape[1] - m + dx]) ** 2).mean()
            if best is None or e < best[0]: best = (e, dx, dy)
    return best[1:]


out = SITE / "img/slider"
for p in out.glob("*.webp"): p.unlink()
meta_p = SITE / "data/media.json"; meta = json.loads(meta_p.read_text())
for seq, frames in PICKS.items():
    k = seq.lower(); sl = []
    for f in frames:
        g, c = native(seq, "gt", f), native(seq, "full", f)
        assert shift(g, c) == (0, 0), (seq, f)
        ps = 10 * np.log10(255 ** 2 / ((np.asarray(g, np.float64) - np.asarray(c, np.float64)) ** 2).mean())
        g.save(out / f"{k}_{f:05d}_gt.webp", "WEBP", quality=92, method=6)
        c.save(out / f"{k}_{f:05d}_cogs.webp", "WEBP", quality=92, method=6)
        sl.append({"f": f, "psnr": round(float(ps), 2)})
        print(f"{seq:10s} frame {f:3d} {g.width}x{g.height} PSNR {ps:5.2f} dB")
    meta["seqs"][k]["slider"] = sl; meta["seqs"][k]["size"] = list(g.size)
    meta["seqs"][k].pop("slider_candidates", None)
meta_p.write_text(json.dumps(meta, indent=1))
page = SITE / "index.html"
page.write_text(re.sub(r'(<script type="application/json" id="mediajson">).*?(</script>)',
                       lambda m: m.group(1) + json.dumps(meta, separators=(",", ":")) + m.group(2), page.read_text(), flags=re.S))
