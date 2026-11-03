"""Convert the CoGS paper figures (dp-papers/simgs/Figures) to web images in img/.

    python3 tools/build_figures.py

Needs pdftoppm (poppler-utils) and Pillow with WebP support. Run from docs/.
"""
import subprocess, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

SITE = Path(__file__).resolve().parents[1]
FIG = Path.home() / "jb_ws/dp-papers/simgs/Figures"
OUT = SITE / "img"; OUT.mkdir(exist_ok=True)


def autocrop(im, pad=12, thr=246):
    """Trim near-white margins."""
    a = np.asarray(im.convert("L"))
    ys, xs = np.where(a < thr)
    if not len(xs): return im
    x0, x1, y0, y1 = max(xs.min() - pad, 0), min(xs.max() + pad, im.width), max(ys.min() - pad, 0), min(ys.max() + pad, im.height)
    return im.crop((x0, y0, x1, y1))


def pdf(name, dpi):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(FIG / name), f"{d}/p"], check=True)
        return Image.open(f"{d}/p.png").convert("RGB")


def save(im, name, width, q=84):
    if im.width > width: im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    p = OUT / name; im.save(p, "WEBP", quality=q, method=6)
    print(f"{name:28s} {im.width}x{im.height} {p.stat().st_size // 1024} KB")
    return im


# Method overview (Fig. 2 of the paper)
save(autocrop(pdf("hugs.pdf", 220)), "fig_overview.webp", 2000)
# NeuMan qualitative comparison (Fig. simgs_qual), trimmed like the paper's \includegraphics trim
save(autocrop(Image.open(FIG / "simgs_qual.png").convert("RGB")), "fig_neuman_qual.webp", 1800, q=86)
# Object-level qualitative comparison
save(autocrop(pdf("object_qual.svg.pdf", 200)), "fig_object_qual.webp", 1400, q=86)
# Visibility-aware anchoring
save(autocrop(Image.open(FIG / "visibility_anchor.png").convert("RGB")), "fig_visibility.webp", 1400)
# Planar-primitive scene regulariser (letter page with the diagram in the middle)
save(autocrop(pdf("planar_prim.svg.pdf", 300)), "fig_planar.webp", 1400)
