"""Build the page's videos and wipe-slider / comparison stills.

    python3 tools/build_media.py /tmp/cogs_render_hr [--stills-only]   # renders from tools/render_cogs.py --image-scale 1.0

Inputs
  <render>/<Seq>/{gt,full,human,object,scene}/NNNNN.jpg   CoGS re-renders of the paper's final HOSNeRF models
                                                          (gt = the dataset frame of the same camera, so pairs are pixel-aligned)
  ~/jb_ws/SimGS/comparison_hosnerf/<Seq>/NNNNN.png         author comparison strips: GT | CoGS | HUGS | 4DGS
Outputs
  media/teaser.mp4 (= Backpack input | CoGS), media/<seq>_sbs.mp4, media/<seq>_branches.mp4 (composite | human | object | scene)
  img/poster_*.webp, img/slider/<seq>_<frame>_{gt,cogs}.webp (native resolution), img/cmp/<seq>_<frame>_{gt,cogs,hugs,4dgs}.webp
  data/media.json (frame lists + per-pair PSNR), also inlined into index.html
"""
import json, re, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

SITE = Path(__file__).resolve().parents[1]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
STILLS_ONLY = "--stills-only" in sys.argv          # skip the (slow) video encodes
REN = Path(ARGS[0] if ARGS else "/tmp/cogs_render_hr")
CMP = Path.home() / "jb_ws/SimGS/comparison_hosnerf"
SEQS = ["Backpack", "Tennis", "Suitcase", "Playground", "Dance", "Lounge"]
FPS = 24
PANEL = (960, 540)                              # video panel size (downscaled from the ~1277x718 native frames)
GAP = 4
N_SLIDER, MIN_PSNR = 6, 22.0
(SITE / "media").mkdir(exist_ok=True)
for d in ("img/slider", "img/cmp"): (SITE / d).mkdir(parents=True, exist_ok=True)


def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def enc(frames_dir, out, crf, fps=FPS):
    ff(["-framerate", str(fps), "-i", f"{frames_dir}/%05d.png", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)])
    print(f"{out.name:28s} {out.stat().st_size / 1e6:.2f} MB")


def native(seq, kind, f):
    im = Image.open(REN / seq / kind / f"{f:05d}.jpg").convert("RGB")
    return im.crop((0, 0, im.width // 2 * 2, im.height // 2 * 2))       # even size, no resampling


def panel(seq, kind, f):
    return native(seq, kind, f).resize(PANEL, Image.LANCZOS)


def frames(seq):
    return sorted(int(p.stem) for p in (REN / seq / "full").glob("*.jpg"))


def compose(tiles, cols):
    w, h = tiles[0].size; rows = (len(tiles) + cols - 1) // cols
    W = Image.new("RGB", (cols * w + (cols - 1) * GAP, rows * h + (rows - 1) * GAP), (255, 255, 255))
    for i, t in enumerate(tiles): W.paste(t, ((i % cols) * (w + GAP), (i // cols) * (h + GAP)))
    return W


def video(seq, kinds, cols, out, crf):
    if STILLS_ONLY: return
    with tempfile.TemporaryDirectory() as td:
        for j, f in enumerate(frames(seq)):
            compose([panel(seq, k, f) for k in kinds], cols).save(f"{td}/{j:05d}.png", compress_level=1)
        enc(td, out, crf)


def webp(im, path, q=82, w=None):
    if w and im.width > w: im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    im.save(path, "WEBP", quality=q, method=6)


def psnr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(10 * np.log10(255 ** 2 / ((a - b) ** 2).mean()))


def shift(a, b, r=6):
    """Integer shift (dx, dy) of b that best matches a (should be (0, 0) for aligned pairs)."""
    A = np.asarray(a.convert("L"), np.float64)[::2, ::2]; B = np.asarray(b.convert("L"), np.float64)[::2, ::2]
    m = r + 2; best = None
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            e = ((A[m:-m, m:-m] - B[m + dy:B.shape[0] - m + dy, m + dx:B.shape[1] - m + dx]) ** 2).mean()
            if best is None or e < best[0]: best = (e, 2 * dx, 2 * dy)
    return best[1:]


def fg_box(seq, kind, f, thr=245):
    a = np.asarray(Image.open(REN / seq / kind / f"{f:05d}.jpg").convert("L"))
    ys, xs = np.where(a < thr)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, len(xs) / a.size) if len(xs) else None


def pick_slider(seq):
    """Representative pairs: every 5th frame (the evaluation split), object and person visible, not among the
    most motion-blurred input frames, PSNR >= MIN_PSNR; then N_SLIDER spread evenly in time (no PSNR ranking)."""
    cand = []
    for f in [f for f in frames(seq) if f % 5 == 4]:
        hb, ob = fg_box(seq, "human", f), fg_box(seq, "object", f)
        if not hb or not ob or ob[4] < 0.002 or hb[4] < 0.01: continue
        gt = native(seq, "gt", f)
        crop = gt.crop(hb[:4]).convert("L").filter(ImageFilter.FIND_EDGES)
        cand.append((f, float(np.asarray(crop, np.float64).var()), psnr(gt, native(seq, "full", f))))
    sharp_cut = np.percentile([c[1] for c in cand], 30)
    ok = [c for c in cand if c[1] >= sharp_cut and c[2] >= MIN_PSNR]
    idx = np.linspace(0, len(ok) - 1, min(N_SLIDER, len(ok))).round().astype(int)
    return [ok[i] for i in idx], len(cand)


def cmp_panels(path):
    """Split an author comparison strip into its four panels; drop the 24 px label band at the top."""
    im = Image.open(path).convert("RGB"); a = np.asarray(im).astype(np.float32)
    white = np.where(a.mean(axis=(0, 2)) > 250)[0]
    runs, s, p = [], white[0], white[0]
    for x in white[1:]:
        if x != p + 1: runs.append((s, p)); s = x
        p = x
    runs.append((s, p))
    e = [0] + [x for r in runs for x in (r[0], r[1] + 1)] + [im.width]
    return [im.crop((e[i], 24, e[i + 1], im.height)) for i in range(0, len(e), 2)]


for p in list((SITE / "img/slider").glob("*.webp")): p.unlink()
meta = {"fps": FPS, "seqs": {}}
report = []
for seq in SEQS:
    k = seq.lower(); fr = frames(seq); mid = fr[len(fr) // 2]
    if k != "backpack":                                        # Backpack's side-by-side is the teaser (built below)
        video(seq, ["gt", "full"], 2, SITE / f"media/{k}_sbs.mp4", crf=31)
    video(seq, ["full", "human", "object", "scene"], 2, SITE / f"media/{k}_branches.mp4", crf=32)
    webp(compose([panel(seq, "gt", fr[0]), panel(seq, "full", fr[0])], 2), SITE / f"img/poster_{k}_sbs.webp", q=70, w=1280)
    webp(compose([panel(seq, x, mid) for x in ("full", "human", "object", "scene")], 2), SITE / f"img/poster_{k}_branches.webp", q=70, w=1280)
    picks, ncand = pick_slider(seq)
    sl = []
    for f, sharp, ps in picks:
        g, c = native(seq, "gt", f), native(seq, "full", f)
        dx, dy = shift(g, c)
        assert (dx, dy) == (0, 0), (seq, f, dx, dy)
        webp(g, SITE / f"img/slider/{k}_{f:05d}_gt.webp", q=90)
        webp(c, SITE / f"img/slider/{k}_{f:05d}_cogs.webp", q=90)
        sl.append({"f": f, "psnr": round(ps, 2)})
        report.append(f"{seq:10s} frame {f:3d}  {g.width}x{g.height}  PSNR {ps:5.2f} dB  shift {dx},{dy}")
    # some author strips have a placeholder (flat dark) HUGS panel: keep only frames where all four panels are real
    cf = [int(p.stem) for p in sorted((CMP / seq).glob("*.png"))
          if min(np.asarray(x.convert("L"), np.float32).std() for x in cmp_panels(p)) > 12]
    cf = [cf[int(i)] for i in np.linspace(0, len(cf) - 1, min(6, len(cf)))]
    for f in cf:
        for m, im in zip(("gt", "cogs", "hugs", "4dgs"), cmp_panels(CMP / seq / f"{f:05d}.png")):
            webp(im, SITE / f"img/cmp/{k}_{f:05d}_{m}.webp", q=80, w=640)
    meta["seqs"][k] = {"name": seq, "frames": len(fr), "size": list(native(seq, "gt", fr[0]).size),
                       "slider": sl, "slider_candidates": ncand, "cmp": cf}

video("Backpack", ["gt", "full"], 2, SITE / "media/teaser.mp4", crf=30)
webp(compose([panel("Backpack", "gt", 0), panel("Backpack", "full", 0)], 2), SITE / "img/poster_teaser.webp", q=72, w=1600)
(SITE / "data/media.json").write_text(json.dumps(meta, indent=1))
page = SITE / "index.html"                                      # inline it too: no extra request, works from file://
page.write_text(re.sub(r'(<script type="application/json" id="mediajson">).*?(</script>)',
                       lambda m: m.group(1) + json.dumps(meta, separators=(",", ":")) + m.group(2), page.read_text(), flags=re.S))
print("\n".join(report))
