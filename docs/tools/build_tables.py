"""Write the paper's result tables into index.html (between <!--TABLE:name--> markers).

    python3 tools/build_tables.py

Numbers are copied from dp-papers/simgs/sections/{method,experiments}.tex (Tables: HOSNeRF comparison,
NeuMan full-frame comparison, and the ablations). Best / second / third cells are ranked per column.
"""
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]

HOS_SEQ = ["Backpack", "Tennis", "Suitcase", "Playground", "Dance", "Lounge"]
HOS = [  # method, [PSNR, LPIPS] x 6
    ("K-Planes", [19.05, .557, 19.31, .536, 18.64, .602, 17.92, .635, 18.17, .623, 24.21, .453]),
    ("D²NeRF", [20.52, .608, 23.97, .540, 20.99, .645, 21.23, .616, 19.92, .647, 27.13, .509]),
    ("Nerfies", [19.56, .559, 22.12, .443, 19.01, .555, 21.14, .533, 19.37, .524, 25.90, .342]),
    ("HyperNeRF", [19.62, .587, 21.26, .510, 19.41, .607, 21.67, .578, 19.30, .601, 27.25, .332]),
    ("NeuMan", [21.21, .478, 23.17, .442, 20.84, .551, 21.46, .551, 21.19, .490, 28.40, .341]),
    ("4DGS", [24.49, .192, 26.57, .162, 17.98, .460, 24.34, .222, 21.34, .212, 30.50, .067]),
    ("D3DGS", [24.06, .099, 25.09, .125, 17.85, .453, 23.93, .141, 21.07, .117, 26.90, .072]),
    ("E-D3DGS", [24.78, .146, 26.53, .161, 18.05, .461, 24.37, .206, 23.87, .159, 30.04, .086]),
    ("Ex4DGS", [18.07, .433, 17.90, .399, 15.25, .557, 16.36, .535, 17.08, .529, 23.15, .310]),
    ("ExAvatar", [24.15, .107, 23.57, .160, 20.32, .260, 25.30, .129, 23.32, .170, 29.43, .048]),
    ("HOSNeRF", [22.56, .243, 24.15, .320, 21.74, .382, 22.67, .336, 22.63, .248, 27.74, .227]),
    ("CoGS (ours)", [25.38, .094, 26.93, .071, 24.25, .117, 25.62, .116, 25.08, .092, 30.94, .031]),
]
HOS_GAIN = ["+2.4%", "+4.7%", "+1.4%", "+43.5%", "+11.5%", "+55.0%", "+1.3%", "+10.1%", "+5.1%", "+21.4%", "+1.4%", "+35.4%"]

NEU_SEQ = ["Seattle", "Citron", "Parking", "Bike", "Jogging", "Lab"]
NEU = [  # method, [PSNR, SSIM, LPIPS] x 6
    ("NeRF-T", [21.84, .69, .37, 12.33, .49, .65, 21.98, .69, .46, 21.16, .71, .36, 20.63, .53, .49, 20.52, .75, .39]),
    ("HyperNeRF", [16.43, .43, .40, 16.81, .41, .56, 16.04, .38, .62, 17.64, .42, .43, 18.52, .39, .52, 16.75, .51, .23]),
    ("Vid2Avatar", [17.41, .56, .60, 14.32, .62, .65, 21.56, .69, .50, 14.86, .51, .69, 15.04, .41, .70, 13.96, .60, .68]),
    ("NeuMan", [23.99, .78, .26, 24.63, .81, .26, 25.43, .80, .31, 25.55, .83, .23, 22.70, .68, .32, 24.96, .86, .21]),
    ("HUGS", [25.94, .85, .13, 25.54, .86, .15, 26.86, .85, .22, 25.46, .84, .13, 23.75, .78, .22, 26.00, .92, .09]),
    ("CoGS (ours)", [29.18, .89, .07, 28.37, .90, .06, 31.28, .87, .10, 29.40, .91, .05, 25.76, .82, .13, 29.31, .93, .05]),
]
NEU_GAIN = ["+12.49%", "+4.71%", "+46.15%", "+11.08%", "+4.65%", "+60.00%", "+16.46%", "+2.35%", "+54.55%", "+15.07%", "+8.33%",
            "+61.54%", "+8.46%", "+5.13%", "+40.91%", "+12.73%", "+1.09%", "+44.44%"]

ABL = [
    ("Generative completion views", "NeuMan <i>Lab</i>, full frame. Limited real removes body-side evidence from the seed views; Wan completion uses the same real inputs.",
     ["Variant", "PSNR↑", "SSIM↑", "LPIPS↓"],
     [["HUGS", "26.00", "0.920", "0.090"], ["Limited real views only", "26.70", "0.925", "0.079"],
      ["Limited real + Wan", "29.02", "0.933", "0.055"], ["All real training views", "29.31", "0.934", "0.049"]]),
    ("Visibility-aware anchoring", "NeuMan <i>Lab</i>, tight human crop.",
     ["Variant", "PSNR↑", "SSIM↑", "LPIPS↓"],
     [["Full model (CoGS)", "26.78", "0.920", "0.050"], ["w/o visibility anchor", "23.94", "0.876", "0.099"],
      ["uniform anchor", "24.04", "0.879", "0.098"], ["visible-only anchor", "23.97", "0.878", "0.097"]]),
    ("Branch losses and planar primitives", "NeuMan <i>Lab</i>, full frame.",
     ["Variant", "PSNR↑", "SSIM↑", "LPIPS↓"],
     [["Full model (CoGS)", "29.31", "0.934", "0.049"], ["w/o planar primitives", "27.98", "0.924", "0.053"],
      ["w/o human alpha / boundary losses", "25.65", "0.918", "0.060"]]),
    ("Optimisation schedule", "NeuMan <i>Lab</i>, full frame.",
     ["Variant", "PSNR↑", "SSIM↑", "LPIPS↓"],
     [["Full model (CoGS)", "29.31", "0.934", "0.049"], ["joint training from iteration 0", "29.10", "0.933", "0.050"],
      ["w/o human-only warm-up", "29.22", "0.933", "0.050"], ["w/o final full refinement", "27.08", "0.833", "0.121"]]),
    ("Prior initialisation", "HOSNeRF <i>Backpack</i>, full frame.",
     ["Variant", "PSNR↑", "LPIPS↓"],
     [["Full model (CoGS)", "25.38", "0.094"], ["w/o human prior seed", "22.73", "0.179"], ["w/o object seed", "23.45", "1.314"]]),
    ("Object-branch diagnostics", "HOSNeRF, object branch inside the object mask. Alpha-IoU at 0.5; leak = alpha outside the mask.",
     ["Sequence", "Obj. PSNR↑", "Alpha-IoU↑", "Leak↓"],
     [["Backpack", "21.21", "0.898", "0.084"], ["Tennis", "11.96", "0.226", "2.580"], ["Playground", "10.67", "0.124", "5.499"],
      ["Dance", "13.39", "0.100", "7.445"], ["Lounge", "10.74", "0.141", "3.885"], ["Suitcase", "20.88", "0.851", "0.138"]]),
]


def ranks(rows, ncol, lower_better):
    out = []
    for c in range(ncol):
        vals = sorted({r[1][c] for r in rows}, reverse=not lower_better(c))
        out.append({v: i + 1 for i, v in enumerate(vals[:3])})
    return out


def fmt(v, c, metrics):
    m = metrics[c % len(metrics)]
    return f"{v:.2f}" if m == "PSNR" else (f"{v:.3f}" if m == "LPIPS" and len(metrics) == 2 else f"{v:.2f}")


def bench(seqs, metrics, rows, gain, caption):
    nm = len(metrics); ncol = nm * len(seqs); dense = " dense" if nm > 2 else ""
    lower = lambda c: metrics[c % nm] == "LPIPS"
    rk = ranks(rows, ncol, lower)
    h = [f'<div class="tblwrap"><table class="res{dense}"><caption class="sr">{caption}</caption><thead><tr><th rowspan="2">Method</th>']
    h += [f'<th colspan="{nm}" class="grp">{s}</th>' for s in seqs]
    h.append("</tr><tr>" + "".join(f'<th{" class=\"grp\"" if i % nm == 0 else ""}>{m}{"↓" if m == "LPIPS" else "↑"}</th>'
                                   for i, m in enumerate(metrics * len(seqs))) + "</tr></thead><tbody>")
    for name, v in rows:
        cls = ' class="ours"' if name.startswith("CoGS") else ""
        tds = []
        for c, x in enumerate(v):
            r = rk[c].get(x); cc = (["grp"] if c % nm == 0 else []) + ([f"r{r}"] if r else [])
            tds.append(f'<td{" class=" + chr(34) + " ".join(cc) + chr(34) if cc else ""}>{fmt(x, c, metrics)}</td>')
        h.append(f"<tr{cls}><td>{name}</td>{''.join(tds)}</tr>")
    h.append('<tr class="gain"><td>vs. best baseline</td>' + "".join(
        f'<td{" class=" + chr(34) + "grp" + chr(34) if c % nm == 0 else ""}>{g}</td>' for c, g in enumerate(gain)) + "</tr>")
    h.append("</tbody></table></div>")
    return "\n".join(h)


def ablations():
    h = ['<div class="ablgrid">']
    for title, note, head, rows in ABL:
        h.append(f'<div><h4>{title}</h4><div class="tblwrap"><table class="res"><thead><tr>' + "".join(f"<th>{x}</th>" for x in head) + "</tr></thead><tbody>")
        for r in rows:
            cls = ' class="ours"' if r[0].startswith("Full model") else ""
            h.append(f"<tr{cls}>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>")
        h.append(f"</tbody></table></div><p>{note}</p></div>")
    h.append("</div>")
    return "\n".join(h)


blocks = {
    "hosnerf": bench(HOS_SEQ, ["PSNR", "LPIPS"], HOS, HOS_GAIN, "HOSNeRF per-sequence full-frame comparison"),
    "neuman": bench(NEU_SEQ, ["PSNR", "SSIM", "LPIPS"], NEU, NEU_GAIN, "NeuMan full-frame comparison"),
    "ablations": ablations(),
}
p = SITE / "index.html"; html = p.read_text()
for k, v in blocks.items():
    html, n = re.subn(rf"(<!--TABLE:{k}-->).*?(<!--/TABLE:{k}-->)", lambda m: m.group(1) + "\n" + v + "\n" + m.group(2), html, flags=re.S)
    assert n == 1, k
p.write_text(html)
print("tables written:", ", ".join(blocks))
