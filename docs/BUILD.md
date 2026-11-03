# CoGS project page

Static site (GitHub Pages ready, served from `docs/`): `index.html` + `img/` (paper figures, stills) + `media/` (videos) + `data/media.json` (frame lists, also inlined in the page).

## Sources
- Text, figures and every number in the tables: the paper, `~/jb_ws/dp-papers/simgs/` (abstract, method, experiments; `Figures/`).
- Videos, branch renders and wipe-slider pairs: re-rendered from the paper's final HOSNeRF models,
  `~/jb_ws/SimGS/hosnerf_six_stage_rerun_20260520_final_16500_to_19000/hosnerf_hoi/<Seq>/hugs_mmlp/*object_final_polish_16500_to_19000/*/ckpt/{human,object,scene}_final.pth`
  (their saved evaluations reproduce the paper's HOSNeRF numbers), at the dataset's native resolution (`--image-scale 1.0`; training used 0.5).
  `gt` and `full` come from the same dataset camera, so wipe pairs are pixel-aligned; `build_media.py` asserts zero shift and PSNR >= 22 dB.
- "Against baselines" stills: `~/jb_ws/SimGS/comparison_hosnerf/<Seq>/` (author strips GT | CoGS | HUGS | 4DGS; the label band is cropped off).
- Nothing from `~/jb_ws/SimGS/exp/v2` is used.

## Rebuild (run from `docs/`)
```bash
python3 tools/build_figures.py                      # paper figures -> img/fig_*.webp (needs pdftoppm)
python3 tools/build_tables.py                       # paper tables -> index.html (between <!--TABLE:...--> markers)
# re-render (GPU, ~2-3 min per sequence; read-only on SimGS)
cd ~/jb_ws/SimGS && for s in Backpack Tennis Suitcase Playground Dance Lounge; do
  PYTHONDONTWRITEBYTECODE=1 ~/miniforge3/envs/hugs/bin/python ~/jb_ws/cogs-site/docs/tools/render_cogs.py \
    --seq $s --out /tmp/cogs_render_hr --image-scale 1.0; done; cd -
python3 tools/build_media.py /tmp/cogs_render_hr   # videos, posters, slider pairs, comparison stills, media.json
python3 tools/make_artifact.py                      # build/artifact.html for the private claude.ai preview
```

Notes
- `render_cogs.py` mirrors `GaussianTrainer.validate()`: black background for the composite and `human_gs.clear_cache()`
  before every frame (without it HUGS_MMLP reuses stale pose features and the person renders blurred and misplaced).
- Suitcase's original preprocessing folder (`hosnerf_suitcase_sam3_humanattached_head_masked`) is not on this machine;
  `render_cogs.py` substitutes `hosnerf_suitcase_sixstage`, which reproduces the run's saved evaluation renders (PSNR within 0.05 dB).

Preview: tunnel `https://markers-fabric-throws-theater.trycloudflare.com/cogs/index.html` (symlink `~/jb_ws/data4d/sports_site/cogs -> docs`).
Local: `python3 -m http.server 8000` in this folder, then open http://localhost:8000.
