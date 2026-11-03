"""Re-render the paper's final CoGS HOSNeRF models (six-stage runs, 2026-05-20), composite and per branch.

Runs inside the SimGS training environment (it builds a GaussianTrainer in eval mode, exactly as
SimGS/scripts/render_hoi_decomposition.py does). Read-only on SimGS: checkpoints and data are only read,
the trainer's log directory is redirected to the output folder.

    cd ~/jb_ws/SimGS && PYTHONDONTWRITEBYTECODE=1 ~/miniforge3/envs/hugs/bin/python \
        ~/jb_ws/cogs-site/docs/tools/render_cogs.py --seq Backpack --out /tmp/cogs_render [--image-scale 1.0] [--frames 4,9,14]

Writes <out>/<seq>/{gt,full,human,object,scene}/<frame>.jpg for every frame of the sequence
(full = the composite the paper evaluates; human/object/scene = branch-only renders on white).
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

SIMGS = Path.home() / "jb_ws/SimGS"
sys.path.insert(0, str(SIMGS))
from hugs.cfg.config import cfg as default_cfg            # noqa: E402
from hugs.renderer.gs_renderer import render_human_scene  # noqa: E402
from hugs.trainer import GaussianTrainer                  # noqa: E402
from hugs.trainer.gs_trainer import get_train_dataset     # noqa: E402
from hugs.utils.general import safe_state                 # noqa: E402

RUNS = SIMGS / "hosnerf_six_stage_rerun_20260520_final_16500_to_19000/hosnerf_hoi"
REMOTE = "/mnt/jerrin_vm_1_volume/ml-hugs/exp/data"
# preprocessing folders that only exist under another name locally (checked by re-render PSNR, see BUILD.md)
ALIAS = {"hosnerf_suitcase_sam3_humanattached_head_masked": "hosnerf_suitcase_sixstage"}


def final_run(seq):
    return Path(sorted(glob.glob(str(RUNS / seq / "hugs_mmlp/*object_final_polish_16500_to_19000/*/")))[-1])


def to_img(t):
    return Image.fromarray((t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8))



@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-scale", type=float, default=None, help="override dataset.image_scale (paper runs: 0.5)")
    ap.add_argument("--frames", default="", help="comma-separated frame indices (default: all)")
    a = ap.parse_args()

    run = final_run(a.seq)
    txt = (run / "config_train.yaml").read_text().replace(REMOTE, str(SIMGS / "data"))
    for a_, b_ in ALIAS.items():
        if not (SIMGS / "data/neuman/dataset" / a_).exists(): txt = txt.replace(a_, b_)
    cfg = OmegaConf.merge(default_cfg, OmegaConf.create(txt))
    for br in ("human", "scene", "object"):
        cfg[br].ckpt = str(run / "ckpt" / f"{br}_final.pth")
    if a.image_scale: cfg.dataset.image_scale = a.image_scale
    out = Path(a.out) / a.seq
    cfg.eval = True
    cfg.train.skip_final_eval = True
    cfg.train.skip_final_animation = True
    cfg.logdir = str(out / "_trainer"); cfg.logdir_ckpt = str(out / "_trainer/ckpt")
    os.makedirs(cfg.logdir_ckpt, exist_ok=True)
    safe_state(seed=cfg.seed)
    tr = GaussianTrainer(cfg)

    ds = get_train_dataset(cfg)                         # train_on_all=True in the paper runs: every frame, in order
    print("run", run, "| frames", len(ds))
    rc = cfg.get("render", {})
    lay = dict(layer_composite=bool(rc.get("layer_composite_val", rc.get("layer_composite", False))),
               layer_base=str(rc.get("layer_base", "scene")),
               layer_overlay_human=bool(rc.get("layer_overlay_human", True)),
               layer_overlay_object=bool(rc.get("layer_overlay_object", True)),
               layer_human_clip_mask=str(rc.get("layer_human_clip_mask", "none")),
               layer_object_clip_mask=str(rc.get("layer_object_clip_mask", "none")))
    white = torch.ones(3, device="cuda"); black = torch.zeros(3, device="cuda")
    scene_out = tr.scene_gs.forward()
    for k in ("gt", "full", "human", "object", "scene"):
        (out / k).mkdir(parents=True, exist_ok=True)

    sel = [int(x) for x in a.frames.split(",") if x] or list(range(len(ds)))
    for i in sel:
        data = ds[i]
        fidx = int(torch.as_tensor(data.get("frame_idx", i)).item())
        if hasattr(tr.human_gs, "clear_cache"):              # HUGS_MMLP caches pose features; stale without this
            tr.human_gs.clear_cache()
        h = tr._forward_human_for_frame(data, fallback_idx=-1, is_train=False)
        o = tr.object_gs.forward(data)
        full = render_human_scene(data=data, human_gs_out=h, scene_gs_out=scene_out, object_gs_out=o, bg_color=black,
                                  human_bg_color=black, object_bg_color=black,   # exactly as GaussianTrainer.validate()
                                  render_mode="human_scene", render_human_separate=True, render_object_separate=True, render_scene_separate=True,
                                  render_human_alpha=True, render_object_alpha=True, **lay)["render"]
        hum = render_human_scene(data=data, human_gs_out=h, scene_gs_out=None, object_gs_out=None, bg_color=white, render_mode="human")["render"]
        sc = render_human_scene(data=data, human_gs_out=None, scene_gs_out=scene_out, object_gs_out=None, bg_color=white, render_mode="scene")["render"]
        ob = render_human_scene(data=data, human_gs_out=None, scene_gs_out=None, object_gs_out=o, bg_color=white, render_mode="human_scene")["render"]
        name = f"{fidx:05d}.jpg"
        for k, t in (("gt", data["rgb"]), ("full", full), ("human", hum), ("object", ob), ("scene", sc)):
            to_img(t).save(out / k / name, quality=93)
        if i % 40 == 0:
            print(i, fidx, tuple(full.shape))


if __name__ == "__main__":
    main()
