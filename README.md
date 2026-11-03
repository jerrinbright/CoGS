# CoGS: Compositional Dynamic Human-Object Scenes Gaussian Splatting from Monocular Video

[Jerrin Bright](https://jerrinbright.com) and [John Zelek](https://uwaterloo.ca/systems-design-engineering/profile/jzelek)  
Vision and Image Processing Lab, University of Waterloo, Canada

[arXiv](https://arxiv.org/abs/2606.28820) · [Project page](https://jerrinbright.github.io/CoGS/)

## Abstract

Monocular human-object reconstruction is underconstrained because articulated body motion, rigid object motion, and the static scene share pixels but obey different dynamics. Without component-aware priors, moving objects can be absorbed by the human or background, persistently hidden body regions remain unconstrained, and low-texture scene surfaces receive little geometric support. We present CoGS, a compositional Gaussian-splatting framework for human-object scene reconstruction. CoGS decomposes the video into three branches: an articulated human field, a rigid manipulated-object field, and a static scene field, each optimized with supervision matched to its dynamics. To address missing human evidence, CoGS builds the canonical human from generative completion views that synthesize body regions the input video never observes, then uses visibility-aware correction so observed Gaussians adapt to real pixels while hidden regions remain anchored to the completed prior. For the static scene, CoGS introduces delayed scene-planar regularization over compatible planar primitives, providing low-frequency structure. A compact staged optimization stabilizes the branches before final compositional refinement. On HOSNeRF and NeuMan, CoGS achieves state-of-the-art full-frame rendering, while human-focused ablations and diagnostics expose the role and limits of each branch.

Code coming soon.

## Citation

```bibtex
@article{bright2026cogs,
  title   = {CoGS: Compositional Dynamic Human-Object Scenes Gaussian Splatting from Monocular Video},
  author  = {Bright, Jerrin and Zelek, John},
  journal = {arXiv preprint arXiv:2606.28820},
  year    = {2026}
}
```
