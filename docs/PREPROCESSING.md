# Preprocessing sources and unresolved runtime checks

Official public documents and model metadata were checked on 2026-10-27. No
weights were downloaded, models executed, credentials configured, or explicit
agreements accepted. This is a source/access audit, not training-environment
validation. Machine-readable observed revisions, file sizes and upstream LFS
SHA256s are in `configs/preprocessing-sources.json`. Those hashes are advertised
upstream identities, not local checksums of downloaded weights.

## License and access findings

| Component | Verified terms/source | Practical consequence |
| --- | --- | --- |
| DepthCrafter default | [Code license](https://github.com/Tencent/DepthCrafter/blob/e5186120a71ecb56b69f9e89654198b4eadfaf60/LICENSE), [weight license](https://huggingface.co/tencent/DepthCrafter/blob/f2207fa67e799d3cac41b0a051bb051401137a5b/LICENSE): inference code and weights are for academic/research/education; commercial or production use excluded | Do not call the complete stack MIT. Hackathon/public-product suitability is not established merely by public download access. |
| Stable Video Diffusion XT | [Model license](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/9e43909513c6714f1bc78bcb44d96e733cd242aa/LICENSE.md): Community agreement; commercial registration and revenue conditions; use/distribution terms | Separate from MIT code. DepthCrafter's NOTICE also references older SVD terms; this audit does not resolve that relationship or accept terms. |
| Metric3D | [Code BSD-2-Clause](https://github.com/YvanYin/Metric3D/blob/eb5b6fac0dc155e4e52f576e304fbf11655ff339/LICENSE); README expressly describes the code license. [Model repository](https://huggingface.co/JUGGHM/Metric3D/tree/80d2d1410afb4b23cd9d18c6be9144483d4b70b6) has no README/license at observed revision | Weight-specific grant remains unverified, not asserted prohibited or permissive. Authors invite commercial inquiries; do not infer blanket model rights from code. |
| UniDepth alternative | [Code license](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/LICENSE): CC BY-NC4.0. Model card links code but has no independent license statement | Not an established commercially permissive substitute; compatibility with current upstream v2 also remains untested. |
| SpaTracker V1 default | [Official license](https://github.com/henry123-boy/SpaTracker/blob/191a3ddf447b87ce61975c6ff0e10653e19f590d/LICENSE): CC BY-NC4.0; required [softsplat source](https://github.com/sniklaus/softmax-splatting) states academic use only | GPU kernel and use restrictions need separate consideration; SpaTracker V2 is not a drop-in validated replacement for the pinned V1 wrapper. |
| RAFT | [Official source](https://github.com/princeton-vl/RAFT/tree/2888e15a51fa41140771d3f498ed8023cff098d1) BSD-3-Clause; README links model archive and `raft-things.pth` | Separate explicit weight grant not located in reviewed README/LICENSE/script. Preserve uncertainty; do not assert a new restriction not stated by authors. |
| BootsTAPIR | [Official README](https://github.com/google-deepmind/tapnet#license), [model card](https://huggingface.co/google/tapnet): explicitly Apache2.0 for code and linked pretrained weights | Exact `bootstapir_checkpoint_v2.pt` is identified. MoSca's wrapper still executes vendored CoTracker visualization, whose separate license must not be erased. |
| CoTracker visualizer | [Official README](https://github.com/facebookresearch/co-tracker#license): majority CC-BY-NC with named exceptions; local visualizer header refers to source license | Exact historical visualizer exception has not been established. BootsTAPIR's Apache license does not automatically cover this wrapper dependency. |

HF metadata reports DepthCrafter, SVD XT, Metric3D, UniDepth v2 and TAPNet public
and ungated. That establishes metadata access only; actual binary/CDN access,
local checksum validation and execution are untested. No credential grant or
click-through was encountered. These findings are not legal clearance for public
commercial use; they also do not assert that every ambiguity forbids private
research. Source footage and generated-output redistribution remain separate.

## Default preprocessing is substantially heavier than checkpoint rendering

Pinned MoSca `profile/demo/demo_prep.yaml` selects DepthCrafter, SpaTracker and
RAFT with metric alignment. The wrappers load DepthCrafter UNet, SVD fp16 image
encoder/VAE, and Metric3D giant2: their identified weight payload alone is
**10,020,174,205 bytes (~9.33GiB)**, before tracker/flow weights, configs, caches,
environment, optimizer state and outputs. This is not a total disk or VRAM bound.

Current wrappers float downstream Git and model revisions. The observed current
revisions in the manifest are candidates for an explicit compatibility check,
not proof of matching the original MoSca training environment. Do not silently
let a production smoke download whatever `main` serves at execution time.

## Actual dependency and operation checks

- Existing A100 evidence covers native_add3, PyTorch3D, and camera/static/dynamic
  loading. It does not cover preprocessing or training operations.
- Default SpaTracker calls CuPy softsplat CUDA JIT kernels; pinned MoSca requires
  `cupy-cuda11x==12.2.0`. Its wrapper requires device index0. Importing CuPy alone
  would not validate kernel compilation or inference.
- RAFT's alternate custom correlation extension is optional and default-disabled
  in the inspected wrapper; do not add an unnecessary mandatory build claim.
- BootsTAPIR uses the PyTorch implementation plus einshape, mediapy, imageio,
  torchvision, absl and its visualizer; the unused JAX model file does not make
  JAX mandatory for that direct path.
- Camera/fit code adds OmegaConf, Kornia and OpenCV operations. Dynamic model
  voxel aggregation calls `torch_geometric.nn.pool.voxel_grid`; exercise its
  compiled backend explicitly, not just `import torch_geometric`.
- Upstream install.sh installs simple-knn and numerous broad dependencies. No
  active simple_knn import was found in the inspected MoSca Python pipeline;
  distinguish broad installer contents from exercised-path requirements.
- The default fit requests 2,000 camera bundle-adjustment steps (fallback in
  lite_moca_reconstruct.py),4,000 scaffold and8,000 photometric steps. Full
  training time/VRAM is unmeasured. Render-only memory must not be reused as a
  training estimate or as evidence of RTX5070Ti support.

## Smallest sensible custom-video compatibility experiment

The source explicitly supports direct `dep_mode='metric3d'` with
`tap_mode='bootstapir'`, avoiding the DepthCrafter/SVD/SpaTracker model stack.
Default Metric3D giant2 still needs5.51GB of weights. Its existing constructor
also supports `metric3d_depth_version='small'` (150.12MB checkpoint), but
`mosca_precompute.get_moca_processor` does not expose that constructor option
through YAML. Selecting it would require an explicit thin invocation wrapper
using the actual constructor, not an invented YAML key. No such switch has
been made or tested. Depth/tracking quality may differ from the default method.

Once the clip and model-use/source decisions are settled: prepare a new workspace,
verify exact dependencies and checksums, then run a bounded subset to measure
load/inference peak memory, time and disk. A 15-minute single-A100 profiling
invocation is a measurement gate, not a promise to finish reconstruction. Preserve
partial priors/logs and use measurements to size the full fit. No custom training
runner or complete preprocessing lock is claimed ready from this audit.
