# CPU preprocessing compatibility evidence

These are actual CPU checks, separate from the earlier A100 renderer validation.
They do not establish CUDA training compatibility or reconstruct the custom clip.
The working reconstruction environment and all original inputs remain unchanged.

## What passed

Using Python 3.12.14, PyTorch 2.10.0+cpu and the separate pinned overlay:

- torchvision, OpenCV, Kornia, OmegaConf, MoSca's BootsTAPIR wrapper and
  `MoCaPrep`, and the pinned Metric3D `hubconf` import successfully.
- Metric3D-small (37,495,051 parameters) and BootsTAPIR (54,699,335 parameters)
  construct with random weights. No pretrained Metric3D weights were loaded.
- MoSca's actual depth-boundary function passes a constant/step-depth check with
  its default 5×5 Laplacian. The constant plane retains all 1,024 pixels; the
  step rejects 128 boundary pixels. This is a synthetic operation check.
- Official Depth Anything V2 Metric VKITTI Small pretrained inference ran on
  four real cat-clip frames on CPU. Its finite, positive depth outputs loaded
  unchanged through MoSca's actual `Saved2D.load_dep()` and preserved frame order.
  The four-frame run took 11.989 seconds including model loading and consumed
  peak process RSS 817,232 KiB. This is not GPU memory or full-training runtime.

The four depth maps were visually inspected: the cat forms a coherent depth
region, while grass produces noisy local depth variation. The configured
boundary filter retains 48.29% of pixels overall, often discarding vegetation
and silhouettes. Metric scale and temporal depth consistency are unverified.
The outdoor model was fine-tuned on Virtual KITTI, so transfer to a close cat
scene is an experiment, not established accuracy. Full reconstruction must
validate whether these priors are usable.

The full 48-frame follow-up completed in **128.139 seconds**,
with peak process RSS **890,996 KiB**. All 48 depth maps are
finite, positive and correctly loaded by MoSca; the four original probe arrays
match the full run exactly. Overall valid-depth fraction is
**51.42%** with the configured boundary filter.
Only the four representative depth maps were visually inspected; this is not a
48-view geometric-quality claim. [Numerical evidence](evidence/custom-depth-cpu-2026-10-27.json)
and [weight-free checks](evidence/preprocessing-cpu-imports-2026-10-27.json) preserve
source/model/input hashes, package versions and per-frame timings.

## License finding and explicit alternative

Metric3D's pinned code is BSD-2, but its distributed weight-specific grant remains
unverified. The owner's commercial-use reply is recorded in PREPROCESSING.md;
no author contact or new agreement occurred. Random-weight construction does not
resolve weight licensing or demonstrate pretrained inference.

The **alternative actually tested** is the official
[Depth Anything V2 Metric VKITTI Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/tree/c725b8589bdf6ab04072cab74c0467830db80d6d).
Its model card explicitly states Apache-2.0, and the upstream code is Apache-2.0.
The 99,221,808-byte checkpoint's SHA256 was verified before loading. This model
writes external depth files via an existing MoSca input contract; MoSca and SAM
are not rewritten. The base/large Depth Anything variants have different terms
and are not substituted. No claim is made that the entire MoSca dependency stack
is Apache: its rasterizer and vendored visualization retain their separate terms.

## Reproduce the local checks

First install the existing `requirements-cpu.lock` environment according to [setup](SETUP.md). This optional overlay shares that environment's CPU Torch package;
it does not modify `.venv`. The overlay inventory is not a CUDA lockfile.

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv --python .venv/bin/python .cache/preprocessing-cpu
.venv/bin/python - <<'PY'
import sysconfig
from pathlib import Path
base = sysconfig.get_paths()['purelib']
overlay = Path('.cache/preprocessing-cpu/lib/python3.12/site-packages')
(overlay/'reframe_existing_cpu.pth').write_text(base + '\n')
PY
uv pip install --python .cache/preprocessing-cpu/bin/python --no-deps \
  setuptools==69.5.1 wheel==0.43.0
uv pip install --python .cache/preprocessing-cpu/bin/python --no-deps \
  --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements-preprocessing-cpu.txt
```

`--no-deps` intentionally keeps the verified CPU base rather than resolving a
new CUDA Torch distribution. The complete overlay pins are supplied. Generic
package metadata checks may not recognize packages inherited through the `.pth`
and flag `opencv-python` because this headless environment provides
`opencv-python-headless`. The checks above exercise the required operations;
this is not a claim that every training dependency is installed.

Use clean checkouts at the revisions in `configs/preprocessing-sources.json`:

```bash
.cache/preprocessing-cpu/bin/python scripts/probe_preprocessing_cpu.py \
  --mosca /path/to/pinned/MoSca --metric3d /path/to/pinned/Metric3D \
  --out outputs/preprocessing-cpu-check --construct
```

The probe disables Python socket connections and common pretrained-weight
loaders. Reports distinguish import/constructor success from inference. It
refuses wrong source revisions, existing outputs and writes into source trees.

For actual CPU depth, download the pinned official weight identified in the
source manifest and verify its size/SHA256 before invoking:

```bash
.cache/preprocessing-cpu/bin/python scripts/probe_depth_prior_cpu.py \
  --source /path/to/pinned/Depth-Anything-V2 \
  --weights /path/to/depth_anything_v2_metric_vkitti_vits.pth \
  --mosca /path/to/pinned/MoSca --prepared outputs/custom-black-cat-input \
  --out outputs/custom-black-cat-depth-check
```

Default indices are 0, 16, 32 and 47 for this 48-frame clip; `--all-frames`
requests all prepared frames. The script verifies source/weight/image hashes,
refuses output inside original data/code/weight directories, disables networking
while inference runs, and saves predictions under `external_depth/*.npz` with
key `dep`. It then uses unmodified MoSca `Saved2D` to check frame order, shape
and exact depth values. This script is CPU-only; it refuses a CUDA/MPS host
because the upstream image helper chooses its device automatically.

## Compatibility corrections and limits

- `mmengine` alone is insufficient: Metric3D `mono/utils/comm.py` still imports
  `mmcv.utils.collect_env`. Pure-Python MMCV 1.7.2 imports successfully here.
- MMCV's source build needs `pkg_resources`; setuptools 69.5.1 and a nonisolated
  build resolved the missing build dependency. No CUDA extension was compiled.
- The probe's initial nondefault 3×3 float32→float64 OpenCV Laplacian failed;
  MoSca's actual default 5×5 path passes. No upstream code was patched.
- Imports required IPython and scikit-image even though the task is headless.
- xFormers is absent; the weight-free constructors and tested depth inference
  worked without it. This does not establish performance on CUDA.
- No real BootsTAPIR tracking, epipolar dynamic classification, camera fitting,
  scaffold optimization, Gaussian fitting or custom reconstruction has run.


## Next compatibility gate

The existing external-prior branch supports constructing
`MoCaPrep(dep_mode="external", tap_mode="bootstapir", flow_mode=None,
align_metric_flag=False)` with these `external_depth` files. Its absent-depth-model
branch skips new depth inference, and `load_dep_list` reads the supplied NPZs.
With `compute_flow=False`, the existing preprocessing routine uses point tracks
for epipolar dynamic evidence. This is a source-verified configuration, not yet
an executed tracking or fit path.

The smallest next GPU check is one bounded A100 invocation: validate the existing
CUDA reference and compiled voxel-grid backend, load pinned BootsTAPIR weights,
track a small explicit query budget across the 48 preserved source frames, and
load the resulting tracks/depth into MoSca. Save tracks, visibility, timings,
peak memory and failure logs; do not start Gaussian training within this
compatibility check. A full tracking query budget and training duration should
follow measured results. This path does not load Metric3D, RAFT, DepthCrafter
or Stable Video Diffusion weights. It does not remove the separate MoSca
rasterizer and visualization license obligations.
