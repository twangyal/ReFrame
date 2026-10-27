# Reproducing the research prototype

## CPU environment

Run `bash scripts/setup_cpu.sh` and `.venv/bin/python -m unittest discover -s tests -v`.
The hash-locked requirements are for Python 3.12/LinuxCPU. One CUDA integration
test is explicitly skipped without GPU/upstream extensions. Parser fixtures are
independently generated synthetic rasters, not model outputs.

## Separate GPU environment

The validated existing-scene stack is A100 40GB, PyTorch 2.1.0/CUDA11.8, pinned MoSca
native_add3 rasterizer, NumPy 1.26.4, SciPy 1.11.4, Matplotlib 3.8.4, Pillow 10.4.0,
PyTorch3D 0.7.8, torch-geometric2.5.3, Open3D CPU0.18.0 and iopath0.1.10. System
packages: git, libgl1, libegl1, libglapi-mesa, libgomp1. Exact pinned image and
PyTorch3D package digest are in scripts/modal_smoke.py and modal_scene_worker.py.
Do not run upstream install.sh without inspection: it removes its named Conda
environment. Existing-scene loading does not validate full training dependencies.

The bounded Modal runner prints a plan by default and requires explicit execution
plus fresh account/resource/cost verification. It never creates credentials.
Authentication is supplied separately through the provider's supported mechanism.
No persistent Volumes or automatic workload retries are used. A generated approval
record is not a provider-enforced spend cap; verify actual billing controls.

## Published checkpoint workflow

Acquire the pinned MoSca source and legally usable sample described in
[PUBLISHED_SAMPLE.md](PUBLISHED_SAMPLE.md). Do not modify original checkpoints.

```bash
REFRAME_MOSCA_ROOT=/path/to/MoSca python -m unittest discover -s tests -p test_cuda_renderer.py -v
python -m reframe.cli probe-mosca --mosca /path/to/MoSca \
  --checkpoint /path/to/scene/log-directory --out outputs/baseline
.venv/bin/python -m reframe.cli prepare-scene-video \
  --scene-source /path/to/scene --fps 3 --out outputs/prepared
```

The chosen playback rate is not evidence of original capture timing. Inspect
source/nearby baselines before association. Hosted segmentation uses MODEL_API_KEY
from the environment; never put credentials in files committed to this repository.
`segment --allow-api` permits a billable call on cache miss, so use it only after
separate cost/authorization checks. Completed caches are reused without calls.

```bash
.venv/bin/python -m reframe.cli segment --video outputs/prepared/scene.mp4 \
  --frames outputs/prepared/frames --prompt 'green dinosaur balloon' \
  --object-id 0 --cache .cache/sam --out outputs/masks
python -m reframe.cli associate --mosca /path/to/MoSca \
  --checkpoint /path/to/scene/log-directory --scene-source /path/to/scene \
  --frames outputs/prepared/frames --masks outputs/masks/masks.npz \
  --baseline outputs/baseline/baseline.json --threshold .8 --min-frame-mass .1 \
  --out outputs/selection
```

The segmentation command above intentionally omits live opt-in and needs a
completed cache. Inspect raw IDs; a prompt may identify multiple objects.
The .1 support threshold was chosen after initial evaluation on Balloon2; its
refined results are exploratory, not an untouched benchmark.

## CPU cleanup and diagnostic replay

```bash
.venv/bin/python -m reframe.cli revise-selection \
  --selection outputs/modal-association-009/selection/selection.npz \
  --exclusions configs/balloon2-cleanup.json --out outputs/new-cleanup
.venv/bin/python -m reframe.demo --config configs/balloon2-demo.json \
  --out outputs/new-diagnostic-bundle
```

The sample recipe pins original private artifacts by hash. A fresh clone has no
such footage/checkpoints/outputs; prepare them separately with appropriate rights.
If intentionally changing inputs, review and repin provenance. Never replace a
selection while retaining its old render report. The two-row exclusion policy is
an exploratory sample-specific ablation; it is not a rule to discard static objects.

`index.html` in the local bundle plays source, SAM and held-render comparisons.
It needs no server/network. It does not establish new GPU results or render the
cleanup candidate. Full and compact selection sidecars retain the object identity
and a separate membership/policy revision digest.

The pending GPU diagnostic uses `render-selection` with the same scene, frames,
masks, and `--selection /path/to/render-selection.npz`; optional `--all-frames`
renders every discrete reconstruction index. It reports absolute visible mass,
IoU and contamination, preserves occluders for evaluation, and requires separate
visual inspection. No SAM request or training association is repeated.
