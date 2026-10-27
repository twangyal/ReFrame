# ReFrame — Experiment 01

Research prototype for **official hosted Meta SAM 3.1 masks → persistent MoSca
Gaussian object selection**.

On published Balloon2, a saved selection identifies 8,227 dynamic Gaussians across
all 12 inspected source/nearby-view timestamps. A manual cleanup removes two
static outliers from the earlier exploratory selection. Source-mask IoU at three
checkpoints is approximately 0.880 / 0.895 / 0.868; these are exploratory results,
not geometric ground truth. [Selection results and limits](docs/RESULTS.md).

A fresh four-second CC0 cat clip has also been reconstructed from new depth and
point-track priors, with 222,409 static and 37,135 dynamic Gaussians. All 48
source/fixed/nearby-camera timestamps were inspected. Head/tail motion remains
recognizable; detail, boundaries and unseen geometry have limitations.
[Custom reconstruction evidence](docs/CUSTOM_RESULTS.md).

These are distinct experiments: no hosted SAM request or semantic Gaussian
selection was run on the cat. Phase 1 milestones 01–06 are complete with these
qualified, separate results;
no Phase 2 editor or transformation is claimed. [Phase 1 audit](docs/PHASE1.md).

## Setup and checks

```bash
bash scripts/setup_cpu.sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m reframe.cli --help
```

Requires Python 3.12, uv, FFmpeg and ffprobe on Linux x86-64. The CPU lock includes
CPU PyTorch: do not install it into the separate MoSca CUDA environment.
No model inference or cloud run is triggered by these checks.

- [Setup and reproducible commands](docs/SETUP.md)
- [Actual upstream interfaces and algorithm](docs/FEASIBILITY.md)
- [Hosted SAM API and caching](docs/SAM_API.md)
- [Published sample provenance and licenses](docs/PUBLISHED_SAMPLE.md)
- [Phase 1 completion evidence and limitations](docs/PHASE1.md)

## Prototype

Modules separate segmentation, reconstruction access, contribution aggregation,
selection revisions, visualization and evaluation. Association uses the official
renderer's visible alpha/transmittance weights, preserving extent, depth order,
opacity and occluders. It does not equate 2D masks with Gaussian IDs or train a new
feature field. Scores and checkpoint-bound membership live in separate sidecars.

New local tools revise saved evidence without repeating inference, explicitly
record manual exclusions, validate render/report provenance, and prepare an
offline diagnostic bundle. The saved-selection render path is prepared and
CPU-tested; its new GPU execution remains pending. See Phase 1 status for limits.

## Provenance and rights

MoSca and SAM are upstream implementations; Feature4X was inspected as a reference
only. Pinned revisions are in upstream-revisions.json. ReFrame's integration code,
tests and documentation are newly authored; upstream reconstruction/segmentation
models and algorithms are not presented as ReFrame inventions. Hackathon event
window eligibility of code remains a separate submission check.

MoSca's MIT core excludes third-party components; its bundled rasterizer has
research/noncommercial restrictions. SAM uses its custom license and hosted
service terms. Feature4X has no top-level license at the inspected revision and
no code is copied here. Model/container/dataset licenses apply separately.

No credentials, upstream code, model weights, benchmark footage, decoded masks or
rendered videos are bundled. Numerical evidence includes actual runs and clearly
labeled synthetic tests. Public code does not grant rights to upstream assets.
