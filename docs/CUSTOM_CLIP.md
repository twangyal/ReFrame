# Fresh custom clip: Black Cat walking

The selected input is a continuous **4-second excerpt (5–9 seconds)** from
[Black Cat walking](https://commons.wikimedia.org/wiki/File:Black_Cat_walking.webm)
by **Dzkouslavia**, Wikimedia Commons. File-page revision **1278831775** declares
own work and `{{self|cc-zero}}`; metadata links the [CC0 1.0 dedication](https://creativecommons.org/publicdomain/zero/1.0/).
CC0 permits reuse/adaptation without an attribution condition; credit is retained
for provenance. This is ordinary newly selected footage, not the Balloon2 source
or any existing reconstruction. No source reconstruction is supplied or reused.

The pinned source/author/license snapshot, URLs, download and excerpt hashes,
trim/resize operations and selected-frame hashes are recorded in
[`configs/custom-black-cat-source.json`](../configs/custom-black-cat-source.json).
The Commons 1080p derivative is 7,706,224 bytes; its SHA256 is
`872ed55486d9d325ff59e2b35f2cecdb937531a4512bacb12526ec65af3f7073`.
The original upload and the downloaded derivative are distinguished explicitly.
No video/image asset is published with this documentation.

## Preparation and actual inspection

Download the exact public derivative URL from the manifest to the recorded path
and verify its size/SHA256 before decoding. Preserve that downloaded file. Then:

```bash
ffmpeg -nostdin -v error -ss 5 -i data/custom/black-cat/commons-1080p.webm \
  -t 4 -map 0:v:0 -an -vf scale=960:540 -c:v libx264rgb -crf 0 \
  -preset fast -movflags +faststart data/custom/black-cat/black-cat-5s-to-9s.mp4
.venv/bin/python -m reframe.custom_scene \
  --video data/custom/black-cat/black-cat-5s-to-9s.mp4 \
  --out outputs/custom-black-cat-input --max-frames 48 --max-side 480
```

The prepared MP4 has 96 decoded frames, 24 fps and 4.000 seconds. 48 images at 480×270
were prepared for MoSca's actual `images/` loader. Source decode indices and PTS
are preserved; subsampling is not silently represented as constant spacing.
The excerpt's timeline starts at 0; the 5-second source offset is recorded separately.
No physical camera calibration or original acquisition wall time is yet known.

All 48 selected frames were visually inspected. One cat remains identifiable with
head/tail motion and slight body movement, a textured background and no cuts.
Limitations: dark fur has weak internal texture, the tail tip is cropped in early
frames, vegetation can move, and handheld parallax is modest. These are risks for
depth/tracking and geometry; preparation is not evidence that reconstruction works.

## Reconstruction outcome

The fresh clip now has real point tracks, an estimated camera/bundle solution,
155 motion nodes, trained Gaussian checkpoints and inspected dynamic playback.
See [measured results and limits](CUSTOM_RESULTS.md) and the
[stage workflow](CUSTOM_GPU.md). This is separate from the Balloon2 object-selection
experiment. No new hosted SAM request was made for the cat.

The actual prior path uses Apache-2.0 Depth Anything V2 Metric VKITTI Small and
BootsTAPIR, with recorded code/weight revisions and hashes. Metric3D was not used
because its weight-license coverage was unresolved. RAFT, DepthCrafter and Stable
Video Diffusion were not required by this path. The [license audit](PREPROCESSING.md)
and [CPU compatibility record](PREPROCESSING_CPU.md) explain that choice.

The successful photometric run used an A100 40GB and completed within the bounded
15-minute invocation. Existing checkpoints and all source data are preserved.
The experiment does not establish a general minimum VRAM requirement or local
RTX 5070 Ti compatibility.
