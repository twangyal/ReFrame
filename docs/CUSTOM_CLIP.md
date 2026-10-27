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

## Remaining reconstruction gates

The CPU follow-up now provides 48 depth priors; see [compatibility evidence](PREPROCESSING_CPU.md).
Point tracks and an estimated camera/bundle solution now exist. No trained Gaussian checkpoint or custom playback
exist yet. No new SAM
request was made. Existing sample masks must never be applied to this different
clip. The hosted SAM one-call allowance remains consumed.

Remote CUDA execution and a verified preprocessing environment remain required.
No custom reconstruction or new hosted inference has run.

See [preprocessing audit](PREPROCESSING.md) for actual model APIs and license
coverage. The smallest source-supported candidate is Metric3D-small plus
BootsTAPIR. Metric3D's explicit weight-license coverage is still unverified (the repository
owner asks commercial users to contact them; see the preprocessing audit);
BootsTAPIR weights are Apache 2.0 but its vendored visualization has separate terms.
No Metric3D weights were downloaded. The official Apache-licensed Depth Anything
V2 Metric Small alternative was downloaded, hash-verified and tested on CPU.

Additional source inspection found that flow can be skipped by the existing
non-RAFT constructor branch together with `compute_flow=False`; preprocessing
then estimates dynamic evidence from point tracks. This avoids RAFT weights but
changes the prior configuration and is not runtime-validated. The real dynamic
resampling parameter is `n_track_pdynamic` (the demo YAML's `n_track_dynamic`
is not read at that call). A thin constructor invocation is needed for the
Metric3D-small option, since the stock config factory does not forward it.

Before training, use a separately bounded compatibility/profile run: native CUDA
reference and voxel-grid backend, pinned prior imports/weights, 4 representative
frames for depth, then bounded point tracking. Record memory/time and preserve
priors/logs. This is not yet a launch-ready dependency lock or a promise that full
camera/scaffold/photometric fitting finishes within 15 minutes. The chosen source
clip resolves the input blocker; access and preprocessing verification remain.
