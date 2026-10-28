# Phase 2: persistent Gaussian object edits

Status: the single approved GPU attempt completed and exported real edited
video. **Gate B is demonstrated through the roadmap's highlight/recolor
fallback. A clean translation result is not accepted; full Phase 2 translation
quality remains unresolved.** This uses the published Balloon2 scene. The
custom cat reconstruction has not been segmented or selected. Phase 3 has not
started.

## October 28 validation result

The worker completed in **226.20 seconds**, preserving original checkpoint
hashes and returning all **120 native PNGs**. Its independent CUDA compositing
test passed. The A100 40GB/80GB hardware guard passed; this worker did not return
the exact allocated VRAM variant. There was one attempt, no retry, no new model
call and no persistent Volume. Implementation commit:
`13e6964fa26b7cf7658ce2901496d3dd91f3c787`.

All twelve source/nearby states were visually inspected. The 72 baseline,
highlight and isolation frames are pixel-identical to the earlier verified
cleanup renders. The balloon remains recognizable as its pose changes; holes,
rough boundaries and a few floaters remain, especially at the last frame.
Recoloring stays on the object without broad wall/person spill, but replaces
shading with a flat constant color.

| Frame | Source IoU | RGB change outside SAM | Absolute RGB change outside SAM |
|---|---:|---:|---:|
| 0 | 0.8801 | 6.91% | 115.39 |
| 6 | 0.8949 | 1.68% | 29.89 |
| 11 | 0.8681 | 7.45% | 150.43 |

Absolute RGB change sums mean absolute channel changes per pixel in renderer
RGB space. Across all twelve frames, source IoU ranges 0.8681–0.9567 and the
outside-SAM fraction of RGB change ranges 0.67%–7.45%. These are exploratory
metrics and do not establish novel-view geometry or zero contamination.

The translation vector was `[0.16714029, -0.00031522, -0.00190873]` in
reconstruction units. The maximum per-row translation residual across all
states was `4.85e-8`; selected rows retain their original time-dependent motion
and unselected positions remain unchanged within this numerical check.
The moved balloon follows the pose sequence in both cameras. However, its old
location exposes a large mottled green remnant/disocclusion artifact. This is
conspicuous in native PNGs and encoded video. Unselected object remnants and
poorly reconstructed background may both contribute; their relative roles
were not isolated by this experiment. This is a successful transform/export
mechanism, **not a clean object-move result**. No inpainting, contact or shadow
correction was attempted.

Five H264 MP4s passed full decoding: baseline, recolored, translated, edit
comparison and highlight/isolation comparison. Each contains twelve states at
chosen 3 fps for four seconds, with no interpolation. All twelve encoded edit
comparison frames were numerically compared to their composite PNG inputs;
beginning/middle/end decoded frames were visually inspected. Real-time playback
was not used as a separate visual test. The 480×270 native views and 1440×640
comparisons remain private in the task; no benchmark footage is published.

The recolor/highlight videos satisfy the fallback's independently viewable
edited-output requirement. Milestones 07–09 have sample-scene evidence; 10 has
translation/motion/export evidence but fails clean-move visual acceptance.
Further scope expansion is stopped. The smallest next diagnostic, if requested,
is to inspect remaining unselected contributions at the original object
location before attempting cleanup or inpainting. No further paid attempt is
authorized by this report.

Latest recorded billing at 04:58:08 UTC: **$4.50088206 cumulative usage, $0 billed**,
zero active tasks/GPUs and $0 effective net-spend limit. Snapshot increase from
preflight: $0.15003275, subject to reconciliation and not a final isolated job
invoice. The original verified $30 grant minus reported credits used gives an
estimated $25.50 remaining; the SDK does not expose the grant balance directly.
See the [hash-bound evidence](evidence/phase-two-balloon2-2026-10-28.json).
CPU verification: 96 passed, one CUDA-only test skipped locally; the actual
remote CUDA test passed separately.

## Current evidence

The frozen object `166dfad9-bd7b-5a98-ae91-b0ed44de8807` contains 8,227 dynamic
and zero static Gaussian rows. Compact selection SHA-256:
`c757ac57b6fd1d73f1bc5f26e4644aec1e08bd1916ffc98551bc99a84c818c19`.
It is bound to the original camera/static/dynamic checkpoints and upstream
revision. The full evidence sidecar retains soft scores and uncertain rows.
Selected foreground evidence fraction: minimum 0.80052, median 1.0; mean
across-frame variance 0.001975; minimum two visible training observations.
These scores are not calibrated probabilities; floating-point accumulation
can put a score slightly above one (observed maximum 1.000000073).

Existing cleanup render evidence, not a new Phase 2 experiment:

| Source frame | IoU | Selected contribution outside SAM | Absolute outside contribution |
|---|---:|---:|---:|
| 0 | 0.8801 | 6.19% | 241.81 |
| 6 | 0.8949 | 1.78% | 76.52 |
| 11 | 0.8681 | 7.97% | 382.61 |

Contributions are alpha-weighted pixel mass, not Gaussian counts. These are
exploratory refinement results: thresholds and manual outlier exclusions used
visual/evaluation feedback. They are not an independent held-out result.
All twelve discrete source/nearby views previously showed a coherent balloon
with residual holes/floaters. Full occlusion/re-entry, wide novel views and
continuous interpolation are untested. See [RESULTS.md](RESULTS.md) for the
original evaluation and limitations.

## Implementation

- `reframe.editing`: immutable parameter edits and selection-bound recipe.
- `reframe.edit_render`: all discrete times, source/nearby cameras, baseline,
  highlight, isolated object, recolor and translation.
- `reframe.edit_export`: verifies every native frame and selection hash,
  produces independently playable H264 videos, and decodes each export.

MoSca returns `(world_xyz, world_frame, scale, opacity, SH_features)` with
static rows followed by dynamic rows. `FrozenMoSca.gaussians(t)` evaluates the
original deformation at each saved timestamp. A fixed world-space vector is
then added only to selected positions. The same rows retain their deformation,
orientation, scale and opacity; unselected rows and saved checkpoints stay
unchanged. The default experiment moves along the first camera's right axis
by 30% of the selected object's robust first-frame extent (5th–95th percentile
box diagonal). The resolved vector is recorded in reconstruction units,
**not meters**, and is fixed across all timestamps.

Recolor sets selected SH DC coefficients to `(RGB - 0.5) / C0` and zeroes
higher bands. This gives a constant orange-red color `[0.95, 0.2, 0.08]` in
renderer RGB space and removes directional shading. It is a simple color edit,
not material relighting. The translation video preserves original appearance.

The five modes produce 120 lossless PNGs for this 12-frame scene. The report
records native frame hashes, selection/recipe binding, translation residual,
source IoU and visible contribution leakage. Recolor's absolute pixel change
outside SAM is also reported, with a fraction of total change; that is not a
geometric correctness metric. All original occluders remain in visible-mask
renders. Translation has no corresponding SAM ground-truth mask.

## Reproduce locally with a compatible CUDA runtime

Prepare a separate recipe JSON (do not overwrite the selection):

```json
{
  "schema": 1,
  "object_id": "166dfad9-bd7b-5a98-ae91-b0ed44de8807",
  "selection_sha256": "c757ac57b6fd1d73f1bc5f26e4644aec1e08bd1916ffc98551bc99a84c818c19",
  "color_rgb": [0.95, 0.2, 0.08],
  "translation_extent_fraction": 0.3
}
```

Use the setup and original asset retrieval instructions in
[README](../README.md), [PUBLISHED_SAMPLE.md](PUBLISHED_SAMPLE.md) and
[MODAL_SMOKE.md](MODAL_SMOKE.md). Check dependency and model license restrictions
before using or redistributing results. Benchmark assets are not in Git.

```bash
python -m reframe.cli render-edits \
  --mosca /path/to/pinned/MoSca --checkpoint /path/to/original/checkpoints \
  --scene-source /path/to/nvidia/Balloon2 \
  --frames outputs/balloon2-prepared/frames --masks outputs/balloon2-sam/masks.npz \
  --selection outputs/balloon2-cleanup-verified/render-selection.npz \
  --recipe outputs/phase-two-preparation/edit-recipe.json --out outputs/phase-two-render
python -m reframe.edit_export --renders outputs/phase-two-render \
  --out outputs/phase-two-videos --fps 3
```

The export plays twelve discrete states at a **chosen** 3 fps (four seconds).
It does not establish the original capture cadence or interpolate motion.
Outputs include baseline, recolor and translation source-view MP4s, an edit
comparison across both views, and a highlight/isolation comparison. Native
frames, report and video hashes remain available for inspection. The exporter
labels its result pending visual review; a playable synthetic fixture proves
codec/export handling, not real object-edit quality.

## Single GPU validation protocol (executed once; authorization consumed)

Use the existing bounded Modal runner with `--real-scene --render-selection
<compact-sidecar> --edit-recipe <recipe>` plus the same cached frames, masks and
baseline. No new SAM call, training or checkpoint alteration. Source staging
validates the recipe against the selection before any upload, using the project
CPU `.venv` prepared by `scripts/setup_cpu.sh`; the separate Modal SDK interpreter
does not need NumPy/Pillow installed. The executed operator's
approval record included `phase_two_edits_authorized: true`, exact
`edit_recipe_sha256` and `selection_sha256`, `authorized_gpus` containing
`A100-40GB` and `A100-80GB`, and all existing verified account/security fields.
A canonical approval can be claimed once, including uncertain failures. This attempt
has consumed its approval; these instructions do not authorize another run.

- One A100 40GB preferred; 80GB fallback. No retries or persistent Volumes.
- 900-second function cap; 850-second subprocess cap; 800-second internal
  worker deadline. The final rendering stage is capped at 240 seconds.
- 120-second startup allowance, two CPU cores, 16 GiB requested RAM.
- Conservative memory billing allowance: 41.6 GiB, including the possible
  default ephemeral disk uplift. Rate ceilings: GPU $4/hour, CPU $0.10/core-hour,
  RAM $0.02/GiB-hour. A further $2 allowance covers image build/transfer overhead.
- Whole-run planning estimate: **$3.4285**, rounded up to **$3.43**. This is an
  estimate, not a provider billing cap. Prior comparable render run: 214.3 seconds;
  budget for 15 minutes because setup/network/render timings can vary.
- Latest known cumulative usage $4.37084931, $0 billed, subject to reconciliation.
  Adding $0.50 reconciliation reserve and $3.43 gives $8.30084931 within the
  existing $30 credit-only budget. Fresh credit/control/rate checks are required.
- Stop on timeout, hardware mismatch, nonfinite outputs, hash mismatch or control
  failure. Confirm app stopped and zero active GPUs/tasks; preserve all logs and
  report cumulative billing without summing overlapping snapshots.

The existing credential was recorded to expire October 28 at 05:07:47 UTC.
It was used without renewal. Do not renew it or launch another run without the
required authorization. No paid
model calls, OPAL, subscriptions or Phase 3 work are included.

## Acceptance and honest stopping point

Inspect all twelve source/nearby pairs and beginning/middle/end enlargements.
Verify the same object remains coherent, background color spill is limited,
translation is visible while animation continues, and exported videos play
independently. Check unselected row invariance and report masks/leakage, missing
geometry, disocclusion holes, unchanged shadows and any temporal artifacts.
Metrics alone cannot pass novel-view geometry. Preserve baseline and prior
selection outputs. The executed run met the playable-output gate via the highlight/recolor fallback.
Its translation mechanics worked, but the clean-move quality criterion failed.
Full Phase 2 translation success is not claimed.
