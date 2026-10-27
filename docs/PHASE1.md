# Phase 1 status

Phase 1 is incomplete. Published-scene reconstruction access, official hosted SAM
tracking, exact-frame alignment and persistent Gaussian association are
demonstrated. Milestone 06 now has qualified sample evidence from the cleanup validation;
milestone 03 custom 3–5-second video reconstruction is not yet demonstrated. A working sample
fallback is not a claim of completing that custom-video milestone.

The raw sample selection has 8,227 dynamic + 2 static rows. A separately saved manual
cleanup revision excludes only the two flagged static outliers and now has its own
hash-bound render measurements. Source/nearby isolation checks cover all 12 discrete times;
wide-view geometry, full occlusion/re-entry and continuous-time behavior remain
unvalidated. No Phase 2 editor or transformation is claimed.

## Exit criteria and evidence

| Roadmap milestone and exit criterion | Evidence and status |
| --- | --- |
| 01 — Environment & GPU setup (CORE): MoSca dependencies install; GPU inference succeeds locally or remotely | **Demonstrated for reconstruction playback:** pinned environment/package inventories, native CUDA reference test and A100 render. Both rasterizers, full training imports and real BootsTAPIR point inference now pass; camera/scaffold/photo execution remains unverified. |
| 02 — Known-good MoSca playback (CORE): provided dynamic reconstruction renders at multiple timestamps and viewpoints | **Demonstrated:** preserved Balloon2 checkpoint, camera parameters, renderer and inspected source/nearby views at frames 0/6/11. |
| 03 — Custom short-video reconstruction (TARGET): a continuous 3–5-second input reconstructs with usable dynamic motion | **Outstanding:** a CC0 four-second clip is selected and 48 images prepared (CUSTOM_CLIP.md); 48 CPU depth priors pass the actual MoSca loader; real BootsTAPIR tracking retains 3,382 of 4,096 points after MoSca filtering; camera solving, scaffold fitting and Gaussian playback remain unverified; no custom checkpoint or playback. CPU input preparation is implemented. |
| 04 — Meta SAM 3.1 segmentation (CORE): chosen object masked/tracked across the same source frames | **Demonstrated on sample:** one hosted request, cached masks, returned object ID 0 through all 12 exact source frames and inspected overlay. No custom-clip inference. |
| 05 — Frame, time & camera alignment (CORE): agreement on order, timestamps, resolution and projection | **Demonstrated within sample index domain:** exact source pixels/order, mask dimensions, calibrated MoSca projection and reprojection comparisons. Synthetic MP4 timing is recorded; original acquisition cadence is unknown. No continuous-time interpolation claim. |
| 06 — 2D mask to 4D Gaussian association (CORE): membership isolates a coherent object across time | **Qualified sample exit criterion demonstrated:** checkpoint-bound 8,227-dynamic selection rendered and visually inspected at all 12 discrete timestamps in source/nearby views. Cleanup changes one false-positive pixel at frame11; ragged/missing geometry remains. See RESULTS.md and exact cleanup evidence; no generalization or geometric-correctness claim. |

**Gate A (Oct 30):** the sample demonstrates that SAM selection can map to actual
4D scene elements rather than only pixels. Evidence is limited to this scene and
sampled nearby views. If coherence fails, investigate projection, visibility and
occlusion before editor work. Passing this limited sample test does not complete
milestone 03 or all of Phase 1.


## Custom clip prerequisites

A CC0 four-second online clip is now prepared; see [source and inspection](CUSTOM_CLIP.md).
For additional input, choose a rights-cleared 3–5-second MP4 with one clearly identifiable object,
reasonable lighting, textured background, no cuts and modest motion. Prepare it:

```bash
.venv/bin/python -m reframe.custom_scene --video /path/to/clip.mp4 \
  --out outputs/custom-input --max-frames 48 --max-side 480
```

This CPU command preserves the source and records original PTS, sampled indices,
resize and hashes. It creates the actual MoSca `images/` input contract. It is
not a reconstruction, and the synthetic-video tests are not custom-scene results.

From the pinned MoSca checkout, the actual upstream entrypoints are:

```bash
python mosca_precompute.py --cfg profile/demo/demo_prep.yaml --ws /path/to/new-scene
python mosca_reconstruct.py --cfg profile/demo/demo_fit.yaml --ws /path/to/new-scene
```

Default preprocessing needs RAFT and SpaTracker weights, DepthCrafter plus Stable
Video Diffusion, and Metric3D alignment. These prior models and their full training
dependency chain are not validated by existing-scene rendering. Inspect each
license/access requirement, pin and hash weights, and prevent implicit downloads
or gated-terms acceptance during a smoke test. No prior weights are bundled here.
The official UniDepth example remains untested; this experiment instead uses externally cached Depth Anything V2 Small priors and the official BootsTAPIR wrapper, now validated on the actual 48 frames.

Default fitting requests 4,000 scaffold and 8,000 photometric steps plus preprocessing
and the 2,000-step camera bundle-adjustment fallback. First profile a bounded preprocessing subset on the chosen
clip, measuring load/inference GPU memory, system RAM, disk and elapsed time;
then use measurements to bound training. No minimum VRAM or full-training runtime
is established. The render-phase 1.184 GiB peak is not a training requirement.
The tested A100/CUDA11.8 environment is not validated for RTX5070Ti/Blackwell.

Completing Phase 1 requires an actual custom reconstruction with preserved inputs,
configs, checkpoints and renders, plus the association exit checks. Paid execution
and any further hosted inference require explicit cost/access checks. No new
model call or cloud job runs merely by invoking the CPU tests or preparation.

The [preprocessing audit](PREPROCESSING.md) records official weight sources,
observed revisions, model sizes and license uncertainties. No prior weights were
downloaded or executed by that audit. Cleanup diagnostics now measure each
excluded row’s absolute visible mass; the exact revision now has GPU measurements in the cleanup evidence.

A [CPU preprocessing follow-up](PREPROCESSING_CPU.md) now verifies real depth
inference and MoSca loading for all 48 custom frames, using an explicitly
Apache-licensed alternative depth prior. This is not custom reconstruction.

Real GPU point tracking completed in 291.12 seconds including setup; see
[evidence](evidence/custom-tracking-2026-10-27.json). Five timestamps were
visually inspected. This is point tracking, not custom SAM segmentation or a 3D
reconstruction. The [bounded stage workflow](CUSTOM_GPU.md) preserves exact
input and checkpoint hashes between camera, scaffold and photometric stages.

The custom camera stage also completed: 500 optimization steps, finite saved
camera/bundle tensors, and preserved priors. Initial FOV estimation required the
upstream fallback, so calibration remains uncertain. Scaffold fitting found 243
dynamic tracks but its first attempt stopped on a diagnostic image encoding
error; a tested compatibility fix is being validated. Milestone 03 still needs
trained Gaussians and inspected reconstructed playback.
