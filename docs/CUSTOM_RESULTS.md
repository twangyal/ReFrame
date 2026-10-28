# Fresh custom-video reconstruction results

A fresh four-second CC0 clip has been reconstructed with MoSca. The cat remains
recognizable across all 48 inspected timestamps, and its head turn and tail
movement persist when the camera is held fixed. This demonstrates recognizable dynamic
playback for this limited feasibility case, with substantial visible blur. It does not establish accurate 3D
geometry or robustness on arbitrary footage.

The input is Dzkouslavia's **Black Cat walking**, a continuous 5–9 second excerpt;
[source, license and preparation](CUSTOM_CLIP.md) preserve the original download,
frame timestamps, resize operations and hashes. No precomputed reconstruction
was used for this clip.

| Stage | Actual evidence |
| --- | --- |
| Depth | 48 Depth Anything V2 Metric VKITTI Small priors; Apache-2.0 weights/code; CPU inference and real MoSca loader verified |
| Tracking | Official BootsTAPIR wrapper; 4,096 raw tracks, 3,382 retained by MoSca |
| Camera | 500 bundle-adjustment steps; finite camera/bundle checkpoints; FOV initialization used the upstream fallback |
| Motion scaffold | 1,000 steps; 155 nodes across 48 times; 243 dynamic tracks before filtering/resampling |
| Gaussian appearance | 2,000 steps; 222,409 static and 37,135 dynamic Gaussians; finite checkpoints |
| Playback inspection | All 48 source/fixed/nearby-camera timestamps inspected; enlarged comparisons at 0, 24 and 47 |

The camera, scaffold and photo stage runtimes were 27.91, 75.96 and 326.78 seconds.
Whole remote runs including setup took 432.75, 435.00 and 749.97 seconds. The
photometric run used A100 40GB, PyTorch 2.1.0 and CUDA 11.8. Peak allocated CUDA
memory during the photo stage was 1,274,663,936 bytes; peak reserved was
1,581,252,608 bytes. These are measurements for this reduced 480×270 profile,
not a minimum hardware requirement for full MoSca training. RTX 5070 Ti / Blackwell
compatibility has not been validated; the reproducible remote path is documented
in [CUSTOM_GPU.md](CUSTOM_GPU.md).

## Limits and failure cases

- Fur, eyes and vegetation lose detail. Tail edges flicker or appear irregular.
- The dynamic Gaussian pool alone contains holes and only part of the body.
  Some slowly moving foreground is represented by static Gaussians. That pool
  is motion-classified, not a semantic cat selection.
- Nearby views preserve a recognizable subject but expose stretched vegetation,
  incomplete surfaces and foreground/background ambiguity. Wide views and
  occlusion/re-entry are untested.
- Initial FOV search found no interior optimum and fell back to 45 degrees;
  final short-side FOV was about 44.2 degrees. Calibration and physical scale
  have no ground-truth validation. Synthetic-driving depth priors are a domain
  mismatch for dark fur and grass.
- MoSca evaluates discrete reconstruction indices. Original PTS are preserved;
  the last selected interval differs slightly from the others. No continuous-time
  motion interpolation is demonstrated.

The original upstream montage plays 48 frames at 10 fps (4.8 seconds). A separate
comparison exporter preserves the four-second source timing by holding sampled
frames, without inventing intermediate motion. A separate render-only run reloaded
the saved models without training and produced
96 native captures (48 source/nearby pairs). All pairs were visually inspected.
The final 1440×318 H.264 comparison decodes completely: 96 frames at 24 fps,
four seconds. Its original bytes and SHA-256 are recorded in the playback evidence.

Source-view RGB fit is **in-sample**, not held-out validation: mean PSNR 22.58 dB,
mean absolute RGB error 0.05378 on [0,1], minimum PSNR 17.85 dB at frame 18.
That frame visibly loses face/fur detail and misrepresents the tail boundary.
These measurements do not establish geometric accuracy. The nearby views were
inspected separately and retain vegetation stretching and incomplete surfaces.

The render-only run took 322.90 seconds including setup, with 25.14 seconds in
the playback stage. No new GPU run was needed for final closeout. At 20:08 UTC,
Modal reported approximately $2.97 gross workspace-cycle usage, $0 billed,
zero active tasks/GPUs and a $0 net-spend limit. Billing remains subject to
reconciliation; earlier snapshots must not be added to this total.

## Distinct experiments

The hosted SAM 3.1 → persistent Gaussian selection result is on **Balloon2**;
see [RESULTS.md](RESULTS.md). No hosted SAM request was made for the cat. No cat
mask IoU or semantic Gaussian selection is claimed, and Balloon2 masks must
never be reused on this footage. No object transformation or Phase 2 editor work
has been performed.

[Tracking](evidence/custom-tracking-2026-10-27.json),
[camera](evidence/custom-camera-2026-10-27.json),
[scaffold](evidence/custom-scaffold-2026-10-27.json),
[photo](evidence/custom-photo-2026-10-27.json), and
[checkpoint reload / final playback](evidence/custom-playback-2026-10-27.json) records bind measurements to exact
inputs and checkpoints. Original footage, model weights, checkpoints and rendered
assets remain separate from the published code repository.

## Controlled quality follow-up — modest improvement

Native PNGs already show the blur. Across 48 matched states, export-to-native
RGB PSNR averages 35.16 dB (source camera) and 35.58 dB (nearby); native source
fit remains 22.58 dB. Compression contributes less degradation than reconstruction.
The original video is1920×1080, prepared clip960×540, and fitting/native renders
480×270. Preview panels are pasted at native size.

One approved 4,000-step experiment keeps exactly the same inputs and resolution.
Only the total-step config changes; upstream also stretches duration-dependent
learning-rate/densification schedules. This reruns appearance fitting from the
preserved scaffold, not an optimizer resume. Hard invocation cap25min, one
A10040GB preferred/80GB fallback, no automatic retries or new model inference.
The conservative whole-run estimate is4.2672USD within a4.30USD planning ceiling.
The single run completed in **1,047.30 seconds (17m27s)** on A100 40GB,
including setup and rendering; the photo stage took 682.23 seconds. It produced
416,633 static and 49,162 dynamic Gaussians on the same 155-node scaffold.
All original baseline/input hashes remain unchanged, and all 96 native captures
were checked. No timeout, retry, extra model inference or enhancement occurred.

| In-sample source metric, mean over 48 frames | 2,000 steps | 4,000 steps |
| --- | ---: | ---: |
| RGB PSNR | 22.5806 dB | 23.8408 dB |
| RGB MAE on [0,1] | 0.053778 | 0.046472 |

The proposed numerical targets passed: **+1.2602 dB PSNR and 13.586% less MAE**.
PSNR improved on 47/48 frames; MAE improved on all 48. Frame 18 remains the worst
fit and slightly regressed in PSNR (17.8472 to 17.8356 dB). These are training-view
metrics, not held-out generalization or geometric accuracy.

All 48 source/nearby pairs were visually inspected, with native-sized comparisons
at 0, 18, 24 and 47. Ground detail and some facial structure improve, but fur,
tail tips and incomplete surfaces remain soft or ragged. The nearby views show
no obvious new gross motion collapse in sampled sequences, but residual artifacts
remain and temporal flicker was not quantitatively measured. Their orbit recipe
is matched; optimized camera matrices can differ, and there is no novel-view
ground truth. **This is a modest improvement, not a resolution of visual quality.**

Two four-second private previews fully decode at 24 fps/96 frames: the candidate
source/nearby comparison and the baseline-versus-candidate comparison. Both retain
sample-and-hold timing and unenhanced native renders. Video hashes, per-frame
metrics and checkpoint provenance are in the [quality evidence](evidence/custom-quality-4000-2026-10-27.json).

At 21:08 UTC the app was stopped with zero tasks/GPUs. Workspace-cycle usage was
approximately $3.80 gross and $0 billed; the difference from the pre-run snapshot
was about $0.83, subject to reconciliation. The $0 net-spend limit remained set.
The authorized single experiment is finished. No further run or Phase 2 work
has been started.

## Fixed scaffold position follow-up — no clear quality gain

One separately approved experiment held the **existing time-varying scaffold
trajectories unchanged during fitting**; it did not stop the cat's animation.
Only `photo_lr_np` and `photo_lr_np_final` changed to zero relative to the
4,000-step profile. The same initial scaffold, source frames, depth maps, tracks,
seed and other configuration were used. Gaussian parameters, node rotations and
camera poses still optimized. Both previous fits remain preserved.

The single attempt completed all 4,000 steps in **969.99 seconds (16m10s)**,
including setup/rendering, under the 25-minute cap. The photo stage took 640.89 s.
Modal allocated the authorized A100 80GB fallback after the ordered 40GB preference.
The saved-state invariant verifies exact equality of all 155 node positions at
all 48 timestamps, plus unchanged timestamp IDs and certainty masks. Finite
checkpoints contain 417,183 static and 51,005 dynamic Gaussians; 96 native renders
cover every source and nearby-view timestamp. No retry or new model inference ran.

| In-sample source metric, mean over 48 frames | Free positions, 4,000 steps | Fixed positions, 4,000 steps |
| --- | ---: | ---: |
| RGB PSNR | 23.8408 dB | 23.9114 dB |
| RGB MAE on [0,1] | 0.046472 | 0.046376 |
| Source-relative temporal-difference error | 0.063287 | 0.062996 |

This is only **+0.0705 dB PSNR and 0.206% less MAE**. PSNR improves on 28/48 frames
and MAE on 25/48. Frame 18 remains worst and slightly regresses: 17.8356 → 17.7966 dB.
Temporal-difference error decreases 0.459%; it compares changes in reconstruction
error between adjacent frames, without motion compensation. It includes camera
motion and fitting error and does not establish a perceptual flicker improvement.

All 48 source/nearby pairs were inspected, with native detail checks at 18, 24 and 47
and a decoded comparison-video frame. Fur and faces remain soft; tail-tip
silhouettes differ and distortions persist in both views. No clear overall visual
improvement or new gross animation collapse was demonstrated. Nearby-camera
matrices can differ despite the same orbit recipe; there is no novel-view ground
truth. Prior 4,000-step training used A100 40GB, while this fit used 80GB. With one run per
setting and no uncertainty estimate, tiny metric gains should not be treated as
statistically established improvements. The previous 4,000-step fit remains the reference.

The node diagnostic improves largely by construction: median residual falls from
3.226 to 0.293 pixels over 5,041 visible/certain node observations; head residual at
frame 18 falls from 7.34 to 2.46 pixels. Yet image quality barely changes. **Eliminating
scaffold-position drift alone is insufficient in this configuration.** Camera/depth
consistency, rotation/skinning/Gaussian fitting, low input resolution and source
softness remain plausible contributors. This experiment does not isolate them.

The private four-second before/after video contains source, free/fixed source-view
renders and free/fixed nearby views, with 96 frames at 24 fps. Native images are held
at original source timing without blending, enhancement or invented intermediate
states. Full decoding passed. The [evidence record](evidence/custom-fixed-positions-2026-10-28.json)
binds metrics, preview hashes and visual review to the saved checkpoints.

At 03:31 UTC the app was stopped, with zero active tasks/GPUs and the $0 net-spend
limit retained. Workspace usage was approximately **$4.37 cumulative and $0 billed**;
the change from the fresh pre-run snapshot was about **$0.70**, subject to billing
reconciliation, not an exact job invoice. No Volumes, extra inference, OPAL or
Phase 2 work was used. No execution blocker remains; visual quality is unresolved.
