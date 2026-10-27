# Fresh custom-video reconstruction results

A fresh four-second CC0 clip has been reconstructed with MoSca. The cat remains
recognizable across all 48 inspected timestamps, and its head turn and tail
movement persist when the camera is held fixed. This demonstrates usable dynamic
playback for this limited feasibility case. It does not establish accurate 3D
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
frames, without inventing intermediate motion. Native-frame capture validation
and final export are currently being closed out.

## Distinct experiments

The hosted SAM 3.1 → persistent Gaussian selection result is on **Balloon2**;
see [RESULTS.md](RESULTS.md). No hosted SAM request was made for the cat. No cat
mask IoU or semantic Gaussian selection is claimed, and Balloon2 masks must
never be reused on this footage. No object transformation or Phase 2 editor work
has been performed.

[Tracking](evidence/custom-tracking-2026-10-27.json),
[camera](evidence/custom-camera-2026-10-27.json),
[scaffold](evidence/custom-scaffold-2026-10-27.json), and
[photo](evidence/custom-photo-2026-10-27.json) records bind measurements to exact
inputs and checkpoints. Original footage, model weights, checkpoints and rendered
assets remain separate from the published code repository.
