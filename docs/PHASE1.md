# Phase 1 status

Phase 1 is incomplete. Published-scene reconstruction access, official hosted SAM
tracking, exact-frame alignment and persistent Gaussian association are
demonstrated. Milestone 06 association cleanup/validation remains in progress;
milestone 03 custom 3–5-second video reconstruction has not run. A working sample
fallback is not a claim of completing that custom-video milestone.

The raw sample selection has 8,227 dynamic + 2 static rows. A separately saved manual
cleanup candidate excludes only the two flagged static outliers; it has not been
rendered and inherits no IoU. Source/nearby view checks cover only three times;
wide-view geometry, full occlusion/re-entry and continuous-time behavior remain
unvalidated. No Phase 2 editor or transformation is claimed.

## Custom clip prerequisites

Choose a rights-cleared 3–5-second MP4 with one clearly identifiable object,
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
The official UniDepth/BootsTAPIR alternate example is also untested in this project.

Default fitting requests 4,000 scaffold and 8,000 photometric steps plus preprocessing
and camera solving. First profile a bounded preprocessing subset on the chosen
clip, measuring load/inference GPU memory, system RAM, disk and elapsed time;
then use measurements to bound training. No minimum VRAM or full-training runtime
is established. The render-phase 1.184 GiB peak is not a training requirement.
The tested A100/CUDA11.8 environment is not validated for RTX5070Ti/Blackwell.

Completing Phase 1 requires an actual custom reconstruction with preserved inputs,
configs, checkpoints and renders, plus the association exit checks. Paid execution
and any further hosted inference require explicit cost/access checks. No new
model call or cloud job runs merely by invoking the CPU tests or preparation.
