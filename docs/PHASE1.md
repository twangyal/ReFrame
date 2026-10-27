# Phase 1 evidence audit

Phase 1 has pipeline evidence for milestones 01–06 and Gate A across two
experiments: Balloon2 supplies hosted SAM-to-Gaussian selection, while the fresh
CC0 cat clip supplies custom reconstruction. **Visual quality remains unresolved:**
the cat reconstruction loses substantial detail even before preview encoding.
The earlier completion statement described execution evidence too broadly.
A single approved 4,000-step comparison improved mean PSNR by 1.26 dB and
MAE by 13.6%, with modest visual improvement and substantial residual blur.
See [quality results](CUSTOM_RESULTS.md). Phase 2 remains stopped.

| Roadmap milestone | Evidence and status |
| --- | --- |
| 01 — Environment & GPU setup (CORE) | Pinned CPU/CUDA environments, both native rasterizers, independent compositing reference, real prior inference, camera/scaffold/Gaussian fitting on A100. Local RTX 5070 Ti compatibility remains untested. |
| 02 — Known-good MoSca playback (CORE) | Preserved published Balloon2 checkpoints, calibrated source/nearby rendering and inspected playback. |
| 03 — Custom short-video reconstruction (TARGET) | Fresh CC0 four-second cat clip; 48 exact inputs, depth/track priors, 500 camera steps, 1,000 scaffold steps, 2,000 photo steps, finite checkpoints and inspected dynamic playback. Saved models independently reloaded; all 96 native captures and final four-second export verified. |
| 04 — Meta SAM 3.1 segmentation (CORE) | One actual hosted request on Balloon2, cached masks and returned object ID 0 across all 12 exact source frames. No custom-cat SAM inference. |
| 05 — Frame, time & camera alignment (CORE) | Sample masks match reconstruction frame order/resolution; calibrated reprojection comparisons exist. Sample acquisition cadence is unknown and synthetic playback timing is disclosed. Custom source PTS, resize and checkpoint/input hashes are preserved. |
| 06 — 2D mask → 4D Gaussian association (CORE) | Renderer contribution evidence yields persistent membership; cleaned 8,227-dynamic selection inspected at all 12 source/nearby times. Manual two-row cleanup is exploratory, not a general rule for static objects. |

**Gate A:** the sample demonstrates that a SAM-selected object maps to actual
persistent 4D Gaussians rather than only pixels. The fresh clip independently
demonstrates reconstruction from new footage. This is not yet a single fresh-clip
SAM-to-selection experiment, broad-scene validation, or a production editor.

The evidence is qualified: ragged/missing geometry remains; wide novel views,
full occlusion/re-entry, continuous-time interpolation and geometric correctness
are unverified. High source-view IoU does not settle those questions. Original
checkpoints and source inputs are preserved.

## Reproduction and evidence

- [Inspected MoSca interfaces and association method](FEASIBILITY.md)
- [Sample selection measurements and limitations](RESULTS.md)
- [Fresh source/license/preparation](CUSTOM_CLIP.md)
- [Custom reconstruction results and limits](CUSTOM_RESULTS.md)
- [Prior license/API audit](PREPROCESSING.md)
- [CPU depth compatibility](PREPROCESSING_CPU.md)
- [Bounded remote tracking, stage fitting and playback](CUSTOM_GPU.md)

No new Meta call, transformation, desktop editor, public asset publication,
hackathon submission or legal agreement is implied by this result. Published
code/test/documentation checkpoints exclude weights, datasets and rendered media.
