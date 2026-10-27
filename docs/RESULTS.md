# Experiment 01 — qualified feasibility result

Updated 2026-10-27. **One real hosted SAM 3.1 track was associated with a persistent
MoSca Gaussian group that remains recognizable at three times and two nearby
camera viewpoints.** This is a limited demonstration on one published sample,
not proof of complete geometry, generalization or production readiness.

## What ran

The author-published Balloon2 checkpoint contains **820,726 static and 46,174
dynamic Gaussians**, a **3,537-node scaffold**, and **12 calibrated 480×270 frames**.
The exact pinned MoSca camera/static/dynamic classes and native_add3 renderer
loaded unchanged checkpoints. All 18 sample-file hashes remain unchanged.
[Sample provenance and download recipe](PUBLISHED_SAMPLE.md).

A lossless RGB H264 clip preserved every original source pixel. Its four-second
playback uses a chosen 3 fps timeline; original acquisition timestamps are unknown.
One hosted `sam-3.1` request, prompt **green dinosaur balloon**,
returned **object0 in all 12 frames** with no official-parser diagnostics. It
accepted the lossless clip without conversion. End-to-end upload/request latency
was **13.578s**; usage reported **50 billed frames**, consistent with **$0.01** at
published rates. No additional inference was made. The repeated 0.93361 confidence
is not twelve independent calibrated accuracy estimates. All 12 overlays were
inspected: rotation was tracked without broad person/wall spill; precise boundary
accuracy and full occlusion/re-entry remain untested. [SAM evidence](evidence/sam-3.1-2026-10-27.json).

Association uses visible alpha/transmittance contribution from the actual
renderer, including opacity, projected extent, depth order and all occluders.
Color gradients passed an independent forward-basis CUDA fixture and full-scene
conservation checks. Nine source frames supplied evidence; **0,6,11** were excluded
from aggregation. Both Gaussian pools were evaluated together. No feature field
or geometry was trained. The scaffold's single grouping label is not a semantic ID.

## Results and corrective experiment

The initial minimum visible mass of 1 pixel selected 2,228 dynamic Gaussians but left
many holes. CPU analysis of saved training evidence showed that small visible
splats were being discarded. One corrective run lowered only this support cutoff
to **0.1 pixel**; membership threshold remained **0.8**, minimum observations 2,
maximum frame weight 32, and maximum evidence variance 0.06. Training target-mass
recovery increased 60.9%→87.2%, with training leakage about 1.1%.

The final persistent object ID is **166dfad9-bd7b-5a98-ae91-b0ed44de8807**, with
**8,227 dynamic + 2 static Gaussians**. The sidecar retains per-row soft scores,
evidence, selected indices, hashes and parameters; **608,217 unknown and 1,885
ambiguous rows** remain separate. Fixed row identity is bookkeeping; the images
supply the evidence that this particular group follows the target.

| Reconstruction frame | Initial source IoU | Refined source IoU | False-positive pixels | False-negative pixels | Dynamic outside-mask visible mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6191 | 0.8801 | 201 | 331 | 6.19% |
| 6 | 0.6359 | 0.8949 | 44 | 478 | 1.78% |
| 11 | 0.5738 | 0.8680 | 334 | 397 | 7.97% |

IoU uses the selected group's **visible mask with all unselected occluders still
present**, threshold 0.5; isolated-object alpha is not substituted. These are
comparisons against SAM masks, not independent human ground truth. The support
correction followed inspection of initial evaluation failures, so revised numbers
are **exploratory**, not an untouched benchmark. Mean selected training-evidence
variance is 0.001975; this does not measure full-occlusion identity recovery.

All six refined highlighted and isolated source/nearby views were inspected.
The same recognizable balloon follows its changing pose, without broad person
or wall geometry in the isolated images. Nearby cameras translate along local
right by 3% of median positive depth; they do not test a wide orbit or unseen back.
Coverage is substantially improved, but ragged boundaries, missing patches and
imperfect texture remain. **High 2D IoU is not geometric correctness.**

The **two static rows are suspicious outliers**, retained honestly in the saved
result. Their outside-mask mass ratios are 45.4%/0%/99.6% at the three evaluation
times. Across training views they contribute only 3.54 pixel-mass, about 0.0092% of
selected visible mass; absolute held-out static mass was not recorded. Do not
use the small row count to claim negligible held-out contamination.
[Separate visual review](evidence/association-visual-review-2026-10-27.json) records
these limits. Original automatic reports retain their conservative pending-review
flags; this review documents the completed inspection without rewriting them.

## Reproducibility and current scope

See [setup](SETUP.md), [source interfaces](FEASIBILITY.md),
[hosted API](SAM_API.md), [sample provenance](PUBLISHED_SAMPLE.md), and
[Phase 1 continuation](PHASE1.md). Numerical evidence is in `docs/evidence/`.
Published JSON omits service response IDs and private account/cleanup records;
scientific hashes, parameters, metrics and limitations are retained. Package
inventories omit local build paths and are labeled filtered rather than exact logs.

The original source footage, weights, decoded masks, renders and private videos
are not included because asset redistribution rights remain unestablished.
Local recipes reference these inputs; a fresh clone does not contain them.

CPU verification for this snapshot: 57 discovered, 56 passed, one explicit CUDA
skip. Earlier independent native CUDA tests passed remotely. The baseline
completed in 215.067s, refined association in 210.475s. Render-phase memory was
1.184GiB allocated /1.375GiB reserved after resetting counters following load;
this is not loading/training memory or a minimum VRAM requirement. Tested GPU:
A10040GB, PyTorch2.1.0, CUDA11.8. RTX5070Ti compatibility remains unvalidated.

Phase 1 is not complete: custom-video reconstruction remains outstanding.
A separate CPU cleanup candidate removes two explicitly flagged static rows,
retaining8,227dynamic rows. It has no rendered quality measurements yet.
Full occlusion/re-entry, wider views, continuous-time interpolation and
translation are unvalidated. No complete editor or transformation is claimed.
