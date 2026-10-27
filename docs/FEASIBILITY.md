# Experiment 01: source inspection and feasibility

Inspected 2026-10-26; updated 2026-10-27. **Qualified end-to-end feasibility is
now demonstrated on one published Balloon2 scene.** A real hosted SAM 3.1 track
maps to a persistent MoSca Gaussian group, inspected at three times from source
and nearby cameras. It retains ragged geometry, missing patches and two static
outliers; no general geometric-correctness claim is made.

The actual native_add3 color derivative passes independent forward-basis checks,
and full-scene contribution conservation passes. Exact classes and source
revisions below are verified by the completed GPU runs, not an assumed stable
SDK. [Current results](RESULTS.md) distinguish measured evidence, exploratory
threshold correction and untested capabilities.

## Recommendation

Freeze an existing MoSca reconstruction and associate masks using its own
alpha-compositing renderer. Recover each Gaussian's visible contribution through
the renderer's gradient with respect to supplied colors. Aggregate foreground
and background evidence over calibrated source frames; retain low-evidence and
conflicting Gaussians as unknown. Store membership separately from checkpoints.

This is preferable to (1) projecting only Gaussian centers, which ignores extent
and occlusion, and (2) Feature4X-style feature distillation, which introduces a
second training problem and is unnecessary for the first binary-object test.
Do not retrain or alter reconstruction geometry for the initial experiment.

## Sources and extension points

| Repository | Inspected commit |
| --- | --- |
| [MoSca](https://github.com/JiahuiLei/MoSca) | `02404d2841b4e0dc2d9074860950bce5da579656` |
| [Feature4X](https://github.com/ShijieZhou-UCLA/Feature4X) | `15d78299ae5397b95a6102d3cf82a5f853c2f82c` |
| [Official SAM](https://github.com/facebookresearch/sam3) | `f1647e8d0c1d280991f4d1b7d974c2704537fe66` |

MoSca references below are relative to the pinned repository. Their line numbers
and methods were inspected directly.

| Component | Actual interface and implications |
| --- | --- |
| Pipeline | `mosca_precompute.py`, `mosca_reconstruct.py`: depth/tracking priors, camera estimation, motion scaffolds, photometric Gaussian optimization. `example.sh`, `profile/demo/demo_{prep,fit}.yaml` provide examples. |
| Static Gaussians | `lib_mosca/static_gs.py:41`, `StaticGaussian`; `load_from_ckpt` at 98; `forward` at 228 returns `(xyz, frame, scale, opacity, sph)`. |
| Dynamic Gaussians | `lib_mosca/dynamic_gs.py:31`, `DynSCFGaussian`; `load_from_ckpt` at 154; `forward(t: int, active_sph_order=None, nn_fusion=None)` at 391 returns the same five fields, deformed at frame index `t`. |
| Scaffold | `lib_mosca/mosca.py:26`, `MoSca`; `warp` at 495, `fast_warp` at 547, `get_skinning_weights` at 595. Node trajectories, rotations, topology, and learned skinning corrections control leaves. |
| Cameras | Use **`lib_moca.camera.MonocularCameras`**, as upstream visualization does, not the similarly named file in `lib_mosca`. `load_from_ckpt` at 62, `T_cw(t)` at 170, `K(H,W)` at 271. |
| Renderer | `lib_render/render_helper.py:31`, `render(gs_param,H,W,K,T_cw,...,colors_precomp=None,add_buffer=None)`. A five-tensor tuple or a list of these is accepted. Static/dynamic tuples concatenate in list order. |
| Backend | `GS_BACKEND` selects `native`, `native_add3` (default), or `gof`, **before import**. Initial harness targets `native_add3`, not untested interchangeability between backends. |
| Load example | `mosca_viz.py:16`, `load_model_cfg`, loads camera/static/dynamic state dictionaries; `mosca_evaluate.py` also demonstrates combined rendering. |
| Views | `viz_utils.py` contains `get_global_3D_cam_T_cw`, `get_move_around_cam_T_cw`, and source-view video routines. A small camera-relative translation is sufficient for our first nearby novel view. |

Checkpoint files saved by `lib_mosca/photo_recon.py:1134` are
`photometric_cam.pth`, `photometric_s_model_<backend>.pth`, and
`photometric_d_model_<backend>.pth`. They are separate PyTorch state dictionaries,
not a generic PLY and not a single scene object.

Both Gaussian models store `_xyz`, quaternion `_rotation`, raw `_scaling`, raw
`_opacity`, `_features_dc`, and `_features_rest`. Scale uses a bounded sigmoid
with `min_scale`/`max_scale`; opacity uses sigmoid. Read activated values from
`forward`, **not raw checkpoint parameters**. The returned `frame` is a 3×3
transform; covariance is `frame @ diag(scale**2) @ frame.T` (do not assume it is
always orthogonal after deformation).

Dynamic state also has `attach_ind`, `ref_time`, `_skinning_weight`, and `scf.*`,
including `_node_xyz[T,M,3]`, `_node_rotation[T,M,4]`, `_node_grouping`,
`topo_knn_ind`, and `topo_knn_mask`. With `leaf_local_flag`, `_xyz` is node-local;
`get_xyz` and `get_R_mtx` convert reference leaves before `scf.warp`. Do not
treat `_xyz` as a live world-space position.

`nn_fusion` masks opacity for unsupported reference times but preserves row
order. Frozen checkpoint rows therefore provide persistent identities:
`(checkpoint digest, static|dynamic, row index)`. Densification/pruning or
retraining invalidates them. Existing `group_id` / `_node_grouping` labels are
reconstruction groups, **not SAM object IDs**.

### Time and cameras

`forward(t)` indexes discrete scaffold frames; fractional seconds are not an
existing public interface. `MoSca.resample_time` interpolates nodes, mutates the
scaffold, and returns a remapping; it is not a drop-in arbitrary-time query.
Version one should use exact reconstruction frame indices. Continuous-time
interpolation is deferred and requires a copied scaffold and reference-time
remapping tests. Preserve original checkpoints.

`T_cw` is world-to-camera. The adapter uses `T_cw_list()[t]`, which also
handles delta cameras without the per-index path's apparent off-by-one assertion
at this revision (source inspection; no real delta checkpoint has been run). `K(H,W)` provides pixel intrinsics with focal lengths
relative to the shorter image side and principal point ratios relative to each
dimension. `cams.project` uses normalized coordinates; do not mistake its output
for pixel coordinates. Upstream renderer handles off-center principal points
by padding and cropping, so contribution extraction should reuse that path.

`Saved2D.load_rgb` (`lib_prior/prior_loading.py:179`) prefers `images.npz`,
otherwise lexicographically sorted `.jpg`/`.png` under `images/`. Record that
exact order, dimensions, and frame hashes. MoSca checkpoints do not establish
the original MP4 timestamps. A manifest must map decode presentation timestamps,
selected frames, resizing/cropping, and reconstruction indices. Never align two
independently decoded videos by guessed FPS; never interpret a missing SAM
response as a reliable all-background mask.

## Contribution algorithm

For Gaussian `g`, source frame `t`, and pixel `p`, the existing rasterizer uses
`w_gtp = T_gtp * alpha_gtp`, where `T` is front-to-back transmittance and `alpha`
includes opacity and projected Gaussian falloff. The actual kernel clamps alpha,
skips tiny contributions, and terminates low-transmittance rays. These details
are retained by using the upstream renderer.

Freeze geometry/opacity and supply a temporary, differentiable `N×3` color
tensor `c`. With black background, RGB is linear in `c`. For binary mask `M`:

```
L = sum_p(R[0,p]*M[p] + R[1,p]*(1-M[p]) + R[2,p])
dL/dc[g,:] = [sum_p w_gp*M[p], sum_p w_gp*(1-M[p]), sum_p w_gp]
```

One forward/backward pass produces foreground, background, and total evidence
without materializing a pixel-by-Gaussian matrix, modifying CUDA, or optimizing
a feature field. This is directly supported by
`gauspl_renderer_native_add3.py` (`colors_precomp`, lines 147–169) and its CUDA
backward kernel (`dchannel_dcolor = alpha*T`, `atomicAdd(dL_dcolors,...)`,
`cuda_rasterizer/backward.cu:535–555`). Independent forward-basis versus
backward-gradient comparison now passes on the actual CUDA backend, including
occlusion and zero opacity. Full-scene all-foreground conservation also passes.
Neither test substitutes for real-mask semantic association and novel-view
selection evaluation.

Render **all static and dynamic Gaussians together** for evidence. Rendering
them separately removes cross-object occluders and corrupts assignment. A
positive raster radius alone means potentially in-frame, not unoccluded.

Aggregate per-Gaussian visible foreground fractions over reliable frames,
weighting each frame by visible support but capping its weight so one close-up
cannot dominate. Record raw sums, visible-frame counts, and variance/conflict.
Require minimum evidence and multiple observations; leave unseen Gaussians
unknown even when the mask center overlaps them. Use separate foreground and
background thresholds with an uncertainty band; thresholds are experiment
parameters, not calibrated probabilities. Reject corrupt/nonfinite evidence.

Store object UUID, selected SAM session/object identity, checkpoint SHA256s,
static/dynamic row indices, soft scores, evidence, and thresholds in sidecars.
Keep the same rows while calling `d_model(t)` across time. Evaluate static and
dynamic contamination separately; the demo config even permits foreground
Gaussians in the static model (`gs_include_fg_in_static: True`).

Scaffold skinning neighborhoods could supply a weak regularizer or identify
disconnected selections. Start without it. Ablate smoothing only after the
baseline: motion similarity is not object identity (person/chair contact,
co-moving objects). Never propagate a semantic label blindly to an entire node
group or convert unseen leaves into confident foreground by smoothing.

## What Feature4X contributes

Feature4X stores static `_semantic_feature` in `lib_4d/gs_static_model.py`; its
dynamic model (`lib_4d/gs_ed_model.py`) obtains semantic information through its
scaffold. Its renderer accepts a sixth semantic field and returns `feature_map`.
`sam2/sam2_segmentation.py:254–284` decodes rendered features with
`Feature_heads.decode("sam2", ...)` and passes them to
`predictor.novel_view_inference_init`. This is trained feature distillation,
not binary-mask projection. Its checkpoint layout is not MoSca's five-field
layout. Reuse the conceptual separation of geometry, features, and views;
do not import Feature4X code or substitute SAM 3.1 masks into its feature decoder.

## SAM 3.1: verified versus unresolved

The official GitHub repository documents the March 27, 2026 SAM 3.1 release in
`RELEASE_SAM3p1.md`. `model_builder.py` downloads
`facebook/sam3.1/sam3.1_multiplex.pt`; the dedicated notebook uses
`build_sam3_multiplex_video_predictor()`. The unified
`build_sam3_predictor(version="sam3.1")` also exists. Its local protocol uses
`handle_request(start_session/resource_path)`, `add_prompt` with `frame_index`
and text/points, and streamed `propagate_in_video` with frame-indexed outputs.
Use returned object IDs, not response array positions; a noun may match multiple
instances. Explicit instance selection and prompt provenance belong in cache keys.

**The requested hosted API has now also been verified in official documentation.**
Hosted SAM uses `https://api.meta.ai/v1/responses`,
`model="sam-3.1"`, a Bearer `MODEL_API_KEY`, and uploaded video file IDs. The
prototype uses Meta's official Python parser, box-local mask placement, explicit
frame markers, and cached completed responses. See [SAM_API.md](SAM_API.md).
One hosted request now completed on the exact source
frames: object0 in all 12 frames,50 billed frames,$0.01 at published rates.
Overlays and source alignment were inspected. No local-model substitution or
additional inference occurred; full occlusion/re-entry remains untested.

Meta explicitly warns that IDs should not be assumed stable across complete
occlusion or exit/re-entry. A sparse or reverse-ordered frame sequence is legal;
an omitted target is marked unobserved and excluded from evidence. Missing or
low confidence (configurable policy, initially 0.5) is also excluded. A completed
empty response means no matches, not an API failure. Imported responses must
include matching original video/manifest/model/prompt provenance.

Cache key: exact source-frame manifest + model/version + prompt/coordinates +
chosen object ID + provider options. Cache completed raw responses and decoded
masks; verify completeness before reuse. Distinguish absent object, valid empty
mask, and missing/failed response. Never store credentials in the cache.

## Sample, hardware, and licenses

MoSca's [README checkpoint folder](https://drive.google.com/drive/folders/14awBsxTmY211ut9SnW5d1vuvVyThYhjl?usp=sharing)
is the first candidate. Its other weight link is for **2D prior models**, not
reconstructed scenes. The checked-in `demo/duck/images` is input imagery, not a
pretrained reconstruction. No scene `.pth` was found in the MoSca checkout.
First obtain one author-published scene plus exact source images and config;
inspect the archive before choosing a scene. The remote folder listing is verified: NVIDIA 5.34 GB, iPhone 19.87 GB,
Sintel 3.88 GB and TUM 3.09 GB archives. The selected Balloon2 archive entries, checksums and checkpoint compatibility
are now verified by real A100 runs; see PUBLISHED_SAMPLE.md and SETUP.md.
Other scenes in the archive have not been validated.

This coding worker has no `nvidia-smi`, `nvcc`, or `/dev/nvidia*` and uses CPU
PyTorch. A bounded remote Modal A100 run has now compiled/tested the actual
rasterizer successfully, then stopped with zero active tasks/GPUs.
The upstream MoSca environment targets Python 3.10, PyTorch 2.1.0/CUDA 11.8 and
CUDA extensions. **Do not expect that binary stack to run on an RTX 5070 Ti
(Blackwell).** A CUDA 12.8-capable stack and extension rebuild/compatibility
validation are needed for the local card. SAM's current README specifies
Python >=3.12 and torch >=2.7, with a torch 2.10/cu128 install example. Separate
environments plus mask files avoid coupling incompatible dependency stacks.

Smallest remote alternative: an existing Ubuntu Ampere GPU machine (e.g. A10 or
A100), matched CUDA 11.8 toolkit/PyTorch 2.1 environment, one published checkpoint,
and the native_add3 extension. The full Balloon2 checkpoint/camera/deformation path now passes on A100
40GB. At 480×270 with 866,900 Gaussians, render-phase peak allocated/reserved
memory was 1.184/1.375 GiB. Loading peaks, training memory, and RTX5070Ti
compatibility remain unmeasured; this is not a minimum-memory requirement.
See RESULTS.md for the scope and evidence.

MoSca's authored core is MIT, but its README excludes third-party priors and
rasterizers. The bundled 3DGS rasterizer license restricts use to research/
evaluation and requires separate consent for commercial use. Feature4X has
third-party licenses but no top-level license file at the inspected revision;
treat it as a reference, not code to copy. SAM code/weights use the custom SAM
License, not Apache/MIT; hosted service terms must be checked separately.
Dataset licenses apply to sample footage and redistribution of outputs. This
research inspection does not grant product or model redistribution rights.
The pinned GPU container also displays NVIDIA Deep Learning Container License
terms. Do not apply MoSca's core MIT license to the container or its components.

## Validation gates and status

1. **Implemented/tested on CPU where applicable:** source inspection, frozen revision
   manifest, read-only scene harness, contribution mathematics tests. **Passed on A100:**
   real Balloon2 checkpoint and six beginning/middle/end source/nearby renders,
   full-scene contribution conservation, and manual baseline inspection.
2. **Demonstrated on this sample:** one official hosted SAM 3.1 request, cached
   masks, returned identity, exact-pixel alignment, and a playable source overlay.
3. **Demonstrated:** actual CUDA derivative reference, nine-frame real-mask
   aggregation, persistent Gaussian sidecar and separate unknown/ambiguous rows.
4. **Demonstrated with limitations:** highlighted/isolated real Gaussian object at
   three times and two views; CLI confidence/support adjustment. All six views
   inspected; ragged boundaries and missing patches remain.
5. **Measured exploratory evaluation:** source IoU 0.880/0.895/0.868, per-pool
   visible contamination, training evidence variance and independent novel-view
   inspection. Frames0/6/11 were excluded from aggregation, but the correction
   followed initial evaluation feedback. These are not pristine benchmark scores.
6. **Deferred:** fixed world-space translation of selected evaluated means after
   `d_model(t)`, preserving deformation. Only after gate 5; document holes,
   disocclusions, overlap, shadows and unselected fragments. Never edit originals.

This establishes limited object-selection feasibility on one existing scene.
Custom reconstruction, wider views, full occlusion/re-entry, continuous-time
interpolation, training requirements and transformation remain unvalidated.
