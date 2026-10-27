# Custom-clip GPU execution

The custom input is the CC0 cat excerpt documented in CUSTOM_CLIP.md. Existing
Balloon2 reconstruction is not an input to this workflow. The 48 external depth
maps come from the separately documented Apache-2.0 Depth Anything V2 Small
checkpoint. No new hosted segmentation request is part of this run.

## Tracking and compatibility probe

`scripts/modal_custom.py` prints its bounded plan without contacting Modal.
Execution requires fresh billing/access evidence and explicit local inputs:

```sh
python scripts/modal_custom.py --execute --approval PRIVATE_VERIFICATION.json \
  --mosca /path/to/pinned/MoSca --priors outputs/custom-black-cat-depth-all48 \
  --weight /path/to/bootstapir_checkpoint_v2.pt \
  --wheel /path/to/torch_cluster-1.6.3+pt21cu118-cp310-cp310-linux_x86_64.whl \
  --out outputs/custom-tracking-new-run
```

Use the isolated Modal 1.6.2 client and existing approved authentication. The
runner never creates credentials. It stages only the exact pinned Git archive,
explicit runner modules, hash-verified source images/depth maps, official
BootsTAPIR weights and the matching compiled clustering wheel. It does not upload
repository history, environment variables, credentials, or unrelated files.

- One A100 (40GB preferred, 80GB authorized), two CPU cores, 16 GiB requested RAM, 900-second invocation;
  850-second subprocess cap leaves time to transfer diagnostics.
- No persistent Volume or automatic invocation retry. Default ephemeral disk
  billing is conservatively included as another 25.6 GiB memory allowance.
- Conservative estimate $3.4285 includes $2 overhead allowance; this is an
  estimate, not a provider billing cap. Fresh account controls remain mandatory.
- Builds both upstream rasterizers, repeats the independent native contribution
  reference, tests CUDA voxel clustering, and imports actual training stages.
- Runs official MoSca BootsTAPIR inference on all 48 frames with 4,096 uniform
  queries, batches of 256, fixed seed 12345, and explicit model evaluation mode.
- Validates time/coordinate/visibility shapes and the real `Saved2D.load_track`
  path. Returns exact tracks, input hashes, package inventory and diagnostics.

The official wheel source is
https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_cluster-1.6.3%2Bpt21cu118-cp310-cp310-linux_x86_64.whl
(3,300,027 bytes; SHA256
`c3c91386aa7037384c7ccbabcdef302e61333186a51a7d5ac58c80b40e22bd5a`).
BootsTAPIR weight provenance is in configs/preprocessing-sources.json.

## Actual reconstruction stage boundaries

Pinned MoSca provides `static_reconstruct(ws, log_path, cfg)` in
`lite_moca_reconstruct.py`, then `scaffold_reconstruct` and
`photometric_reconstruct` in `mosca_reconstruct.py`. A dedicated output directory
and OmegaConf config are required. Use `depth_dirname: external_depth` and
`tap_mode: bootstapir`; no dense RAFT flow is required for track-derived epipolar
classification. Skip photometric warmup (`photo_static_warm_steps: -1`) because
its mask path requires dense epipolar data.

Preserve completed boundaries:

1. `bundle/bundle_cams.pth`, `bundle/bundle.pth`, `tracker_epi.npz`.
2. `track_identification.npz`, `mosca/mosca.pth`.
3. `photometric_cam.pth`, `photometric_s_model_native_add3.pth`,
   `photometric_d_model_native_add3.pth`.

The upstream CLI does not resume training. Direct stage calls can reuse completed
prior stages, but Gaussian checkpoint files do not preserve optimizer/scheduler
state. Do not label a restarted photo fit as continuation of its optimizer.
Positive step counts are necessary to write valid stage outputs. Final videos
may still run even when periodic visualization is disabled, and model saves may
precede a visualization error: report process completion separately.

Camera FOV estimation needs more than 64 retained static correspondences in
suitable frame pairs at offsets 10/20. Dark cat fur and moving vegetation may
make the chosen sequence fail this requirement. Inspect dynamic track evidence
before accepting scaffold output: upstream's insufficient-dynamic-track fallback
uses `topk(largest=False)` and may select low-error static tracks. A checkpoint
alone is not evidence of usable dynamic reconstruction.

A real A100 40GB run built both rasterizers, passed the independent renderer
reference and CUDA voxel grouping, then exposed an upstream import defect before
tracking: `lib_mosca/camera.py` is a regular file containing only
`../lib_moca/camera.py`. The compatibility adapter binds the canonical
`lib_moca.camera` module to both `camera` and `lib_mosca.camera`, refusing
conflicting imports and leaving upstream files unchanged. The corrected imports
and actual point tracking passed on a real A100 80GB
(see evidence/custom-tracking-2026-10-27.json). Reconstructed playback remains pending.

Subsequent bounded runs can use `--cumulative-budget-usd 30` only under the
existing explicit credit authorization, with fresh balance/control/rate evidence.
The default envelope remains $5 and the per-run conservative estimate remains
$3.4285. This does not permit paid overage or new Meta inference.

Camera, scaffold and photo stages are available through `--fit-stage`, with
`--tracks` pointing to successful hash-bound tracking artifacts, `--config` to
the stage configuration and `--previous` to the completed predecessor archive
for later stages. Model/depth inputs must match the original frame hashes.
These stage runners are CPU tested preparation until actual GPU results exist.
The proposed bounded profile uses 500 camera, 1,000 scaffold and 2,000 photometric
steps. All saved model
tensors must be finite before a stage is accepted; visual quality is evaluated
separately.

## Allocation diagnosis

A subsequent exact `A100-40GB` request received a detected
`NVIDIA A100-SXM4-80GB` (85,118,156,800 bytes). The guard rejected it after 3.4
seconds, before setup or inference. The SDK preserves the requested GPU string;
no client fallback is configured. Modal's per-app billing report was not yet
populated, so free-upgrade billing is not established for this allocation.

`--gpu A100-80GB` is prepared as an explicit alternative. It requires matching
`explicit_gpu_authorization` in the fresh operator evidence record; the default
remains 40GB and different allocations are rejected. Published/provider rates at
inspection were $2.50/hour for 80GB and $2.10/hour for 40GB. A 15-minute run adds
at most $0.10 GPU cost, within the same conservative $3.4285 whole-run estimate.
The user subsequently authorized both sizes, preferring 40GB. `--gpu A100` now
requests ordered 40GB/80GB alternatives and requires both in `authorized_gpus` in
the evidence record. Both allocations pass the same architecture and CUDA checks;
other devices or memory sizes still fail. This avoids repeatedly rejecting an
authorized 80GB allocation. Actual custom point tracking is now demonstrated: 4,096 tracks, 3,382 retained by MoSca, across 48 frames. No custom reconstruction checkpoint is yet demonstrated.

The bounded fitting profile derives from MoSca's
`profile/demo/demo_fit.yaml` at the pinned revision. It retains upstream parameter
names and adjusts iteration counts, stage timing and flow-free tracking for this
experiment. The upstream core copyright/permission notice is preserved in
[mosca-core-MIT.txt](licenses/mosca-core-MIT.txt). This notice does not relicense
third-party rasterizers, model weights or data; their separate terms still apply.
