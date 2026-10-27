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

- One A100 40GB, two CPU cores, 16GiB requested RAM, 900-second invocation;
  850-second subprocess cap leaves time to transfer diagnostics.
- No persistent Volume or automatic invocation retry. Default ephemeral disk
  billing is conservatively included as another 25.6GiB memory allowance.
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

GPU execution and custom reconstructed playback remain unverified until recorded
results and visual inspection are added. CPU staging tests are separate evidence.
