# Reproducing the custom-clip GPU stages

The input is the fresh CC0 cat excerpt in [CUSTOM_CLIP.md](CUSTOM_CLIP.md), with
48 images at 480×270, external Depth Anything V2 Small depth maps and official
BootsTAPIR tracks. Balloon2 checkpoints are not inputs to this workflow.
[Measured results](CUSTOM_RESULTS.md) distinguish this reconstruction from SAM
object-selection evidence on the published sample.

## Environment and limits

`scripts/modal_custom.py` prints a plan by default; only `--execute` starts work.
Use the isolated Modal 1.6.2 client with existing workspace authentication and
fresh account/rate/control evidence. The runner never creates credentials.
Verification records are private, single-use and valid for at most one hour.
The SDK does not expose the remaining promotional-grant balance; establish it
separately and retain a reconciliation reserve. The estimate is not a billing cap.

The successful environment uses the pinned PyTorch 2.1.0 / CUDA 11.8 image,
Python 3.10, native_add3 rendering and both compiled upstream rasterizers.
Package inventories and exact source/weight hashes are in the evidence records.
Do not install the CPU dependency lock into this CUDA environment.

- One A100, preferring 40GB; explicitly authorized 80GB is supported.
- Two CPU cores and 16 GiB RAM limits; conservative billing includes another
  25.6 GiB for the default ephemeral disk's possible memory charge.
- 900-second invocation, 850-second outer process limit and 800-second cumulative
  subprocess budget. No persistent Volumes or automatic invocation retries.
- Conservative whole-run estimate: $3.4285, including $2 setup/transfer allowance.
  The default cumulative envelope is $5; `--cumulative-budget-usd 30` requires
  the existing explicit credit authorization and fresh checks, with no paid overage.
- `--gpu A100` requests ordered 40GB/80GB alternatives and requires both in the
  verification record. Family, memory size, architecture and CUDA are checked.

The runner exports only the pinned upstream Git archive, explicit runner files,
verified priors, weights and compiled clustering wheel. It excludes repository
history, credentials, unrelated files and environment secrets. It builds both
rasterizers and runs an independent compositing reference and voxel-grid check.

## Inputs and tracking

```sh
python scripts/modal_custom.py --execute --gpu A100 \
  --approval PRIVATE_TRACKING_VERIFICATION.json --cumulative-budget-usd 30 \
  --mosca /path/to/pinned/MoSca --priors outputs/custom-black-cat-depth-all48 \
  --weight /path/to/bootstapir_checkpoint_v2.pt \
  --wheel /path/to/torch_cluster-1.6.3+pt21cu118-cp310-cp310-linux_x86_64.whl \
  --out outputs/custom-tracking
```

This experiment enforces 48 frames. Official BootsTAPIR inference uses 4,096
uniform queries, batches of 256, seed 12345 and evaluation mode. Actual MoSca
loading retains 3,382 tracks on the chosen clip. Track identities are not semantic
SAM identities. No hosted segmentation request is made by this workflow.

Official wheel source:
https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_cluster-1.6.3%2Bpt21cu118-cp310-cp310-linux_x86_64.whl

Wheel SHA256:
`c3c91386aa7037384c7ccbabcdef302e61333186a51a7d5ac58c80b40e22bd5a`.
Prior weight/code revisions and licenses are in `configs/preprocessing-sources.json`.

## Camera, scaffold, appearance and playback

For each stage, repeat the explicit common inputs above with a **new verification
record and new output directory**, plus these arguments:

| Stage | Additional arguments | Saved boundary |
| --- | --- | --- |
| Camera | `--fit-stage camera --tracks outputs/custom-tracking/artifacts --config configs/custom-cat-camera.json` | `bundle/bundle_cams.pth`, `bundle/bundle.pth`, `tracker_epi.npz` |
| Scaffold | `--fit-stage scaffold --tracks outputs/custom-tracking/artifacts --config configs/custom-cat-fit.json --previous outputs/custom-camera/artifacts/stages.zip` | `track_identification.npz`, `mosca/mosca.pth` |
| Photo | `--fit-stage photo --tracks outputs/custom-tracking/artifacts --config configs/custom-cat-fit.json --previous outputs/custom-scaffold/artifacts/stages.zip` | `photometric_cam.pth`, `photometric_s_model_native_add3.pth`, `photometric_d_model_native_add3.pth` |
| Playback only | `--fit-stage playback --tracks outputs/custom-tracking/artifacts --config configs/custom-cat-fit.json --previous outputs/custom-photo/artifacts/stages.zip` | Native source/nearby PNGs, a separate diagnostic video and `playback-report.json` |

Actual upstream functions are `static_reconstruct` in `lite_moca_reconstruct.py`,
then `scaffold_reconstruct` and `photometric_reconstruct` in `mosca_reconstruct.py`.
Playback loads the three saved model states and uses the existing visualization
function with gradients disabled. It never runs an optimizer and checks model
hashes again afterward. Each predecessor must have passed, match every input
hash, and retain its required checkpoint bytes.

The upstream CLI does not resume optimizer/scheduler state. Reusing a completed
stage boundary is supported here; restarting photo fitting is not optimizer
continuation. Failed stages retain available diagnostics and never overwrite
original checkpoints. A finite checkpoint alone is not a quality verdict.

The profile uses 500 camera, 1,000 scaffold and 2,000 photo steps. Camera FOV search
used its fallback on this clip. Scaffold classification requires at least 32 real
dynamic tracks and refuses the upstream low-error fallback. The native_add3
backend does not apply GOF-specific normal losses even if those configuration
keys are present. Dense RAFT flow and photometric warmup are not used.

## Verified compatibility corrections

- Pinned `lib_mosca/camera.py` is a text-only symlink placeholder. Bind both legacy
  import names to canonical `lib_moca.camera`; refuse conflicting modules.
- `dynamic_solver.viz_curve` returns normalized float RGB. Convert only its
  still-image call to uint8; its full-sequence caller already scales floats.
  The first corrected run's double-scaled curve video is excluded from evidence.
- Capture the prediction panel from the actual `viz2d_total_video` function's
  globals. Patching the package-qualified visualization module misses MoSca's
  legacy top-level import. The photo run's original montage is valid, but that
  first hook produced no native PNGs; render-only validation is separate.
- A 60-second apt-install substep expired during successful package downloads.
  Its allowance is now 180 seconds, still inside the unchanged whole-run limits.
  Automatic apt retries remain disabled.

Upstream files remain unchanged. The fitting profile derives from MoSca's pinned
`profile/demo/demo_fit.yaml`; retain [its MIT notice](licenses/mosca-core-MIT.txt).
This does not relicense third-party rasterizers, weights or footage; see the
separate dependency/model license audit.

## Timed comparison export

After successful photo or playback capture:

```sh
python scripts/make_custom_preview.py \
  --scene outputs/custom-black-cat-input \
  --priors outputs/custom-black-cat-depth-all48 \
  --stages outputs/custom-playback/stages \
  --out outputs/custom-preview
```

The video labels source, source-camera reconstruction and moving nearby-camera
reconstruction. It holds the 48 sampled states on the original four-second
source timeline at 24 fps, without motion interpolation. The sidecar records all
96 output-to-reconstruction indices, hashes and the 0.51-microsecond tolerance
for ffprobe's six-decimal timestamp rounding.

The exporter binds the original timing manifest through the depth-prior report
to training inputs. Reloaded captures additionally bind the playback report to
the original photo report and all three trained checkpoints. Changed timing,
render bytes or model bytes are rejected before output creation. No SAM mask,
semantic selection or object edit is implied by this preview.

Final validation: the corrected render-only run reloaded all three trained models,
preserved their hashes, and exported 96 native captures without further training.
All captures and the encoded comparison passed closeout checks; see
[playback evidence](evidence/custom-playback-2026-10-27.json).

## Explicit quality experiment

`--quality-fit --fit-stage photo --config configs/custom-cat-quality-4000.json`
selects the separately authorized25min profile. Default runs retain15min.
Quality mode requires an approval record with `quality_fit_4000_authorized: true`,
rejects config changes beyond4000total steps, and includes the larger cost in
the cumulative credit check. Its worker/child/invocation deadlines are1400/1450/1500s.
This does not implement resumable optimization or authorize further experiments.

The approved quality run completed on A100 40GB in 1047.30 seconds including
setup/rendering. See [quality results](CUSTOM_RESULTS.md) for measured improvement,
residual failures and billing. This single completed run does not authorize retries.

## Fixed scaffold position comparison

The named `--quality-fit --fixed-positions --fit-stage photo` experiment uses
`configs/custom-cat-fixed-positions-4000.json`. It differs from the preceding
4,000-step profile only in `photo_lr_np=0` and `photo_lr_np_final=0`. Scaffold
rotations, Gaussian parameters, cameras and all other fitting settings remain
eligible for the same optimization as before. This tests position drift; it is
not an assumption that freezing positions improves reconstructed images.

The same 1,500-second invocation cap and $4.2672 conservative whole-run estimate
apply. Fresh private execution evidence must include both
`quality_fit_4000_authorized=true` and `fixed_positions_fit_authorized=true`;
prior one-use authorization records cannot be reused. The generic photo command
rejects this zero-position-learning-rate configuration without the named flags.

Start from the preserved scaffold-stage archive, use a new output directory,
and preserve both earlier photometric fits. Before reporting success, the runner
requires exact equality of saved scaffold positions, timestamp IDs and certainty
masks against the input scaffold. This invariant is separate from visual quality:
compare matched source and nearby renders, head/tail detail and temporal artifacts.
