# Published Balloon2 sample

Inspected 2026-10-26 after the user enabled `drive.usercontent.google.com`.
The earlier CONNECT 403 is resolved. No TLS or network-policy bypass was used.

The [MoSca author's folder](https://drive.google.com/drive/folders/14awBsxTmY211ut9SnW5d1vuvVyThYhjl)
contains `nvidia.zip`, file ID `1zZiZcCAZzB0ezAH0RO5cBJeMuUnbukBt`.
HTTP Range and the ZIP central directory establish **5,734,207,337 archive bytes,
2,570 entries**. The entire archive was not downloaded. Eighteen selected files
total **227,565,085 uncompressed bytes**, with ZIP CRC verification and measured
SHA-256 hashes in [the sample manifest](../configs/balloon2-sample.json).
These hashes ensure repeatability against our inspected copy; they are not
authenticated upstream signatures.

Selected reconstruction:
`nvidia/Balloon2/logs/nvidia_fit_native_add3_20241128_090115/`.
It contains `photometric_cam.pth`, `photometric_s_model_native_add3.pth`, and
`photometric_d_model_native_add3.pth`. No COLFREE variant is mixed into this set.

CPU inspection with `torch.load(weights_only=True, map_location='cpu')` succeeded:

| Data | Observed shape/count |
| --- | --- |
| Static Gaussian positions | 820,726 × 3 |
| Dynamic Gaussian positions | 46,174 × 3 |
| Dynamic skinning weights | 46,174 × 16 |
| Scaffold node positions | 12 × 3,537 × 3 |
| Scaffold topology indices | 3,537 × 16 |
| Camera quaternion/translation | 12 × 4 / 12 × 3 |
| Camera dimensions | H=270, W=480; non-delta poses; isotropic focal |
| Source images | `images/000.png` through `011.png`, all 480×270 |

The archive's command records specify `mosca_precompute.py` with
`profile/nvidia/nvidia_prep.yaml` and `mosca_reconstruct.py` with
`profile/nvidia/nvidia_fit.yaml`. Those configs exist in the inspected upstream
revision. A source frame was visually inspected: person holding a green balloon
in front of a painted wall. This is a **12-frame benchmark reconstruction**, not
the requested new 3–5 second video. Original acquisition/playback FPS is not
established. The prepared `outputs/balloon2-prepared/scene.mp4` uses an explicitly
chosen 3 fps / four-second playback timeline; every decoded RGB pixel matches
the original 12 images. It does not establish the original recording duration.

Reproduce the selected download (roughly 228 MB extracted, no full ZIP needed):

```bash
python scripts/fetch_mosca_sample.py \
  --manifest configs/balloon2-sample.json \
  --out data/mosca/published/extracted
```

The downloader checks bounded exact ranges, file sizes, CRC and manifest hashes;
it refuses to overwrite different files. Failed partial files remain for
inspection, preventing silent retries. Public Google large-file confirmation
is parsed at runtime; its transient URL is not saved in tracked code.

The author advertises checkpoints for reproduction, and MoSca's data instructions
require obeying original dataset licenses. The archive contains no LICENSE or
README granting asset redistribution. DynamicNeRF's repository code license does
not establish footage/checkpoint publication rights. Keep source images, weights
and derived sample renders local and unpublished pending that review. No gated
agreement was accepted, and original files are preserved.

## GPU probe

`scripts/modal_smoke.py --real-scene` retains the existing explicit account,
network and budget gates. It repeats the synthetic renderer reference, installs
the full checkpoint import dependencies, fetches the pinned upstream Git commit,
downloads only the manifest members, and calls `probe-mosca`. It uses the official
PyTorch3D 0.7.8 conda build `py310_cu118_pyt210` (BSD), SHA-256
`8b657140d23cdfa4e3f7a641bbc0dcd3e7c8830d4061481b0ef1eaa30e83d49f`.
Conda performs prefix relocation from the verified local package without solving
or replacing Torch/CUDA. Open3D's official CPU package supplies its imported
point-cloud utilities; MoSca deformation/rendering still runs on CUDA. PyPI
metadata identifies Open3D CPU 0.18.0, torch-geometric 2.5.3 and iopath 0.1.10
as MIT. Git is GPL-2.0 software installed only in the disposable container.
These dependency licenses do not grant sample-data or model publication rights.

The scaffold has only one node group (`0`), with attachment indices 0–3536
and reference times 0–11. These groups are not semantic object IDs. The initial
mask association therefore evaluates both static and dynamic Gaussian pools
together; using only dynamic points or treating a scaffold group as the balloon
would be unjustified. Shared motion near the person’s hands may also make
scaffold smoothing leak labels across the object boundary.

No SAM inference request, association claim, custom training, persistent Volume or automatic
retry is part of this probe. Actual outcomes are recorded in [RESULTS](RESULTS.md).
