"""Run the official Apache-2.0 metric-depth Small model on CPU, then load its
outputs with pinned MoSca Saved2D. No tracking, cameras or reconstruction fitting.
Weights must already exist locally and match the published SHA256; no downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

CODE_REVISION = 'a561b849ebae10a6f5ef49e26c83cbbcd36c71bf'
WEIGHT_REVISION = 'c725b8589bdf6ab04072cab74c0467830db80d6d'
WEIGHT_SHA256 = '9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--mosca', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--all-frames', action='store_true')
    args = p.parse_args()
    for protected in (args.prepared, args.source, args.mosca, args.weights.parent):
        if args.out.resolve().is_relative_to(protected.resolve()):
            p.error('output must be separate from source images, code and weights')
    root = Path(__file__).resolve().parents[1]
    mosca_revision = json.loads((root/'upstream-revisions.json').read_text())['mosca']['commit']
    for path, expected in [(args.source, CODE_REVISION), (args.mosca, mosca_revision)]:
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            p.error('source revision differs from inspected revision')
    if args.weights.stat().st_size != 99221808 or digest(args.weights) != WEIGHT_SHA256:
        p.error('weight identity differs from the official published checkpoint')
    manifest_path = args.prepared/'custom-scene.json'
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['frames']
    if not 4 <= len(rows) <= 120 or max(manifest['width'], manifest['height']) > 1024:
        p.error('prepared input exceeds bounded probe dimensions/count')
    indices = list(range(len(rows))) if args.all_frames else sorted({0, len(rows)//3, 2*len(rows)//3, len(rows)-1})
    for i in indices:
        row = rows[i]
        if Path(row['file']).name != row['file'] or digest(args.prepared/'images'/row['file']) != row['sha256']:
            p.error('unsafe or changed prepared frame')
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      MPLCONFIGDIR=str((args.out/'matplotlib').resolve()),
                      XDG_CACHE_HOME=str((args.out/'cache').resolve()))
    def deny_network(event, arguments):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('network disabled during CPU inference')
    sys.addaudithook(deny_network)
    import torch
    import cv2
    import numpy as np
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        p.error('this probe is CPU-only; upstream infer_image chooses device automatically')
    torch.set_num_threads(2)
    sys.path.insert(0, str((args.source/'metric_depth').resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2
    started = time.monotonic()
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=80)
    model.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
    model.cpu().eval()
    model_load_seconds = time.monotonic()-started
    images = args.out/'images'; images.mkdir()
    depths = args.out/'external_depth'; depths.mkdir()
    report = {'kind': 'real-CPU-depth-prior-probe', 'source_model': 'depth-anything/Depth-Anything-V2-Metric-VKITTI-Small',
              'model_code_revision': CODE_REVISION, 'weight_revision': WEIGHT_REVISION,
              'weight_sha256': WEIGHT_SHA256, 'weight_license': 'Apache-2.0',
              'mosca_revision': mosca_revision, 'prepared_manifest_sha256': manifest_hash,
              'prepared_frame_indices': indices, 'all_prepared_frames': args.all_frames,
              'input_size': 518, 'max_depth': 80, 'depth_units_claimed_by_model': 'meters',
              'physical_scale_verified': False, 'cpu_threads': 2, 'torch': torch.__version__,
              'model_load_seconds': model_load_seconds, 'rows': [], 'completed': False,
              'sam_called': False, 'gpu_used': False, 'tracking_completed': False,
              'reconstruction_completed': False, 'network_disabled_during_inference': True}
    def save():
        report['peak_process_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    for i in indices:
        row = rows[i]; src = args.prepared/'images'/row['file']
        before = time.monotonic()
        with torch.inference_mode():
            depth = model.infer_image(cv2.imread(str(src)), input_size=518).astype(np.float32)
        if depth.shape != (manifest['height'], manifest['width']) or not np.isfinite(depth).all() or (depth <= 0).any():
            raise ValueError('depth output is nonpositive, nonfinite or misaligned')
        shutil.copyfile(src, images/row['file'])
        target = depths/(Path(row['file']).stem+'.npz')
        np.savez_compressed(target, dep=depth)
        report['rows'].append({'prepared_index': i, 'file': row['file'], 'source_image_sha256': row['sha256'],
                               'source_pts_seconds': row['source_pts_seconds'], 'depth_sha256': digest(target),
                               'depth_min': float(depth.min()), 'depth_median': float(np.median(depth)),
                               'depth_max': float(depth.max()), 'elapsed_seconds': time.monotonic()-before})
        save(); print(f'CPU depth completed for prepared frame {i}', flush=True)
    sys.path[:0] = [str(args.mosca.resolve()), str((args.mosca/'lib_prior').resolve())]
    from prior_loading import Saved2D
    loaded = Saved2D(str(args.out)).load_dep('external_depth', depth_boundary_th=.3)
    assert tuple(loaded.dep.shape) == (len(indices), manifest['height'], manifest['width'])
    assert loaded.frame_names == [Path(rows[i]['file']).stem for i in indices]
    for j, i in enumerate(indices):
        original = np.load(depths/(Path(rows[i]['file']).stem+'.npz'), allow_pickle=False)['dep']
        assert np.array_equal(original, loaded.dep[j].numpy())
        assert digest(args.prepared/'images'/rows[i]['file']) == rows[i]['sha256']
    assert digest(manifest_path) == manifest_hash
    report.update(completed=True, mosca_depth_loader_passed=True,
                  loaded_valid_depth_fraction=float(loaded.dep_mask.float().mean()),
                  elapsed_seconds=time.monotonic()-started)
    save()
    print(f'CPU depth and actual MoSca loading passed for {len(indices)} frames; reconstruction has NOT run.')


if __name__ == '__main__':
    main()
