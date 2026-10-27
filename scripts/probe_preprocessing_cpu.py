"""Probe pinned preprocessing code on CPU without pretrained weights or networking.

This checks imports, optional random-weight construction and a deterministic
upstream depth-mask operation. It is not depth/tracking inference or a CUDA test.
"""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mosca', type=Path, required=True)
    parser.add_argument('--metric3d', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--construct', action='store_true', help='construct models with random weights on CPU')
    args = parser.parse_args()
    for protected in (args.mosca, args.metric3d):
        if args.out.resolve().is_relative_to(protected.resolve()):
            parser.error('probe output must remain outside original source checkouts')
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root/'configs/preprocessing-sources.json').read_text())
    sources = {'mosca': (args.mosca, pins['mosca_revision']),
               'metric3d': (args.metric3d, pins['code_revisions_observed']['Metric3D'])}
    for name, (path, expected) in sources.items():
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            parser.error(f'{name} source revision differs from the inspected revision')
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      MPLCONFIGDIR=str((args.out/'matplotlib').resolve()),
                      XDG_CACHE_HOME=str((args.out/'cache').resolve()))
    # Python audit hooks remain active for this short-lived process. Also refuse
    # common weight loaders so model construction cannot silently fetch/load weights.
    def deny_network(event, arguments):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise RuntimeError('network disabled for this CPU probe')
    sys.addaudithook(deny_network)
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.manual_seed(0)
    def deny_weights(*arguments, **keywords):
        raise RuntimeError('pretrained weight loading disabled for this CPU probe')
    torch.load = deny_weights
    torch.hub.load_state_dict_from_url = deny_weights
    sys.path[:0] = [str(args.mosca.resolve()), str((args.mosca/'lib_prior').resolve()),
                   str(args.metric3d.resolve())]
    report = {'kind': 'weight-free-CPU-preprocessing-probe', 'network_disabled': True,
              'pretrained_weights_loaded': False, 'preprocessing_inference_validated': False,
              'cuda_validated': False, 'torch': torch.__version__, 'python': sys.version.split()[0],
              'source_revisions': {name: expected for name, (_, expected) in sources.items()},
              'checks': []}
    def check(name, function):
        started = time.monotonic()
        try:
            details = function()
            row = {'name': name, 'passed': True, 'details': details}
        except Exception as error:
            row = {'name': name, 'passed': False, 'error_type': type(error).__name__, 'error': str(error)}
        row['elapsed_seconds'] = time.monotonic()-started
        report['checks'].append(row)
        (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    for name in ('torchvision', 'cv2', 'kornia', 'omegaconf', 'tracking.bootstapir_wrapper',
                 'moca_processor', 'hubconf'):
        check('import '+name, lambda name=name: {'module': importlib.import_module(name).__name__})
    def depth_mask_check():
        from prior_loading import laplacian_filter_depth
        depths = np.ones((2, 32, 32), np.float32)
        depths[1, :, 16:] = 3
        valid, errors = laplacian_filter_depth(depths, threshold_ratio=.5)
        assert valid.shape == depths.shape and np.isfinite(errors).all()
        assert valid[0].all() and (errors[0] == 0).all()
        assert not valid[1, 4:-4, 15:17].any()
        assert valid[1, 4:-4, 4:12].all()
        return {'synthetic_depth_only': True, 'constant_valid_pixels': int(valid[0].sum()),
                'step_invalid_pixels': int((~valid[1]).sum())}
    check('upstream constant/step depth boundary operation', depth_mask_check)
    if args.construct:
        def metric_constructor():
            import hubconf
            model = hubconf.metric3d_vit_small(pretrain=False).cpu().eval()
            return {'parameter_count': sum(p.numel() for p in model.parameters()),
                    'random_weights': True, 'forward_executed': False}
        def tapir_constructor():
            from tracking.tapnet_pt.tapir_model import TAPIR
            model = TAPIR(pyramid_level=1).cpu().eval()
            return {'parameter_count': sum(p.numel() for p in model.parameters()),
                    'random_weights': True, 'forward_executed': False}
        check('Metric3D-small random-weight constructor', metric_constructor)
        check('BootsTAPIR random-weight constructor', tapir_constructor)
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name')
        if name and name not in packages:
            packages[name] = dist.version
    report['packages'] = dict(sorted(packages.items()))
    report['all_requested_checks_passed'] = all(row['passed'] for row in report['checks'])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['all_requested_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
