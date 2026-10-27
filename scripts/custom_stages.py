"""Guard actual MoSca reconstruction stage boundaries; never resume an optimizer."""
from pathlib import Path


def required_inputs(stage):
    camera=('bundle/bundle_cams.pth','bundle/bundle.pth','tracker_epi.npz')
    if stage=='camera':return ()
    if stage=='scaffold':return camera
    if stage=='photo':return camera+('mosca/mosca.pth','track_identification.npz')
    raise ValueError('unknown reconstruction stage')


def validate_predecessor(directory,stage):
    for name in required_inputs(stage):
        p=directory/name
        if not p.is_file() or p.is_symlink() or p.stat().st_size==0:
            raise FileNotFoundError(f'missing completed predecessor artifact: {name}')


def require_dynamic_evidence(mask,minimum=32):
    import numpy as np
    mask=np.asarray(mask)
    if mask.ndim!=1 or mask.dtype!=np.bool_ or int(mask.sum())<minimum:
        raise ValueError('insufficient actual dynamic tracks; refusing upstream lowest-error fallback')
    return int(mask.sum())


def install_camera_alias(canonical_module,registry):
    """Resolve pinned upstream's text-only symlink placeholder without editing it."""
    names=('camera','lib_mosca.camera')
    if any(name in registry and registry[name] is not canonical_module for name in names):
        raise ValueError('conflicting camera module already imported')
    for name in names:registry[name]=canonical_module


def validate_binding(directory,stage,input_hashes):
    import json
    from scripts.custom_tracking import digest
    validate_predecessor(directory,stage)
    if stage=='camera':return
    predecessor={'scaffold':'camera','photo':'scaffold'}[stage]
    report=json.loads((directory/(predecessor+'-report.json')).read_text())
    if (report.get('passed') is not True or report.get('original_inputs_preserved') is not True
        or report.get('input_sha256')!=input_hashes):
        raise ValueError('predecessor failed or belongs to different input priors/tracks')
    for name in required_inputs(stage):
        if digest(directory/name)!=report['output_sha256'].get(name):
            raise ValueError('predecessor checkpoint hash mismatch')
