"""Guard actual MoSca reconstruction stage boundaries; never resume an optimizer."""
from pathlib import Path


def uint8_curve_renderer(renderer):
    """Convert only MoSca curve diagnostics from normalized RGB for image writers.

    Matches viz_helper.viz_scene's own save conversion for the still-image call.
    The full-sequence caller already scales floats to bytes, so retain its
    original contract. Model tensors and upstream files remain unchanged.
    """
    from functools import wraps
    import numpy as np
    @wraps(renderer)
    def render(*args,**kwargs):
        frames=renderer(*args,**kwargs)
        if not kwargs.get('only_viz_last_frame',False):return frames
        result=[]
        for frame in frames:
            frame=np.asarray(frame)
            if frame.ndim!=3 or frame.shape[-1]!=3 or not np.isfinite(frame).all():
                raise ValueError('curve diagnostic must be finite RGB')
            if frame.dtype==np.uint8:result.append(frame)
            elif np.issubdtype(frame.dtype,np.floating):
                result.append(np.clip(frame*255,0,255).astype(np.uint8))
            else:raise ValueError('unsupported curve diagnostic dtype')
        return result
    return render


def capture_rgb_diagnostics(renderer,output):
    """Save RGB panels before upstream montage/video compression, without rerendering.

    Pinned viz2d_one_frame starts with [target, prediction, error] at native H/W.
    Only original-camera and explicitly named moving-nearby-camera calls count.
    """
    from functools import wraps
    import numpy as np
    from PIL import Image
    @wraps(renderer)
    def render(model_tid,s2d,cams,*args,**kwargs):
        frame=renderer(model_tid,s2d,cams,*args,**kwargs)
        view='source' if kwargs.get('T_cw') is None and kwargs.get('view_cam_id') is None else None
        if kwargs.get('prefix_text')=='Novel':view='nearby'
        if view is not None:
            h,w=int(cams.default_H),int(cams.default_W)
            if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or frame.shape[0]<h or frame.shape[1]!=3*w:
                raise ValueError('unexpected upstream RGB diagnostic layout')
            output.mkdir(parents=True,exist_ok=True)
            path=output/f'{int(model_tid):06d}_{view}.png'
            if path.exists():raise FileExistsError('refusing to overwrite captured diagnostic')
            Image.fromarray(frame[:h,w:2*w]).save(path)
        return frame
    return render


def install_rgb_capture(video_renderer,output):
    """Patch the function's actual globals, including legacy top-level imports."""
    namespace=video_renderer.__globals__
    namespace['viz2d_one_frame']=capture_rgb_diagnostics(namespace['viz2d_one_frame'],output)


def required_inputs(stage):
    camera=('bundle/bundle_cams.pth','bundle/bundle.pth','tracker_epi.npz')
    if stage=='camera':return ()
    if stage=='scaffold':return camera
    if stage=='photo':return camera+('mosca/mosca.pth','track_identification.npz')
    if stage=='playback':return camera+('photometric_cam.pth','photometric_s_model_native_add3.pth','photometric_d_model_native_add3.pth')
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
    predecessor={'scaffold':'camera','photo':'scaffold','playback':'photo'}[stage]
    report=json.loads((directory/(predecessor+'-report.json')).read_text())
    if (report.get('passed') is not True or report.get('original_inputs_preserved') is not True
        or report.get('input_sha256')!=input_hashes):
        raise ValueError('predecessor failed or belongs to different input priors/tracks')
    for name in required_inputs(stage):
        if digest(directory/name)!=report['output_sha256'].get(name):
            raise ValueError('predecessor checkpoint hash mismatch')


def validate_checkpoint(path):
    """Check saved tensor state without permitting arbitrary pickle execution."""
    import torch
    count=0
    def visit(value):
        nonlocal count
        if isinstance(value,torch.Tensor):
            if not torch.isfinite(value).all():raise ValueError('nonfinite checkpoint tensor')
            count+=value.numel()
        elif isinstance(value,dict):
            for child in value.values():visit(child)
        elif isinstance(value,(tuple,list)):
            for child in value:visit(child)
    visit(torch.load(path,map_location='cpu',weights_only=True))
    if count==0:raise ValueError('empty checkpoint tensor state')
    return count


def verify_fixed_positions(original, fitted):
    """Require exact saved scaffold positions and identities after a frozen-position fit."""
    import torch
    for key in ('_node_xyz','_node_certain','_t_list'):
        if not torch.equal(original[key],fitted['scf.'+key]):
            raise ValueError('fixed-position scaffold changed: '+key)
    return dict(positions_unchanged=True,node_identity_unchanged=True,
                node_count=int(original['_node_xyz'].shape[1]),
                timestamps=int(original['_node_xyz'].shape[0]))


from contextlib import contextmanager

@contextmanager
def finite_loss_guard():
    """Abort on nonfinite backward inputs; restore PyTorch after this local stage."""
    import torch
    original=torch.Tensor.backward
    def checked(tensor,*args,**kwargs):
        if not torch.isfinite(tensor.detach()).all().item():
            raise FloatingPointError('nonfinite training loss; refusing backward')
        return original(tensor,*args,**kwargs)
    torch.Tensor.backward=checked
    try:
        yield
    finally:
        torch.Tensor.backward=original
