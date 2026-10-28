"""Immutable edits of evaluated MoSca rows; no training or checkpoint writes."""
import numpy as np


def _vector(value,name,*,unit=False):
    a=np.asarray(value,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all() or (unit and ((a<0)|(a>1)).any()):
        raise ValueError(f'invalid {name}')
    return a


def edit_parameters(params,selected,*,color=None,offset=None):
    """Translate after deformation in world units, or set constant RGB via SH DC.

    Higher SH bands are zeroed on recolored rows, deliberately removing their
    directional shading. Other rows, geometry, opacity and row order are kept.
    """
    import torch
    selected=np.asarray(selected)
    if (len(params)!=5 or selected.dtype!=bool or selected.shape!=(len(params[0]),)
        or not selected.any()):raise ValueError('invalid or empty selected rows')
    n=len(selected)
    if (params[0].shape!=(n,3) or params[1].shape!=(n,3,3) or params[2].shape!=(n,3)
        or params[3].shape!=(n,1) or params[4].ndim!=2 or params[4].shape[0]!=n
        or params[4].shape[1] not in (3,12,27,48)
        or any(not torch.isfinite(p).all() for p in params)):
        raise ValueError('invalid Gaussian parameter tuple')
    mask=torch.as_tensor(selected,device=params[0].device)
    result=list(params)
    if offset is not None:
        delta=torch.as_tensor(_vector(offset,'world offset'),device=params[0].device,dtype=params[0].dtype)
        result[0]=params[0].clone();result[0][mask]+=delta
    if color is not None:
        rgb=torch.as_tensor(_vector(color,'RGB color',unit=True),device=params[4].device,dtype=params[4].dtype)
        result[4]=params[4].clone();result[4][mask]=0
        result[4][mask,:3]=(rgb-.5)/.28209479177387814
    return tuple(result)


def validate_edit_recipe(recipe,selection):
    required={'schema','object_id','selection_sha256','color_rgb','translation_extent_fraction'}
    if (set(recipe)!=required or type(recipe['schema']) is not int or recipe['schema']!=1
        or recipe['object_id']!=selection['metadata']['object_id']
        or recipe['selection_sha256']!=selection['sha256']):
        raise ValueError('edit recipe differs from saved selection or supported schema')
    _vector(recipe['color_rgb'],'RGB color',unit=True)
    fraction=recipe['translation_extent_fraction']
    if isinstance(fraction,bool) or not isinstance(fraction,(float,int)) or not np.isfinite(fraction) or not 0<fraction<=.5:
        raise ValueError('translation must be within (0, .5] of selected first-frame extent')
    return recipe


def identity_summary(selection):
    """Summarize frozen evidence, not a calibrated probability or geometry proof."""
    selected=selection['selected'];meta=selection['metadata'];n=len(selected)
    for key in ('score','variance','visible_frames'):
        if key not in selection or selection[key].shape!=(n,):raise ValueError('full confidence evidence required')
    if not selected.any():raise ValueError('empty selected object')
    scores=selection['score'][selected];variance=selection['variance'][selected]
    visible=selection['visible_frames'][selected]
    if not np.isfinite(scores).all() or not np.isfinite(variance).all():raise ValueError('nonfinite membership evidence')
    return {'kind':'persistent-gaussian-identity','object_id':meta['object_id'],
            'selection_sha256':selection['sha256'],'checkpoint_sha256':meta['checkpoint_sha256'],
            'membership_sha256':meta.get('membership_sha256'),
            'selected_static':int(selected[:meta['static_count']].sum()),
            'selected_dynamic':int(selected[meta['static_count']:].sum()),
            'reconstruction_frames':meta['frames'],'aggregation':meta['aggregation'],
            'confidence':{'kind':'visible foreground evidence fraction; not calibrated probability',
                          'minimum':float(scores.min()),'median':float(np.median(scores)),
                          'maximum':float(scores.max()),'mean_temporal_variance':float(variance.mean()),
                          'minimum_visible_training_frames':int(visible.min())},
            'geometric_stability_verified':False,
            'limitation':'Fixed pool-local rows persist by construction. Visibility, leakage and coherence need rendered evaluation.'}
