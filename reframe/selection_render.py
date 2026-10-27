"""Render fixed saved membership through MoSca deformation; no reassociation."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .evaluation import mask_metrics
from .selection import validate_scene_binding
from .visualization import save_rgb,selection_parameters


def selected_mass_metrics(evidence,selected,static_count):
    e=np.asarray(evidence);selected=np.asarray(selected)
    if (e.ndim!=2 or e.shape[1]!=3 or selected.dtype!=bool or selected.shape!=(len(e),)
        or type(static_count) is not int or not 0<=static_count<=len(e)
        or not np.isfinite(e).all() or (e<0).any()
        or not np.allclose(e[:,0]+e[:,1],e[:,2],rtol=2e-3,atol=2e-4)):
        raise ValueError('invalid per-row visibility evidence')
    result={}
    for pool,sl in [('static',slice(0,static_count)),('dynamic',slice(static_count,None))]:
        masses=e[sl][selected[sl]].sum(axis=0,dtype=np.float64)
        result[pool]={'inside':float(masses[0]),'outside':float(masses[1]),'total':float(masses[2]),
                      'outside_fraction':float(masses[1]/masses[2]) if masses[2]>0 else None}
    return result


def render_selection(scene,selection,masks,reliable,out,*,times=None):
    """Use all original occluders for visible masks; isolate only for inspection."""
    import torch
    from .reconstruction import nearby_camera
    validate_scene_binding(selection,scene.provenance())
    selected=selection['selected'];meta=selection['metadata']
    if not selected.any():raise ValueError('selection contains no rows to render')
    masks=np.asarray(masks);reliable=np.asarray(reliable)
    if (masks.dtype!=bool or masks.shape!=(scene.frame_count,scene.height,scene.width)
        or reliable.dtype!=bool or reliable.shape!=(scene.frame_count,)):
        raise ValueError('cached mask dimensions or reliability differ from scene')
    times=sorted({0,scene.frame_count//2,scene.frame_count-1}) if times is None else list(times)
    if not times or len(set(times))!=len(times) or any(type(t) is not int or not 0<=t<scene.frame_count for t in times):
        raise ValueError('invalid requested render timestamps')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={**meta,'kind':'saved-selection-render','selection_sha256':selection['sha256'],
            'status':'rendering','evaluation':[],'rendered_frames':times,
            'real_selection_success':False,'novel_view_review':'pending inspection of this exact revision',
            'evaluation_is_exploratory':True,'sam_called':False,'training_evidence_recomputed':False}
    # Preserve an explicit incomplete status if rendering fails.
    report_path=out/'report.json';report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    colors=torch.as_tensor(selected,device=scene.device,dtype=torch.float32)[:,None].repeat(1,3)
    for t in times:
        p=scene.gaussians(t);pose=scene.poses[t]
        depth=(p[0]@pose[:3,:3].T+pose[:3,3])[:,2];positive=depth[depth>0]
        if not len(positive):raise ValueError('no scene geometry in front of the camera')
        shift=float(positive.median())*.03
        for view,cam in [('source',pose),('nearby',nearby_camera(pose,shift))]:
            with torch.no_grad():
                for mode,params in [('baseline',p),('highlight',selection_parameters(p,selected,'highlight')),
                                    ('isolated',selection_parameters(p,selected,'isolated'))]:
                    save_rgb(scene.render(params,t,pose=cam)['rgb'],out/f'{t:06d}_{view}_{mode}.png')
                visible=scene.render(p,t,pose=cam,colors=colors)['rgb'][0].cpu().numpy()
            if not np.isfinite(visible).all():raise ValueError('nonfinite visible selection mask')
            Image.fromarray((np.clip(visible,0,1)*255).round().astype(np.uint8)).save(out/f'{t:06d}_{view}_visible_mask.png')
            if view=='source':
                row={'frame':t,'reliable':bool(reliable[t]),'mask_threshold':.5,
                     'aggregation_role':'training' if t in meta['train_frames'] else 'excluded_from_aggregation'}
                if reliable[t]:
                    row.update(mask_metrics(visible>=.5,masks[t]))
                    row['visible_mass']=selected_mass_metrics(scene.evidence(t,masks[t]),selected,scene.static_count)
                    diff=np.stack([visible>=.5,masks[t],np.logical_xor(visible>=.5,masks[t])],axis=-1)
                    Image.fromarray(diff.astype(np.uint8)*255).save(out/f'{t:06d}_mask_comparison.png')
                report['evaluation'].append(row)
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report.update(status='rendered_pending_visual_review',completed=True)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
