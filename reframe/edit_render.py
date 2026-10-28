"""Render one frozen object identity and its edits at every discrete scene time."""
import json
from pathlib import Path
import numpy as np
from .editing import edit_parameters,validate_edit_recipe
from .evaluation import mask_metrics
from .media import sha256_file
from .reconstruction import nearby_camera
from .selection import validate_scene_binding
from .selection_render import selected_mass_metrics
from .visualization import save_rgb,selection_parameters

MODES=('baseline','highlight','isolated','recolored','translated')
VIEWS=('source','nearby')


def render_edits(scene,selection,masks,reliable,recipe,out):
    import torch
    validate_scene_binding(selection,scene.provenance());validate_edit_recipe(recipe,selection)
    selected=selection['selected'];meta=selection['metadata']
    if not selected.any():raise ValueError('empty selected object')
    if (masks.dtype!=bool or masks.shape!=(scene.frame_count,scene.height,scene.width)
        or reliable.dtype!=bool or reliable.shape!=(scene.frame_count,)):
        raise ValueError('cached masks differ from scene')
    first=scene.gaussians(0)
    points=first[0][torch.as_tensor(selected,device=scene.device)]
    extent=torch.linalg.vector_norm(torch.quantile(points,.95,dim=0)-torch.quantile(points,.05,dim=0))
    if not torch.isfinite(extent) or float(extent)<=0:raise ValueError('selected object has no usable spatial extent')
    # Camera-right at t=0 defines a fixed WORLD direction, never recomputed per frame.
    direction=torch.linalg.inv(scene.poses[0])[:3,0]
    offset=direction/torch.linalg.vector_norm(direction)*extent*recipe['translation_extent_fraction']
    offset=offset.detach().cpu().tolist()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={**meta,'kind':'gaussian-object-edits','selection_sha256':selection['sha256'],
            'recipe':recipe,'translation_world':offset,'translation_units':'reconstruction units, not calibrated meters',
            'selected_extent_05_95':float(extent),'translation_direction':'fixed first-camera right in world space',
            'rendered_frames':list(range(scene.frame_count)),'modes':list(MODES),'views':list(VIEWS),
            'status':'rendering','phase_two_complete':False,'visual_review':'pending',
            'evaluation_is_exploratory':True,'sam_called':False,'original_models_modified':False,
            'maximum_translation_residual':0.,'evaluation':[],'files':{},
            'limitations':['discrete reconstruction times only; no interpolation',
                           'constant RGB recolor removes directional shading',
                           'translation reveals missing background; no inpainting or shadow correction',
                           'source SAM is not ground truth for the translated object or novel views',
                           'fixed row identity alone does not establish geometric coherence']}
    def write():
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write()
    colors=torch.as_tensor(selected,device=scene.device,dtype=torch.float32)[:,None].repeat(1,3)
    for t in report['rendered_frames']:
        with torch.no_grad():
            params=scene.gaussians(t);pose=scene.poses[t]
            translated=edit_parameters(params,selected,offset=offset)
            delta=translated[0]-params[0]
            expected=torch.as_tensor(offset,device=scene.device,dtype=params[0].dtype)*colors[:,:1]
            residual=float((delta-expected).abs().max())
            if not torch.allclose(delta,expected,rtol=1e-5,atol=1e-5):raise ValueError('translation changed row motion or unselected positions')
            report['maximum_translation_residual']=max(report['maximum_translation_residual'],residual)
            variants={'baseline':params,'highlight':selection_parameters(params,selected,'highlight'),
                      'isolated':selection_parameters(params,selected,'isolated'),
                      'recolored':edit_parameters(params,selected,color=recipe['color_rgb']),
                      'translated':translated}
            depth=(params[0]@pose[:3,:3].T+pose[:3,3])[:,2];positive=depth[depth>0]
            if not len(positive):raise ValueError('no positive scene depth')
            cameras={'source':pose,'nearby':nearby_camera(pose,float(positive.median())*.03)}
            source={}
            for view,cam in cameras.items():
                for mode,p in variants.items():
                    rgb=scene.render(p,t,pose=cam)['rgb']
                    path=out/f'{t:06d}_{view}_{mode}.png';save_rgb(rgb,path)
                    report['files'][path.name]=sha256_file(path)
                    if view=='source' and mode in ('baseline','recolored'):source[mode]=rgb.detach().cpu().numpy()
            visible=scene.render(params,t,colors=colors)['rgb'][0].cpu().numpy()
        if not np.isfinite(visible).all():raise ValueError('nonfinite visible selection mask')
        row={'frame':t,'reliable':bool(reliable[t]),'selected_static_count':meta['selected_static_count'],
             'selected_dynamic_count':meta['selected_dynamic_count'],
             'aggregation_role':'training' if t in meta['train_frames'] else 'excluded_from_aggregation'}
        if reliable[t]:
            row.update(mask_metrics(visible>=.5,masks[t]))
            row['visible_mass']=selected_mass_metrics(scene.evidence(t,masks[t]),selected,scene.static_count)
            change=np.abs(source['recolored']-source['baseline']).mean(axis=0).astype(np.float64)
            total=float(change.sum());outside=float(change[~masks[t]].sum())
            row['recolor_change']={'total_absolute':total,'outside_sam_absolute':outside,
                                  'outside_sam_fraction':outside/total if total>0 else None,
                                  'meaning':'pixel RGB change, not a geometric contamination verdict'}
        report['evaluation'].append(row);write()
    report.update(status='rendered_pending_visual_review',completed=True);write()
    return report
