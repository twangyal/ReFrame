"""Minimal staged research workflow. Run `python -m reframe.cli --help`."""
import argparse
import json
from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image

from .media import prepare_video, prepare_scene_video, scene_images, validate_manifest, sha256_file


def new_output(path, forbidden=()):
    path=Path(path).resolve()
    if any(path==Path(p).resolve() or Path(p).resolve() in path.parents for p in forbidden):
        raise ValueError('outputs must be outside original reconstruction/source directories')
    path.mkdir(parents=True,exist_ok=False)
    return path


def segment(args):
    from .segmentation import fetch_or_cache,decode_response,validate_import
    from .visualization import save_overlay
    frames=Path(args.frames);m=validate_manifest(frames)
    if not 0 <= args.min_confidence <= 1: raise ValueError('min-confidence must be in [0,1]')
    if args.response:
        response=validate_import(json.loads(Path(args.response).read_text()),m['video_sha256'],
                                 sha256_file(frames/'manifest.json'),args.prompt)
        key='imported-'+sha256_file(args.response)
    else:
        response,key=fetch_or_cache(args.video,m['video_sha256'],args.prompt,args.cache,allow_api=args.allow_api)
    masks,observed,confidence=decode_response(response,args.object_id,m['frame_count'],m['height'],m['width'])
    reliable=observed & np.isfinite(confidence) & (confidence>=args.min_confidence)
    if not observed.any():
        raise ValueError('requested object ID never appears; inspect cached raw IDs or choose another prompt')
    out=new_output(args.out,[frames])
    metadata={'schema':1,'provider':'meta-hosted-sam-3.1','response_origin':'imported' if args.response else 'api',
              'response_id':response.get('id'),'cache_key':key,'sam_object_id':str(args.object_id),
              'prompt':args.prompt,'frame_manifest_sha256':sha256_file(frames/'manifest.json'),
              'min_confidence':args.min_confidence,'missing_confidence':'unreliable'}
    np.savez_compressed(out/'masks.npz',masks=masks,observed=observed,confidence=confidence,
                        reliable=reliable,metadata=json.dumps(metadata))
    (out/'response.json').write_text(json.dumps(response))
    envelope={'provenance':{'model':'sam-3.1','prompt':args.prompt,'video_sha256':m['video_sha256'],
                           'frame_manifest_sha256':metadata['frame_manifest_sha256']},'response':response}
    (out/'export.json').write_text(json.dumps(envelope))
    for i,f in enumerate(m['frames']):
        save_overlay(np.asarray(Image.open(frames/f['file']).convert('RGB')),masks[i],out/f'overlay_{i:06d}.png')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(f'Masks and overlays saved: {out}; target observed in {int(observed.sum())}/{len(observed)} frames')


def probe(args):
    import torch
    from .reconstruction import FrozenMoSca,nearby_camera
    from .visualization import save_rgb
    scene=FrozenMoSca(args.mosca,args.checkpoint)
    out=new_output(args.out,[args.checkpoint,args.mosca])
    torch.cuda.reset_peak_memory_stats()
    views=[]
    for t in sorted({0,scene.frame_count//2,scene.frame_count-1}):
        p=scene.gaussians(t)
        pose=scene.poses[t]
        depth=(p[0]@pose[:3,:3].T+pose[:3,3])[:,2]
        positive=depth[depth>0]
        if not len(positive): raise ValueError('no Gaussians in front of camera')
        shift=float(positive.median())*.03
        for name,cam in [('source',pose),('nearby',nearby_camera(pose,shift))]:
            with torch.no_grad(): r=scene.render(p,t,pose=cam)
            if not (r['alpha']>0).any(): raise ValueError('blank reconstruction render')
            save_rgb(r['rgb'],out/f'{t:06d}_{name}_baseline.png')
            views.append({'frame':t,'view':name,'T_cw':cam.cpu().tolist()})
        e=scene.evidence(t,np.ones((scene.height,scene.width),dtype=bool))
        if not np.allclose(e[:,1],0,atol=1e-5): raise ValueError('all-foreground mask has background contributions')
    report={**scene.provenance(),'completed':True,'kind':'real-mosca-baseline',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'views':views,'color_gradient_check':'all-ones mask, per-row conservation and alpha-sum passed'}
    (out/'baseline.json').write_text(json.dumps(report,indent=2))
    print(f'Baseline renders saved to {out}; inspect both views before object association')


def check_scene_frames(scene_source,frames,manifest):
    """Compare decoded pixels, in exactly the Saved2D load order, before association."""
    source=list(scene_images(scene_source))
    if len(source)!=manifest['frame_count']: raise ValueError('source frame count differs')
    for i,a in enumerate(source):
        b=np.asarray(Image.open(Path(frames)/manifest['frames'][i]['file']).convert('RGB'))
        if not np.array_equal(a,b):
            raise ValueError(f'frame {i} differs from reconstruction source; use a lossless clip of its exact frames')


def associate(args):
    import torch
    from .reconstruction import FrozenMoSca,nearby_camera
    from .association import aggregate_evidence
    from .visualization import save_rgb,selection_parameters
    from .evaluation import mask_metrics
    m=validate_manifest(args.frames)
    check_scene_frames(args.scene_source,args.frames,m)
    with np.load(args.masks,allow_pickle=False) as z:
        masks=z['masks'];observed=z['observed'];confidence=z['confidence'];reliable=z['reliable'];sam=json.loads(str(z['metadata']))
    if sam['frame_manifest_sha256']!=sha256_file(Path(args.frames)/'manifest.json'):
        raise ValueError('masks belong to another frame manifest')
    if masks.shape!=(m['frame_count'],m['height'],m['width']) or observed.shape!=(m['frame_count'],):
        raise ValueError('mask bundle shape mismatch')
    if masks.dtype!=bool or observed.dtype!=bool or reliable.dtype!=bool: raise ValueError('mask bundle must use boolean arrays')
    if confidence.shape!=observed.shape or reliable.shape!=observed.shape:
        raise ValueError('confidence/observation shapes differ')
    if not np.array_equal(reliable,observed & np.isfinite(confidence) & (confidence>=sam['min_confidence'])):
        raise ValueError('mask reliability does not match its recorded policy')
    scene=FrozenMoSca(args.mosca,args.checkpoint)
    if (scene.frame_count,scene.height,scene.width)!=masks.shape:
        raise ValueError('scene and mask frame dimensions/counts differ')
    baseline=json.loads(Path(args.baseline).read_text())
    if (baseline.get('completed') is not True or baseline.get('kind')!='real-mosca-baseline'
        or baseline.get('backend')!='native_add3' or baseline.get('mosca_revision')!=scene.provenance()['mosca_revision']
        or baseline.get('checkpoint_sha256')!=scene.fingerprints):
        raise ValueError('a completed baseline for these exact checkpoints is required')
    heldout=sorted({0,scene.frame_count//2,scene.frame_count-1})
    # Cap at 12 source frames for a first inexpensive association experiment.
    available=[i for i in range(scene.frame_count) if reliable[i] and i not in heldout]
    if len(available)<2: raise ValueError('need at least two observed training frames outside the three evaluation frames')
    train=[available[i] for i in np.unique(np.linspace(0,len(available)-1,min(12,len(available))).astype(int))]
    evidence=np.stack([scene.evidence(t,masks[t]) for t in train])
    aggregation={'foreground_threshold':args.threshold,'background_threshold':.2,
                 'min_visible_frames':2,'min_frame_mass':args.min_frame_mass,'max_frame_weight':32.,'max_variance':.06}
    stats=aggregate_evidence(evidence,**aggregation)
    selected=stats['state']=='foreground'
    out=new_output(args.out,[args.checkpoint,args.mosca,args.scene_source,args.frames])
    identity=json.dumps({'checkpoint':scene.fingerprints,'sam_cache':sam['cache_key'],'sam_object_id':sam['sam_object_id']},sort_keys=True)
    meta={**scene.provenance(),'object_id':str(uuid.uuid5(uuid.NAMESPACE_URL,identity)),
          'segmentation':sam,'train_frames':train,'heldout_frames':heldout,
          'aggregation':aggregation,'score_kind':'visible foreground evidence fraction',
          'selected_static_count':int(selected[:scene.static_count].sum()),
          'selected_dynamic_count':int(selected[scene.static_count:].sum()),
          'novel_view_review':'pending human/visual inspection','real_selection_success':False}
    np.savez_compressed(out/'selection.npz',**stats,evidence=evidence,
                        static_indices=np.flatnonzero(selected[:scene.static_count]),
                        dynamic_indices=np.flatnonzero(selected[scene.static_count:]),metadata=json.dumps(meta))
    if not selected.any():
        meta['status']='no_confident_gaussians; evidence retained for diagnosis'
        (out/'report.json').write_text(json.dumps(meta,indent=2))
        raise ValueError('no confident Gaussians; saved evidence for threshold/visibility diagnosis')
    reports=[]
    for t in heldout:
        p=scene.gaussians(t);pose=scene.poses[t]
        depth=(p[0]@pose[:3,:3].T+pose[:3,3])[:,2]
        shift=float(depth[depth>0].median())*.03
        colors=torch.as_tensor(selected,device=scene.device,dtype=torch.float32)[:,None].repeat(1,3)
        for view,cam in [('source',pose),('nearby',nearby_camera(pose,shift))]:
            with torch.no_grad():
                for mode,params in [('baseline',p),('highlight',selection_parameters(p,selected,'highlight')),
                                    ('isolated',selection_parameters(p,selected,'isolated'))]:
                    save_rgb(scene.render(params,t,pose=cam)['rgb'],out/f'{t:06d}_{view}_{mode}.png')
                visible=scene.render(p,t,pose=cam,colors=colors)['rgb'][0].cpu().numpy()
            Image.fromarray((np.clip(visible,0,1)*255).round().astype(np.uint8)).save(out/f'{t:06d}_{view}_visible_mask.png')
            if view=='source':
                row={'frame':t,'target_observed':bool(observed[t]),'reliable':bool(reliable[t]),'mask_threshold':.5}
                if reliable[t]:
                    row.update(mask_metrics(visible>=.5,masks[t]))
                    e=scene.evidence(t,masks[t])
                    for pool,sl in [('static',slice(0,scene.static_count)),('dynamic',slice(scene.static_count,None))]:
                        masses=e[sl][selected[sl]].sum(axis=0)
                        row[pool+'_visible_contamination']=float(masses[1]/masses[2]) if masses[2]>0 else None
                    diff=np.stack([visible>=.5,masks[t],np.logical_xor(visible>=.5,masks[t])],axis=-1)
                    Image.fromarray(diff.astype(np.uint8)*255).save(out/f'{t:06d}_mask_comparison.png')
                reports.append(row)
    meta['evaluation']=reports
    meta['uncertain_gaussians']=int(np.isin(stats['state'],['unknown','ambiguous']).sum())
    meta['selected_temporal_evidence_variance_mean']=float(stats['variance'][selected].mean())
    meta['limitation']='fixed row identity is by construction; geometric coherence and SAM ID continuity require visual review'
    (out/'report.json').write_text(json.dumps(meta,indent=2,allow_nan=False))
    print(f'Candidate object and diagnostics saved to {out}; success remains unconfirmed until visual review')


def render_saved(args):
    from .selection import load_selection
    from .selection_render import render_selection
    from .reconstruction import FrozenMoSca
    selection=load_selection(args.selection)
    m=validate_manifest(args.frames)
    check_scene_frames(args.scene_source,args.frames,m)
    with np.load(args.masks,allow_pickle=False) as z:
        masks=z['masks'];reliable=z['reliable'];observed=z['observed'];confidence=z['confidence']
        sam=json.loads(str(z['metadata']))
    if (sam!=selection['metadata']['segmentation']
        or sam['frame_manifest_sha256']!=sha256_file(Path(args.frames)/'manifest.json')):
        raise ValueError('saved selection and cached masks belong to different tracks/frames')
    if (observed.dtype!=bool or reliable.dtype!=bool or observed.shape!=(m['frame_count'],)
        or confidence.shape!=observed.shape or reliable.shape!=observed.shape
        or not np.array_equal(reliable,observed & np.isfinite(confidence) & (confidence>=sam['min_confidence']))):
        raise ValueError('invalid cached mask reliability policy')
    out=Path(args.out).resolve()
    for source in (args.checkpoint,args.mosca,args.scene_source,args.frames,str(Path(args.selection).parent)):
        source=Path(source).resolve()
        if out==source or source in out.parents:raise ValueError('render outputs must stay outside source directories')
    if out.exists():raise FileExistsError(out)
    scene=FrozenMoSca(args.mosca,args.checkpoint)
    report=render_selection(scene,selection,masks,reliable,out,
                            times=list(range(scene.frame_count)) if args.all_frames else None)
    # Preserve the exact compact selection used to render, for report/output binding.
    import shutil
    shutil.copyfile(args.selection,out/'selection.npz')
    print(f'Saved selection rendered at {len(report["rendered_frames"])} times; visual review remains pending: {out}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    prep=commands.add_parser('prepare-video');prep.add_argument('video');prep.add_argument('--out',required=True)
    prep_scene=commands.add_parser('prepare-scene-video')
    prep_scene.add_argument('--scene-source',required=True);prep_scene.add_argument('--fps',type=float,default=30.)
    prep_scene.add_argument('--out',required=True)
    seg=commands.add_parser('segment')
    seg.add_argument('--video');seg.add_argument('--frames',required=True);seg.add_argument('--prompt',required=True)
    seg.add_argument('--object-id',required=True);seg.add_argument('--cache',default='.cache/sam')
    seg.add_argument('--response',help='import export.json with completed API response and source provenance')
    seg.add_argument('--min-confidence',type=float,default=.5)
    seg.add_argument('--allow-api',action='store_true',help='explicitly authorize upload and billable inference on cache miss')
    seg.add_argument('--out',required=True)
    revise=commands.add_parser('revise-selection',help='CPU-only revision from saved contribution evidence')
    revise.add_argument('--selection',required=True)
    revise.add_argument('--threshold',type=float)
    revise.add_argument('--min-frame-mass',type=float)
    revise.add_argument('--exclusions',help='JSON list of explicit pool/index/reason records')
    revise.add_argument('--out',required=True)
    render=commands.add_parser('render-selection',help='GPU render of saved rows; no SAM or training association')
    for name in ('mosca','checkpoint','scene-source','frames','masks','selection','out'):
        render.add_argument('--'+name,required=True)
    render.add_argument('--all-frames',action='store_true',help='render every discrete reconstruction frame')
    for name in ['probe-mosca','associate']:
        q=commands.add_parser(name)
        q.add_argument('--mosca',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--out',required=True)
        if name=='associate':
            q.add_argument('--frames',required=True);q.add_argument('--scene-source',required=True)
            q.add_argument('--masks',required=True);q.add_argument('--baseline',required=True)
            q.add_argument('--threshold',type=float,default=.8)
            q.add_argument('--min-frame-mass',type=float,default=1.,help='minimum visible alpha-weighted pixel mass per observation')
    a=p.parse_args()
    try:
        if a.command=='prepare-video': print(json.dumps(prepare_video(a.video,a.out),indent=2))
        elif a.command=='prepare-scene-video': print(json.dumps(prepare_scene_video(a.scene_source,a.out,fps=a.fps),indent=2))
        elif a.command=='segment':
            if not a.response and not a.video: p.error('segment requires --video or --response')
            segment(a)
        elif a.command=='probe-mosca': probe(a)
        elif a.command=='render-selection': render_saved(a)
        elif a.command=='revise-selection':
            from .selection import revise_selection
            report=revise_selection(a.selection,a.out,threshold=a.threshold,min_frame_mass=a.min_frame_mass,
                                    exclusions=json.loads(Path(a.exclusions).read_text()) if a.exclusions else None)
            print(json.dumps(report,indent=2,allow_nan=False))
        else: associate(a)
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as e:
        p.exit(2,f'{e}\n')


if __name__=='__main__':main()
