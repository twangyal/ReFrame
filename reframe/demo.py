"""Build a private offline evidence bundle; never runs models or calls services."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from .media import sha256_file,validate_manifest
from .selection import load_selection,validate_scene_binding
from .visualization import save_overlay


def validate_inputs(config,root):
    root=Path(root).resolve();pins=config['sha256']
    def check(path):
        path=Path(path).resolve()
        if not path.is_relative_to(root):raise ValueError('demo inputs must be inside the project root')
        key=str(path.relative_to(root))
        if pins.get(key)!=sha256_file(path):raise ValueError(f'unpinned or changed demo input: {key}')
        return path
    paths={key:root/config[key] for key in ('prepared','sam','selection_dir','initial_report','visual_review')}
    frames=paths['prepared']/'frames';check(frames/'manifest.json');manifest=validate_manifest(frames)
    for row in manifest['frames']:check(frames/row['file'])
    selection=load_selection(check(paths['selection_dir']/'selection.npz'))
    meta=selection['metadata'];report=json.loads(check(paths['selection_dir']/'report.json').read_text())
    validate_scene_binding(selection,report)
    for key in ('object_id','segmentation','aggregation','train_frames','heldout_frames','selected_static_count','selected_dynamic_count'):
        if meta.get(key)!=report.get(key):raise ValueError(f'render report differs from selection: {key}')
    if meta.get('status')=='unrendered_exploratory_candidate' and not (
        report.get('completed') is True and report.get('kind')=='saved-selection-render'
        and report.get('selection_sha256')==selection['sha256']):
        raise ValueError('an unrendered candidate cannot supply measured demo renders')
    if report.get('selection_sha256'):
        if report['selection_sha256']!=selection['sha256']:raise ValueError('render report selection digest differs')
    else:
        legacy=config.get('legacy_render_binding',{})
        if ('revision_id' in meta or meta.get('exclusions') or 'evidence' not in selection
            or legacy.get('selection_sha256')!=selection['sha256']
            or legacy.get('report_sha256')!=sha256_file(paths['selection_dir']/'report.json')):
            raise ValueError('legacy renders require explicit unchanged original-evidence binding')
    if (meta['frames'],meta['height'],meta['width'])!=(manifest['frame_count'],manifest['height'],manifest['width']):
        raise ValueError('selection/source dimensions differ')
    with np.load(check(paths['sam']/'masks.npz'),allow_pickle=False) as z:
        masks=z['masks'];sam=json.loads(str(z['metadata']))
    if sam!=meta['segmentation'] or sam['frame_manifest_sha256']!=sha256_file(frames/'manifest.json'):
        raise ValueError('SAM track or frame manifest differs from selection')
    if masks.dtype!=bool or masks.shape!=(meta['frames'],meta['height'],meta['width']):raise ValueError('invalid demo masks')
    times=sorted({0,meta['frames']//2,meta['frames']-1})
    evaluated=[r['frame'] for r in report['evaluation']]
    if not set(times).issubset(evaluated) or len(set(evaluated))!=len(evaluated):
        raise ValueError('missing or duplicate evaluation frames')
    for t in times:
        for view in ('source','nearby'):
            for mode in ('baseline','highlight','isolated'):
                p=check(paths['selection_dir']/f'{t:06d}_{view}_{mode}.png')
                with Image.open(p) as image:
                    if image.size!=(meta['width'],meta['height']):raise ValueError('render dimensions differ')
        check(paths['selection_dir']/f'{t:06d}_mask_comparison.png')
    initial=json.loads(check(paths['initial_report']).read_text());validate_scene_binding(selection,initial)
    if initial.get('object_id')!=meta['object_id']:raise ValueError('initial report belongs to another object')
    review=json.loads(check(paths['visual_review']).read_text())
    if (review.get('object_id')!=meta['object_id'] or review.get('selected_dynamic')!=meta['selected_dynamic_count']
        or review.get('selected_static')!=meta['selected_static_count']):raise ValueError('visual review belongs to another selection')
    candidate=None
    if config.get('candidate_dir'):
        folder=root/config['candidate_dir'];candidate=json.loads(check(folder/'revision.json').read_text())
        compact_path=check(folder/'render-selection.npz');compact=load_selection(compact_path)
        validate_scene_binding(compact,meta)
        if (candidate.get('parent_selection_sha256')!=selection['sha256']
            or candidate.get('render_selection_sha256')!=sha256_file(compact_path)
            or candidate.get('object_id')!=meta['object_id']
            or candidate.get('revision_id')!=compact['metadata'].get('revision_id')
            or candidate.get('status')!='unrendered_exploratory_candidate'
            or candidate.get('real_selection_success') is not False or 'evaluation' in candidate):
            raise ValueError('cleanup candidate provenance/status differs')
    return {'paths':paths,'manifest':manifest,'selection':selection,'report':report,'initial':initial,
            'review':review,'candidate':candidate,'masks':masks,'times':times}


def build_demo(config,root,out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    data=validate_inputs(config,root);m=data['manifest'];paths=data['paths']
    if (m['frame_count'],m['width'],m['height'])!=(12,480,270):
        raise ValueError('current preview layout is explicitly scoped to the inspected Balloon2 sample')
    root=Path(root).resolve()
    for p in paths.values():
        p=p.resolve()
        if out==p or p in out.parents:raise ValueError('demo output must stay outside source directories')
    out.mkdir(parents=True);overlays=out/'overlays';overlays.mkdir()
    for i,row in enumerate(m['frames']):
        with Image.open(paths['prepared']/'frames'/row['file']) as image:
            save_overlay(np.asarray(image.convert('RGB')),data['masks'][i],overlays/f'overlay_{i:06d}.png')
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    subprocess.run([sys.executable,str(scripts/'make_sample_previews.py'),'--prepared',str(paths['prepared']),
                    '--baseline',str(paths['selection_dir']),'--sam',str(overlays),'--out',str(out/'previews'),
                    '--sam-object-id',data['selection']['metadata']['segmentation']['sam_object_id']],check=True)
    subprocess.run([sys.executable,str(scripts/'make_selection_preview.py'),'--selection',str(paths['selection_dir']),
                    '--out',str(out/'selection-preview')],check=True,stdout=subprocess.DEVNULL)
    evidence=out/'evidence';evidence.mkdir()
    for name,source in [('initial-report.json',paths['initial_report']),('refined-report.json',paths['selection_dir']/'report.json'),
                        ('visual-review.json',paths['visual_review'])]:shutil.copyfile(source,evidence/name)
    for t in data['times']:shutil.copyfile(paths['selection_dir']/f'{t:06d}_mask_comparison.png',evidence/f'{t:06d}_mask_comparison.png')
    if data['candidate']:
        folder=root/config['candidate_dir']
        for name in ('revision.json','render-selection.npz'):shutil.copyfile(folder/name,evidence/('candidate-'+name))
    summary={'kind':'private-offline-evidence-demo','models_run':False,'network_calls':False,
             'measured_selection_sha256':data['selection']['sha256'],'object_id':data['report']['object_id'],
             'initial_evaluation':data['initial']['evaluation'],'exploratory_refined_evaluation':data['report']['evaluation'],
             'cleanup_candidate':data['candidate'],'candidate_renders_included':False,
             'geometry_verified':False,'continuous_reconstruction_motion_included':False,
             'limitations':['source-mask IoU is not geometric correctness','nearby views only',
                            'full occlusion/re-entry untested','refinement used evaluation feedback',
                            'cleanup candidate has not been rendered','private assets; publication rights unresolved']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    videos=[('Source: chosen 3 fps playback','previews/balloon2-source-preview.mp4'),
            ('Actual hosted SAM 3.1 masks','previews/balloon2-sam-overlay.mp4'),
            ('MoSca baseline: three held timestamps','previews/balloon2-reconstruction-sampled-preview.mp4'),
            ('Measured selection: held source/nearby renders, not the cleanup candidate','selection-preview/balloon2-selection-sampled-comparison.mp4')]
    page=['<!doctype html><meta charset="utf-8"><title>ReFrame private experiment evidence</title>',
          '<h1>ReFrame — private experiment evidence</h1>',
          '<p>One published scene. This is a local evidence bundle, not a complete hackathon submission.</p>',
          '<p>Refined results are exploratory. Cleanup candidate is unrendered. No translation or smooth 4D animation is claimed.</p>']
    for label,rel in videos:
        page.append(f'<h2>{html.escape(label)}</h2><video controls preload="metadata" width="960" style="max-width:100%" src="{rel}"></video>')
    page.append('<h2>Evidence</h2><p><a href="summary.json">Measured results and limitations</a></p>')
    for t in data['times']:page.append(f'<p>Source frame {t}: predicted red, SAM green, disagreement blue</p><img src="evidence/{t:06d}_mask_comparison.png" alt="mask comparison frame {t}">')
    (out/'index.html').write_text('\n'.join(page)+'\n')
    (out/'README.md').write_text('# Private ReFrame demo bundle\n\nOpen index.html locally. No server, network or model inference is required.\nVideos are independently playable. The selected reconstruction holds three\nrendered timestamps; source and SAM clips play all twelve supplied frames.\nThe cleanup candidate is a separate unrendered ablation and has no new IoU.\nPublic redistribution rights for benchmark assets remain unresolved.\n\nThis is not the final three-minute narrated hackathon video or a desktop editor.\n')
    output_files={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha256_file(p)}
                  for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps({'schema':1,'private_only':True,'input_recipe':config,
                                              'outputs':output_files},indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try:
        summary=build_demo(json.loads(args.config.read_text()),args.root,args.out)
    except (ValueError,FileNotFoundError,FileExistsError) as e:p.exit(2,f'{e}\n')
    print(f'Private demo assembled: {args.out}; cleanup candidate rendered: {summary["candidate_renders_included"]}')


if __name__=='__main__':main()
