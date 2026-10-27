#!/usr/bin/env python3
"""Print a local plan by default. Remote execution requires verified gate evidence.

No credentials are created. This gate records an operator's verification; it
cannot itself verify credit balances or enforce Modal billing limits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'pytorch/pytorch@sha256:558b78b9a624969d54af2f13bf03fbad27907dbb6f09973ef4415d6ea24c80d9'
PLAN = {
    'executed': False, 'gpu': 'A100-40GB', 'gpu_count': 1,
    'routing_region': 'us-east',
    'expected_input_plane_host': 'input-plane.us-east.modal.com',
    'cpu_cores_limit': 2, 'memory_mib_limit': 16384, 'ephemeral_disk_mib': None,
    'ephemeral_disk_policy': 'provider default; disk-related memory billing bounded separately',
    'billed_memory_gib_bound': 41.6,  # 16 requested + conservative 512 GiB disk / 20
    'input_timeout_seconds': 900, 'startup_timeout_seconds': 120,
    'max_containers': 1, 'retries': 0, 'min_containers': 0,
    'assumed_rate_ceiling_gpu_usd_hour': 4.0,
    'assumed_rate_ceiling_cpu_usd_core_hour': .10,
    'assumed_rate_ceiling_ram_usd_gib_hour': .02,
    'image_build_storage_transfer_overhead_allowance_usd': 2.0,
    'estimated_ceiling_usd': round((900+120+2)/3600*(4+2*.10+41.6*.02)+2, 4),
    'first_run_budget_usd': 5, 'total_authorization_existing_credit_usd': 30,
    'image': IMAGE, 'modal_sdk_version': '1.6.2',
    'warning': 'Estimate, not a billing cap. Live account/rate/control verification required.',
}


def validate_approval(value):
    for key in ('no_overage_verified', 'rates_within_plan_verified',
                'overhead_within_plan_verified', 'secure_auth_verified',
                'network_access_verified', 'no_concurrent_spend_verified',
                'ephemeral_disk_cost_bound_verified'):
        if value.get(key) is not True:
            raise ValueError(f'approval requires {key}=true from actual operator verification')
    balance = float(value.get('balance_usd', 0))
    if not math.isfinite(balance) or balance < PLAN['first_run_budget_usd']:
        raise ValueError('insufficient verified existing credit for first-run reservation')
    prior = float(value.get('prior_trial_usage_usd', float('nan')))
    pending = float(value.get('prior_billing_uncertainty_allowance_usd', float('nan')))
    if (not math.isfinite(prior) or not math.isfinite(pending) or min(prior,pending) < 0
            or prior + pending + PLAN['estimated_ceiling_usd'] > PLAN['first_run_budget_usd']):
        raise ValueError('cumulative trial estimate exceeds $5 or prior billing evidence is missing')
    when = datetime.fromisoformat(value['verified_at'])
    if when.tzinfo is None or not 0 <= (datetime.now(timezone.utc)-when).total_seconds() <= 3600:
        raise ValueError('approval must be timezone-aware and verified within the past hour')
    if not value.get('workspace') or not value.get('evidence'):
        raise ValueError('approval needs workspace and non-secret billing/control evidence reference')


def claim_approval(value, directory):
    """One local launch per canonical approval record, including uncertain failures."""
    digest = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True,exist_ok=True)
    claim = directory/(digest+'.json')
    with claim.open('x') as handle:
        json.dump({'approval_sha256':digest,'claimed_at':datetime.now(timezone.utc).isoformat(),
                   'state':'claimed; inspect remote job/billing before any retry'},handle)
    return claim


def verify_workspace(actual, approval):
    if actual != approval['workspace']:
        raise ValueError('active Modal workspace differs from verified approval; refusing upload/build')


def stage_sources(mosca, destination, real_scene=False):
    """Export tracked pinned sources only; never upload the repository or secrets."""
    import io
    import tarfile
    revision = json.loads((ROOT/'upstream-revisions.json').read_text())['mosca']['commit']
    actual = subprocess.check_output(['git','-C',str(mosca),'rev-parse','HEAD'],text=True).strip()
    if actual != revision:
        raise ValueError('MoSca revision differs from upstream-revisions.json')
    source_paths = ['lib_render/render_helper.py','lib_render/gauspl_renderer_native_add3.py',
                    'lib_render/sh_utils.py','lib_render/diff-gaussian-rasterization-alphadep-add3']
    archive = subprocess.check_output(['git','-C',str(mosca),'archive',revision,*source_paths])
    # git archive excludes untracked files and never includes .git credentials.
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('unexpected source archive member')
            out = destination/'mosca'/path
            if member.isdir():
                out.mkdir(parents=True,exist_ok=True)
            else:
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_bytes(tar.extractfile(member).read())
    for relative in ('reframe/__init__.py','reframe/association.py','tests/test_cuda_renderer.py',
                     'scripts/modal_smoke_worker.py'):
        out = destination/relative
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes((ROOT/relative).read_bytes())
    if real_scene:
        relatives = [str(p.relative_to(ROOT)) for p in (ROOT/'reframe').glob('*.py')]
        relatives += ['scripts/modal_scene_worker.py','scripts/fetch_mosca_sample.py',
                      'scripts/artifact_stream.py','configs/balloon2-sample.json']
        for relative in relatives:
            out = destination/relative
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes((ROOT/relative).read_bytes())
    hashes = {str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(destination.rglob('*')) if p.is_file()}
    return {'mosca_revision':revision,'source_sha256':hashes}


def stage_association(frames,masks,baseline,destination,*,min_frame_mass=1.,threshold=.8,
                      selection=None,all_frames=False):
    """Copy only explicit, bounded inputs; never a cache, key or user directory."""
    manifest=json.loads((frames/'manifest.json').read_text())
    if (manifest['frame_count'],manifest['height'],manifest['width'])!=(12,270,480):
        raise ValueError('bounded association runner expects the inspected Balloon2 sample')
    paths=[(frames/'manifest.json','frames/manifest.json'),(masks,'masks.npz'),(baseline,'baseline.json')]
    if selection is not None:paths.append((selection,'render-selection.npz'))
    if all_frames and selection is None:raise ValueError('all-frame render requires a saved selection')
    for row in manifest['frames']:
        name=row['file']
        if Path(name).name!=name or Path(name).suffix!='.png':raise ValueError('unsafe frame path')
        p=frames/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('frame hash differs')
        paths.append((p,'frames/'+name))
    root=destination/'association-input';hashes={}
    if not math.isfinite(min_frame_mass) or min_frame_mass<=0 or not .2<threshold<=1:
        raise ValueError('invalid association thresholds')
    for src,rel in paths:
        if src.stat().st_size>4*1024**2:raise ValueError('input exceeds inline staging bound')
        data=src.read_bytes();out=root/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
        hashes[rel]=hashlib.sha256(data).hexdigest()
    settings=json.dumps({'min_frame_mass':min_frame_mass,'threshold':threshold,
                         'mode':'render-selection' if selection is not None else 'associate',
                         'all_frames':all_frames},sort_keys=True).encode()
    (root/'settings.json').write_bytes(settings)
    hashes['settings.json']=hashlib.sha256(settings).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--mosca',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--real-scene',action='store_true',help='probe pinned Balloon2 checkpoint without SAM')
    parser.add_argument('--associate',action='store_true',help='associate existing cached masks, no SAM inference')
    parser.add_argument('--render-selection',type=Path,help='render a compact saved sidecar without reassociation')
    parser.add_argument('--all-frames',action='store_true',help='saved-selection mode only; render all 12 discrete times')
    parser.add_argument('--frames',type=Path)
    parser.add_argument('--masks',type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--min-frame-mass',type=float,default=1.)
    parser.add_argument('--threshold',type=float,default=.8)
    args = parser.parse_args()
    if args.associate and args.render_selection:parser.error('choose association or saved-selection rendering')
    if args.all_frames and not args.render_selection:parser.error('--all-frames requires --render-selection')
    stream_selection=bool(args.associate or args.render_selection)
    if not args.execute:
        print(json.dumps({**PLAN,'real_scene':args.real_scene,'associate':args.associate,
                          'saved_selection_render':bool(args.render_selection),'all_frames':args.all_frames},indent=2))
        return
    if not args.approval:
        parser.error('--execute requires --approval with verified billing/access evidence')
    approval = json.loads(args.approval.read_text())
    validate_approval(approval)
    if args.mosca is None or args.out is None:
        parser.error('--execute also requires --mosca and a new --out directory')
    if stream_selection and (not args.real_scene or not all([args.frames,args.masks,args.baseline])):
        parser.error('selection work requires --real-scene, --frames, --masks and --baseline')
    # Output creation and source validation precede any cloud calls.
    args.out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='reframe-smoke-') as temporary:
        source = Path(temporary)
        provenance = stage_sources(args.mosca,source,args.real_scene)
        provenance['real_scene'] = args.real_scene
        provenance['associate'] = args.associate
        provenance['saved_selection_render'] = bool(args.render_selection)
        if stream_selection:
            provenance['association_input_sha256']=stage_association(args.frames,args.masks,args.baseline,source,
                min_frame_mass=args.min_frame_mass,threshold=args.threshold,
                selection=args.render_selection,all_frames=args.all_frames)
        (args.out/'plan.json').write_text(json.dumps({**PLAN,**provenance},indent=2))
        import importlib.metadata
        if importlib.metadata.version('modal') != PLAN['modal_sdk_version']:
            raise RuntimeError('Use the inspected modal==1.6.2 SDK in a separate environment')
        # SDK defaults include api.modal2.com failover. Use only the explicitly
        # reviewed primary endpoint; a denied host must not trigger a bypass.
        if os.environ.get('MODAL_SERVER_URL', 'https://api.modal.com') != 'https://api.modal.com':
            raise RuntimeError('This bounded runner permits only the reviewed primary Modal endpoint')
        os.environ['MODAL_SERVER_URL'] = 'https://api.modal.com'
        # grpclib otherwise uses certifi alone, which omits this managed host's
        # proxy CA. System trust preserves certificate verification and policy.
        import truststore
        truststore.inject_into_ssl()
        import modal
        # Read-only identity lookup precedes all source uploads and image work.
        workspace = modal.Workspace.from_context().hydrate()
        verify_workspace(workspace.name, approval)
        validate_approval(approval)
        claim = claim_approval(approval,ROOT/'.cache/modal-launch-claims')
        (args.out/'launch-claim.txt').write_text(str(claim))
        app = modal.App('reframe-renderer-smoke',include_source=False)
        image = (modal.Image.from_registry(IMAGE).env({'PYTHONPATH':'/smoke'})
                 .add_local_dir(source,'/smoke',copy=False))
        # Importable staged source avoids cloudpickling local Python bytecode
        # into the older reference image's Python interpreter.
        sys.path.insert(0,str(ROOT))
        if stream_selection:
            from scripts.modal_scene_worker import association_entrypoint as remote_entrypoint
        elif args.real_scene:
            from scripts.modal_scene_worker import remote_entrypoint
        else:
            from scripts.modal_smoke_worker import remote_entrypoint
        smoke = app.function(image=image,gpu='A100-40GB',cpu=(2,2),memory=(16384,16384),
                      routing_region='us-east',
                      timeout=900,startup_timeout=120,
                      # SDK 1.6.2 forbids even retries=0 for generators;
                      # generators do not support retries at all.
                      **({} if stream_selection else {'retries':0}),
                      min_containers=0,max_containers=1,buffer_containers=0,
                      scaledown_window=2,single_use_containers=True,
                      serialized=False)(remote_entrypoint)

        with modal.enable_output():
            with app.run():
                # SDK 1.6.2 metadata was inspected; preserve the control/input
                # plane distinction before sending any function input.
                metadata = smoke._get_metadata()
                remote = {'app_id':app.app_id,'function_id':smoke.object_id,
                          'input_plane_host':urlsplit(metadata.input_plane_url).hostname,
                          'input_plane_region':metadata.input_plane_region}
                (args.out/'remote.json').write_text(json.dumps(remote,indent=2))
                if remote['input_plane_host'] != PLAN['expected_input_plane_host']:
                    raise RuntimeError('Unverified input-plane destination; refusing invocation')
                if stream_selection:
                    from scripts.artifact_stream import receive_artifacts
                    stream=iter(smoke.remote_gen())
                    first=next(stream)
                    if first.get('kind')!='result':raise ValueError('missing association result header')
                    result=first['result']
                    result['artifact_transfer']='pending'
                    (args.out/'result.json').write_text(json.dumps(result,indent=2))
                    try:
                        receive_artifacts(stream,args.out/'selection')
                        if result['exit_code']==0 and not all((args.out/'selection'/n).exists() for n in ['selection.npz','report.json']):
                            raise ValueError('successful association omitted required sidecars')
                        result['artifact_transfer']='complete'
                    except Exception as error:
                        result['artifact_transfer']='failed'
                        result['transfer_error_type']=type(error).__name__
                        raise
                    finally:
                        (args.out/'result.json').write_text(json.dumps(result,indent=2))
                else:
                    result = smoke.remote()
        (args.out/'result.json').write_text(json.dumps(result,indent=2))
        if result['exit_code'] or not result.get('report',{}):
            raise RuntimeError('Smoke test failed; inspect result.json before any retry')
        if args.real_scene and not stream_selection:
            import base64
            artifacts = args.out/'baseline'
            artifacts.mkdir()
            for name, data in result.get('artifacts',{}).items():
                if Path(name).name != name or Path(name).suffix not in ('.json','.png','.jpg'):
                    raise ValueError('unexpected returned artifact path')
                (artifacts/name).write_bytes(base64.b64decode(data,validate=True))
        print(f'Renderer smoke result saved to {args.out}; no real-scene selection established.')


if __name__ == '__main__':
    main()
