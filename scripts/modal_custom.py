#!/usr/bin/env python3
"""Bounded custom-clip tracking probe. Default prints plan; never creates credentials."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.modal_smoke import PLAN,IMAGE,validate_approval,claim_approval,verify_workspace
from scripts.custom_tracking import stage_priors,stage_weight,digest

WHEEL='torch_cluster-1.6.3+pt21cu118-cp310-cp310-linux_x86_64.whl'
WHEEL_SHA='c3c91386aa7037384c7ccbabcdef302e61333186a51a7d5ac58c80b40e22bd5a'

def stage(mosca,priors,weight,wheel,destination):
    revision=json.loads((ROOT/'upstream-revisions.json').read_text())['mosca']['commit']
    actual=subprocess.check_output(['git','-C',str(mosca),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:raise ValueError('incorrect MoSca revision')
    archive=subprocess.check_output(['git','-C',str(mosca),'archive',revision])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            p=Path(member.name)
            if p.is_absolute() or '..' in p.parts or not (member.isfile() or member.isdir()):
                raise ValueError('unsafe source archive')
            out=destination/'mosca'/p
            if member.isdir():out.mkdir(parents=True,exist_ok=True)
            else:
                out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(tar.extractfile(member).read())
    inputs=stage_priors(priors,destination/'priors')
    sha=stage_weight(weight,destination/'weights/bootstapir_checkpoint_v2.pt')
    if wheel.name!=WHEEL or digest(wheel)!=WHEEL_SHA:raise ValueError('compiled cluster wheel digest mismatch')
    shutil.copyfile(wheel,destination/'weights'/WHEEL)
    for relative in ['scripts/modal_custom_worker.py','scripts/custom_tracking.py','scripts/artifact_stream.py',
                     'scripts/modal_smoke_worker.py','scripts/custom_stages.py','scripts/run_custom_stage.py','scripts/training_artifacts.py','tests/test_cuda_renderer.py','reframe/__init__.py','reframe/association.py']:
        out=destination/relative;out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,out)
    return dict(mosca_revision=revision,input_sha256=inputs,tracking_weight_sha256=sha,
                wheel_sha256=WHEEL_SHA,source_sha256={str(p.relative_to(destination)):digest(p)
                  for p in destination.rglob('*') if p.is_file() and 'weights' not in p.parts and 'priors' not in p.parts})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    for name in ('approval','mosca','priors','weight','wheel','out'):p.add_argument('--'+name,type=Path)
    p.add_argument('--gpu',choices=['A100-40GB','A100-80GB'],default='A100-40GB')
    p.add_argument('--fit-stage',choices=['camera','scaffold','photo'])
    for name in ('tracks','config','previous'):p.add_argument('--'+name,type=Path)
    p.add_argument('--cumulative-budget-usd',type=float,default=5,help='explicit verified existing-credit envelope, never above 30')
    args=p.parse_args()
    budget=args.cumulative_budget_usd
    if not args.execute:
        print(json.dumps({**PLAN,'kind':args.fit_stage or 'custom-tracking-compatibility','queries':4096,'frames':48,'sam_calls':0,'cumulative_budget_usd':budget,'gpu':args.gpu},indent=2));return
    if not all(getattr(args,k) for k in ('approval','mosca','priors','weight','wheel','out')):p.error('execution requires explicit inputs, new output and approval evidence')
    approval=json.loads(args.approval.read_text());validate_approval(approval,cumulative_budget_usd=budget)
    if args.gpu!='A100-40GB' and approval.get('explicit_gpu_authorization')!=args.gpu:
        raise ValueError('nondefault GPU requires explicit resource authorization in the evidence record')
    args.out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='reframe-custom-') as temporary:
        source=Path(temporary)
        provenance=stage(args.mosca,args.priors,args.weight,args.wheel,source)
        (source/'resource-settings.json').write_text(json.dumps({'gpu':args.gpu}))
        if args.fit_stage:
            if args.tracks is None or args.config is None:p.error('fitting requires saved tracking artifact directory and config')
            if args.fit_stage!='camera' and args.previous is None:p.error('later fitting requires completed predecessor archive')
            report=json.loads((args.tracks/'report.json').read_text())
            name='uniform_dep=external_bootstapir_tap.npz'
            if not report['passed'] or digest(args.tracks/name)!=report['track_sha256']:raise ValueError('tracking result is unverified')
            for name_,sha_ in report['source_image_sha256'].items():
                if Path(name_).name!=name_ or digest(source/'priors/images'/name_)!=sha_:raise ValueError('tracking uses different frames')
            if len(report['source_image_sha256'])!=48:raise ValueError('tracking frame provenance incomplete')
            shutil.copyfile(args.tracks/name,source/'priors'/name)
            shutil.copyfile(args.config,source/'fit-config.json')
            if args.previous is not None:
                if args.previous.stat().st_size>512*1024**2:raise ValueError('predecessor exceeds input bound')
                shutil.copyfile(args.previous,source/'previous.zip')
            settings=dict(stage=args.fit_stage,config_sha256=digest(args.config),track_sha256=report['track_sha256'],previous_sha256=digest(args.previous) if args.previous else None)
            (source/'stage.json').write_text(json.dumps(settings,indent=2))
            provenance['fitting']=settings
        (args.out/'plan.json').write_text(json.dumps({**PLAN,**provenance,'kind':args.fit_stage or 'custom-tracking-compatibility','cumulative_budget_usd':budget,'gpu':args.gpu},indent=2))
        import importlib.metadata
        if importlib.metadata.version('modal')!=PLAN['modal_sdk_version']:raise ValueError('incorrect Modal SDK')
        if os.environ.get('MODAL_SERVER_URL','https://api.modal.com')!='https://api.modal.com':raise ValueError('unreviewed API endpoint')
        os.environ['MODAL_SERVER_URL']='https://api.modal.com'
        import truststore
        truststore.inject_into_ssl()
        import modal
        verify_workspace(modal.Workspace.from_context().hydrate().name,approval)
        validate_approval(approval,cumulative_budget_usd=budget)
        claim=claim_approval(approval,ROOT/'.cache/modal-launch-claims')
        (args.out/'launch-claim.txt').write_text(str(claim))
        app=modal.App('reframe-custom-tracking',include_source=False)
        image=modal.Image.from_registry(IMAGE).env({'PYTHONPATH':'/smoke'}).add_local_dir(source,'/smoke',copy=False)
        from scripts.modal_custom_worker import remote_entrypoint
        worker=app.function(image=image,gpu=args.gpu,cpu=(2,2),memory=(16384,16384),routing_region='us-east',
            timeout=900,startup_timeout=120,min_containers=0,max_containers=1,buffer_containers=0,
            scaledown_window=2,single_use_containers=True,serialized=False)(remote_entrypoint)
        # Generators have no retries in this SDK. No Volumes, secrets or persistent disks.
        from scripts.artifact_stream import receive_artifacts
        with modal.enable_output(),app.run():
            metadata=worker._get_metadata()
            remote=dict(app_id=app.app_id,function_id=worker.object_id,input_plane_host=urlsplit(metadata.input_plane_url).hostname)
            (args.out/'remote.json').write_text(json.dumps(remote,indent=2))
            if remote['input_plane_host']!=PLAN['expected_input_plane_host']:raise ValueError('unreviewed input plane')
            stream=iter(worker.remote_gen());first=next(stream)
            if first.get('kind')!='result':raise ValueError('missing result header')
            result=first['result'];result['artifact_transfer']='pending'
            (args.out/'result.json').write_text(json.dumps(result,indent=2))
            try:
                receive_artifacts(stream,args.out/'artifacts',suffixes={'.json','.npz','.png','.zip'},max_bytes=512*1024**2)
                result['artifact_transfer']='complete'
            except Exception as error:
                result['artifact_transfer']='failed'
                result['transfer_error_type']=type(error).__name__
                raise
            finally:(args.out/'result.json').write_text(json.dumps(result,indent=2))
        if result['exit_code']!=0:raise RuntimeError('custom tracking failed; inspect preserved diagnostics before further work')
        if args.fit_stage:
            from scripts.training_artifacts import unpack
            unpack(args.out/'artifacts/stages.zip',args.out/'stages')
            report=json.loads((args.out/'stages'/(args.fit_stage+'-report.json')).read_text())
            if not report.get('passed'):raise RuntimeError('stage did not pass')
            print('Completed stage artifacts preserved:',args.fit_stage,'; visual quality must be evaluated separately.')
        else:
            if not (args.out/'artifacts/uniform_dep=external_bootstapir_tap.npz').exists():raise RuntimeError('tracking output missing')
            print('Custom tracking artifacts preserved; reconstruction has not yet been trained.')

if __name__=='__main__':main()
