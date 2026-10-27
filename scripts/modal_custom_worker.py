"""Timed disposable custom-clip compatibility and actual BootsTAPIR inference."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def runtime_limits(quality_fit=False):
    if type(quality_fit) is not bool:raise ValueError('quality_fit must be boolean')
    return dict(worker_timeout_seconds=1400 if quality_fit else 800,
                child_timeout_seconds=1450 if quality_fit else 850,
                input_timeout_seconds=1500 if quality_fit else 900)


def remote_entrypoint():
    from scripts.artifact_stream import emit_artifacts
    started=time.monotonic()
    settings=json.loads(Path('/smoke/resource-settings.json').read_text())
    limits=runtime_limits(settings.get('quality_fit',False))
    try:
        r=subprocess.run([sys.executable,'/smoke/scripts/modal_custom_worker.py'],cwd='/smoke',capture_output=True,text=True,timeout=limits['child_timeout_seconds'])
        result=dict(exit_code=r.returncode,stdout=r.stdout[-100000:],stderr=r.stderr[-100000:])
    except subprocess.TimeoutExpired as e:
        def tail(v):return (v.decode(errors='replace') if isinstance(v,bytes) else v or '')[-100000:]
        result=dict(exit_code=124,stdout=tail(e.stdout),stderr=tail(e.stderr),timed_out=True)
    result.update(elapsed_seconds=time.monotonic()-started,artifacts_are_incomplete_diagnostics=result['exit_code']!=0)
    yield {'kind':'result','result':result}
    out=Path('/tmp/custom-output')
    if out.exists():yield from emit_artifacts(out,suffixes={'.json','.npz','.png','.zip'},max_bytes=512*1024**2)
    else:yield {'kind':'complete','files':0}


def main():
    started=time.monotonic();out=Path('/tmp/custom-output');out.mkdir()
    import torch
    properties=torch.cuda.get_device_properties(0)
    hardware=dict(name=properties.name,total_memory_bytes=properties.total_memory,
        capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,cuda=torch.version.cuda)
    (out/'hardware.json').write_text(json.dumps(hardware,indent=2))
    from scripts.modal_smoke_worker import validate_gpu
    settings=json.loads(Path('/smoke/resource-settings.json').read_text())
    limits=runtime_limits(settings.get('quality_fit',False))
    validate_gpu(properties.name,properties.total_memory,settings['gpu'])
    env={**os.environ,'PYTHONPATH':'/smoke:/smoke/mosca','GS_BACKEND':'native_add3','OMP_NUM_THREADS':'2',
         'MAX_JOBS':'2','TORCH_CUDA_ARCH_LIST':'8.0','PIP_DISABLE_PIP_VERSION_CHECK':'1',
         'REFRAME_EXPECTED_GPU':settings['gpu']}
    def run(cmd,seconds):
        remaining=limits['worker_timeout_seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('custom probe time budget exhausted')
        print('Stage:',cmd[:5],flush=True)
        subprocess.run(cmd,check=True,env=env,timeout=min(remaining,seconds))
    run(['apt-get','-o','Acquire::Retries=0','update'],60)
    run(['apt-get','-o','Acquire::Retries=0','install','-y','--no-install-recommends','libgl1','libegl1','libglapi-mesa','libgomp1'],180)
    run([sys.executable,'/smoke/scripts/modal_smoke_worker.py'],300)
    run([sys.executable,'-m','pip','install','--no-cache-dir','--retries','0','--timeout','30',
        'numpy==1.26.4','scipy==1.11.4','matplotlib==3.8.4','Pillow==10.4.0','torch-geometric==2.5.3',
        'open3d-cpu==0.18.0','iopath==0.1.10','kornia==0.7.3','kornia_rs==0.1.9','omegaconf==2.3.0',
        'tensorboardX==2.6.2.2','imageio==2.34.2','imageio-ffmpeg==0.5.1','opencv-python-headless==4.6.0.66',
        'mediapy==1.2.2','einshape==1.0','absl-py==2.1.0','scikit-image==0.22.0','transforms3d==0.4.2',
        'lpips==0.1.4','evo==1.30.0',
        '/smoke/weights/torch_cluster-1.6.3+pt21cu118-cp310-cp310-linux_x86_64.whl'],180)
    import urllib.request,hashlib
    package=Path('/tmp/pytorch3d-0.7.8-py310_cu118_pyt210.tar.bz2')
    with urllib.request.urlopen('https://conda.anaconda.org/pytorch3d/linux-64/'+package.name,timeout=30) as r,package.open('xb') as f:
        for b in iter(lambda:r.read(1024**2),b''):f.write(b)
    if hashlib.sha256(package.read_bytes()).hexdigest()!='8b657140d23cdfa4e3f7a641bbc0dcd3e7c8830d4061481b0ef1eaa30e83d49f':raise ValueError('PyTorch3D digest mismatch')
    run(['conda','install','--offline','--no-deps','--yes',str(package)],90)
    shutil.copytree('/smoke/mosca/lib_render/diff-gaussian-rasterization-alphadep','/tmp/native-build')
    run([sys.executable,'-m','pip','install','--no-cache-dir','--no-deps','--no-build-isolation','/tmp/native-build'],180)
    if Path('/smoke/stage.json').exists():
        from scripts.training_artifacts import pack,unpack
        settings=json.loads(Path('/smoke/stage.json').read_text())
        ws=Path('/tmp/custom-workspace');shutil.copytree('/smoke/priors',ws)
        fit=Path('/tmp/custom-fit')
        if Path('/smoke/previous.zip').exists():unpack(Path('/smoke/previous.zip'),fit)
        try:
            run([sys.executable,'/smoke/scripts/run_custom_stage.py','--stage',settings['stage'],
                 '--mosca','/smoke/mosca','--workspace',str(ws),'--output',str(fit),'--config','/smoke/fit-config.json'],max(1,limits['worker_timeout_seconds']-(time.monotonic()-started)))
        finally:
            if fit.exists():pack(fit,out/'stages.zip')
        report=json.loads((fit/(settings['stage']+'-report.json')).read_text())
    else:
        run([sys.executable,'/smoke/scripts/modal_custom_worker.py','--track'],max(1,limits['worker_timeout_seconds']-(time.monotonic()-started)))
        report=json.loads((out/'report.json').read_text())
    report['whole_worker_seconds']=time.monotonic()-started
    report['packages']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (out/'report.json').write_text(json.dumps(report,indent=2))


def track():
    import numpy as np,torch,imageio.v2 as imageio
    sys.path.insert(0,'/smoke/mosca')
    torch.manual_seed(12345);np.random.seed(12345);torch.set_num_threads(2)
    from torch_geometric.nn.pool import voxel_grid
    pos=torch.tensor([[0.,0.,0.],[.1,0,0],[2.,0,0]],device='cuda')
    clusters=voxel_grid(pos,size=1.)
    assert clusters[0]==clusters[1] and clusters[0]!=clusters[2]
    from lib_moca import camera as canonical_camera
    from scripts.custom_stages import install_camera_alias
    install_camera_alias(canonical_camera,sys.modules)
    from mosca_reconstruct import scaffold_reconstruct,photometric_reconstruct
    from lite_moca_reconstruct import static_reconstruct
    from lib_prior.tracking import bootstapir_wrapper as tracker
    # The upstream wrapper saves tracks after visualization. Disable that final
    # optional video so a visualization failure cannot discard inference.
    class NoTrackingVideo(tracker.Visualizer):
        def visualize(self, **kwargs):
            return None
    tracker.Visualizer=NoTrackingVideo
    get_bootstapir_model=tracker.get_bootstapir_model
    bootstapir_process_folder=tracker.bootstapir_process_folder
    from lib_prior.prior_loading import Saved2D
    from scripts.custom_tracking import validate_tracks,digest
    ws=Path('/tmp/custom-workspace');shutil.copytree('/smoke/priors',ws)
    out=Path('/tmp/custom-output')
    files=sorted((ws/'images').glob('*.png'));images=[imageio.imread(p) for p in files]
    assert len(images)==48 and all(im.shape==(270,480,3) for im in images)
    before={str(p.relative_to(ws)):digest(p) for p in ws.rglob('*') if p.is_file()}
    model=get_bootstapir_model('/smoke/weights/bootstapir_checkpoint_v2.pt',device=torch.device('cuda:0'))
    model.eval();t=time.monotonic()
    bootstapir_process_folder(str(ws),images,np.ones((48,270,480),dtype=bool),model,4096,
                             chunk_size=256,save_name='uniform_dep=external',max_viz_cnt=64,device=torch.device('cuda:0'))
    name='uniform_dep=external_bootstapir_tap.npz';shutil.copyfile(ws/name,out/name)
    with np.load(ws/name,allow_pickle=False) as data:stats=validate_tracks(data['tracks'],data['visibility'])
    saved=Saved2D(str(ws)).load_dep('external_depth',depth_boundary_th=.3).load_track('*uniform*bootstapir',min_valid_cnt=4)
    filtered_stats=validate_tracks(saved.track.cpu().numpy(),saved.track_mask.cpu().numpy())
    assert all(digest(ws/rel)==sha for rel,sha in before.items())
    report=dict(kind='actual-custom-clip-tracking',passed=True,seed=12345,model_eval=True,queries=4096,
        tracking_seconds=time.monotonic()-t,statistics=stats,filtered_statistics=filtered_stats,
        tracking_video_disabled=True,mosca_loaded_tracks=int(saved.track.shape[1]),
        compiled_voxel_grid_passed=True,full_training_imports_passed=True,original_hashes_preserved=True,
        source_image_sha256={p.name:digest(p) for p in files},track_sha256=digest(ws/name),
        cuda=torch.version.cuda,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        reconstruction_trained=False,sam_called=False)
    (out/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    if '--track' in sys.argv:track()
    else:main()
