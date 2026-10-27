"""Bounded real-checkpoint probe; no segmentation request or model training."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def collect_artifacts(directory):
    import io
    from PIL import Image
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix in ('.png','.json')]
    artifacts = {p.name:base64.b64encode(p.read_bytes()).decode() for p in files}
    encoding = 'PNG lossless'
    if len(json.dumps(artifacts)) > 1500000:
        artifacts = {}
        for p in files:
            if p.suffix == '.png':
                buffer = io.BytesIO()
                with Image.open(p) as image:
                    image.convert('RGB').save(buffer,format='JPEG',quality=95,optimize=True)
                artifacts[p.with_suffix('.jpg').name] = base64.b64encode(buffer.getvalue()).decode()
            else: artifacts[p.name] = base64.b64encode(p.read_bytes()).decode()
        encoding = 'JPEG quality 95 previews; original render dimensions'
    if len(json.dumps(artifacts)) > 1500000:
        # Always preserve the structured result, even if unexpected future
        # scene dimensions exceed this specifically bounded transport budget.
        artifacts = {k:v for k,v in artifacts.items() if k.endswith('.json')}
        encoding = 'images exceed inline budget; report only, visual validation blocked'
    return artifacts, encoding


def remote_entrypoint():
    try:
        result = subprocess.run([sys.executable,'/smoke/scripts/modal_scene_worker.py'],
                                cwd='/smoke',capture_output=True,text=True,timeout=850)
    except subprocess.TimeoutExpired as error:
        def text(value):
            return (value.decode('utf-8',errors='replace') if isinstance(value,bytes) else value or '')[-100000:]
        return {'exit_code':124,'stdout':text(error.stdout),'stderr':text(error.stderr),
                'report':None,'artifacts':{},'timed_out':True}
    path = Path('/tmp/scene-result.json')
    report = json.loads(path.read_text()) if path.exists() else None
    artifacts = {}
    if result.returncode == 0:
        artifacts, encoding = collect_artifacts(Path('/tmp/scene-baseline'))
        report['returned_image_encoding'] = encoding
    return {'exit_code':result.returncode,'stdout':result.stdout[-100000:],
            'stderr':result.stderr[-100000:],'report':report,'artifacts':artifacts}


def association_entrypoint():
    """One timed invocation; stream exact PNGs and persistent sidecars privately."""
    from scripts.artifact_stream import emit_artifacts
    try:
        result=subprocess.run([sys.executable,'/smoke/scripts/modal_scene_worker.py'],
                              cwd='/smoke',capture_output=True,text=True,timeout=850)
        path=Path('/tmp/scene-result.json')
        report=json.loads(path.read_text()) if path.exists() else None
        outcome={'exit_code':result.returncode,'stdout':result.stdout[-100000:],
                 'stderr':result.stderr[-100000:],'report':report}
    except subprocess.TimeoutExpired as error:
        def text(value):return (value.decode(errors='replace') if isinstance(value,bytes) else value or '')[-100000:]
        outcome={'exit_code':124,'stdout':text(error.stdout),'stderr':text(error.stderr),
                 'report':None,'timed_out':True}
    directory=Path('/tmp/scene-selection')
    outcome['artifacts_are_incomplete_diagnostics']=outcome['exit_code']!=0
    yield {'kind':'result','result':outcome}
    if directory.exists():
        yield from emit_artifacts(directory)
    else:
        yield {'kind':'complete','files':0}


def selection_command(settings,checkpoint):
    mode=settings.get('mode','associate')
    if mode not in ('associate','render-selection'):raise ValueError('invalid selection worker mode')
    command=[sys.executable,'-m','reframe.cli',mode,'--mosca','/tmp/mosca',
             '--checkpoint',str(checkpoint),'--scene-source','/tmp/scene-originals/nvidia/Balloon2',
             '--frames','/smoke/association-input/frames','--masks','/smoke/association-input/masks.npz',
             '--out','/tmp/scene-selection']
    if mode=='render-selection':
        command+=['--selection','/smoke/association-input/render-selection.npz']
        if settings.get('all_frames'):command+=['--all-frames']
    else:
        command+=['--baseline','/smoke/association-input/baseline.json','--threshold',str(settings['threshold']),
                  '--min-frame-mass',str(settings['min_frame_mass'])]
    return command


def main():
    started = time.monotonic()
    env = {**os.environ,'PYTHONPATH':'/smoke','GS_BACKEND':'native_add3',
           'PIP_DISABLE_PIP_VERSION_CHECK':'1',
           'OMP_NUM_THREADS':'2'}
    def run(command, seconds):
        remaining = 800-(time.monotonic()-started)
        if remaining <= 0: raise TimeoutError('real-scene probe wall time exhausted')
        print('Stage:', command[0], command[1:4], flush=True)
        subprocess.run(command,check=True,env=env,timeout=min(seconds,remaining))

    # The verified CUDA base lacks git. Install it explicitly in this timed,
    # disposable container; no persistent image/Volume or automatic retries.
    run(['apt-get','-o','Acquire::Retries=0','update'],60)
    run(['apt-get','-o','Acquire::Retries=0','install','-y','--no-install-recommends',
         'git','libgl1','libegl1','libglapi-mesa','libgomp1'],60)
    # Rebuild and repeat the independent three-Gaussian reference in this
    # ephemeral environment before introducing full-scene dependencies.
    run([sys.executable,'/smoke/scripts/modal_smoke_worker.py'],600)
    run([sys.executable,'-m','pip','install','--no-cache-dir','--retries','0',
         '--timeout','30','numpy==1.26.4','scipy==1.11.4','matplotlib==3.8.4',
         'Pillow==10.4.0','torch-geometric==2.5.3','open3d-cpu==0.18.0','iopath==0.1.10'],180)
    package = Path('/tmp/pytorch3d-0.7.8-py310_cu118_pyt210.tar.bz2')
    with urllib.request.urlopen('https://conda.anaconda.org/pytorch3d/linux-64/'+package.name,timeout=30) as response, package.open('xb') as out:
        for block in iter(lambda:response.read(1024**2),b''): out.write(block)
    if hashlib.sha256(package.read_bytes()).hexdigest() != '8b657140d23cdfa4e3f7a641bbc0dcd3e7c8830d4061481b0ef1eaa30e83d49f':
        raise ValueError('official PyTorch3D package digest mismatch')
    # Conda performs binary prefix relocation; explicit local archive and
    # --no-deps avoid replacing the verified Torch/CUDA stack.
    run(['conda','install','--offline','--no-deps','--yes',str(package)],90)
    run(['git','init','/tmp/mosca'],10)
    from reframe.reconstruction import MOSCA_REVISION
    run(['git','-C','/tmp/mosca','fetch','--depth=1','https://github.com/JiahuiLei/MoSca.git',MOSCA_REVISION],90)
    run(['git','-C','/tmp/mosca','checkout','--detach','FETCH_HEAD'],20)
    # Validate the complete upstream import chain before another data download.
    run([sys.executable,'-c',
         'import sys; sys.path.insert(0,"/tmp/mosca"); '
         'from lib_moca.camera import MonocularCameras; '
         'from lib_mosca.static_gs import StaticGaussian; '
         'from lib_mosca.dynamic_gs import DynSCFGaussian; '
         'from pytorch3d.ops import knn_points; '
         'print("Full MoSca camera/static/dynamic imports passed")'],45)
    run([sys.executable,'/smoke/scripts/fetch_mosca_sample.py','--manifest',
         '/smoke/configs/balloon2-sample.json','--out','/tmp/scene-originals'],240)
    manifest = json.loads(Path('/smoke/configs/balloon2-sample.json').read_text())
    checkpoint = Path('/tmp/scene-originals')/manifest['checkpoint_relative']
    association=Path('/smoke/association-input').exists()
    if association:
        settings=json.loads(Path('/smoke/association-input/settings.json').read_text())
        run(selection_command(settings,checkpoint),240)
    else:
        run([sys.executable,'-m','reframe.cli','probe-mosca','--mosca','/tmp/mosca',
             '--checkpoint',str(checkpoint),'--out','/tmp/scene-baseline'],180)
    from scripts.fetch_mosca_sample import digest
    for m in manifest['members']:
        if digest(Path('/tmp/scene-originals')/m['archive_path']) != m['sha256']:
            raise ValueError('original sample changed during probe')
    report = {'kind':('saved-selection-render' if settings.get('mode')=='render-selection' else 'real-mask-association') if association else 'real-mosca-baseline','passed':True,
              'real_scene_selection_demonstrated':False,'sam_called':False,
              'elapsed_seconds':time.monotonic()-started,
              'original_hashes_preserved':True,
              'evaluation' if association else 'baseline':json.loads(Path('/tmp/scene-selection/report.json' if association else '/tmp/scene-baseline/baseline.json').read_text()),
              'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
    Path('/tmp/scene-result.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__': main()
