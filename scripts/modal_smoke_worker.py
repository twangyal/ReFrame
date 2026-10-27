"""Runs only inside the explicitly authorized ephemeral GPU function."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def remote_entrypoint():
    """Import this staged module on the GPU; do not serialize client bytecode."""
    result = subprocess.run([sys.executable,'/smoke/scripts/modal_smoke_worker.py'],
                            cwd='/smoke',capture_output=True,text=True,timeout=850)
    report_path = Path('/tmp/smoke-result.json')
    return {'exit_code':result.returncode,'stdout':result.stdout[-100000:],
            'stderr':result.stderr[-100000:],
            'report':json.loads(report_path.read_text()) if report_path.exists() else None}


def main():
    started = time.monotonic()
    for command in ('nvcc','g++'):
        if shutil.which(command) is None:
            raise RuntimeError(f'pinned image lacks {command}; inspect before rebuilding')
    # Compile during the single timed GPU invocation. No unbounded image RUN
    # layer, apt install, checkpoint download, training, or paid model request.
    env = {**os.environ,'MAX_JOBS':'2','TORCH_CUDA_ARCH_LIST':'8.0',
           'REFRAME_MOSCA_ROOT':'/smoke/mosca','GS_BACKEND':'native_add3',
           'PYTHONPATH':'/smoke','PIP_DISABLE_PIP_VERSION_CHECK':'1'}
    def run(command, seconds):
        remaining = 800-(time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError('smoke test wall time exhausted')
        subprocess.run(command,check=True,env=env,timeout=min(seconds,remaining))
    run([sys.executable,'-m','pip','install','--no-cache-dir','numpy==1.26.4',
         'ninja==1.11.1.1','setuptools==69.5.1','wheel==0.43.0'],180)
    import torch
    if not torch.cuda.is_available() or torch.version.cuda != '11.8':
        raise RuntimeError('expected working CUDA 11.8 reference image')
    if not torch.__version__.startswith('2.1.0') or torch.cuda.get_device_capability() != (8,0):
        raise RuntimeError('unexpected Torch version or GPU architecture')
    properties = torch.cuda.get_device_properties(0)
    if 'A100' not in properties.name or not 35*1024**3 <= properties.total_memory <= 45*1024**3:
        raise RuntimeError('expected exactly A100 40GB; no implicit GPU substitution')
    source = Path('/smoke/mosca/lib_render/diff-gaussian-rasterization-alphadep-add3')
    build = Path('/tmp/native-add3-build')
    shutil.copytree(source,build)
    run([sys.executable,'-m','pip','install','--no-cache-dir','--no-deps',
         '--no-build-isolation',str(build)],600)
    run([sys.executable,'-m','unittest','discover','-s','/smoke/tests',
         '-p','test_cuda_renderer.py','-v'],60)
    report = {'kind':'synthetic_actual_renderer_smoke','passed':True,
              'real_scene_selection_demonstrated':False,'gpu':properties.name,
              'gpu_total_bytes':properties.total_memory,'torch':torch.__version__,
              'cuda':torch.version.cuda,'elapsed_seconds':time.monotonic()-started,
              'nvcc':subprocess.check_output(['nvcc','--version'],text=True),
              'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)}
    Path('/tmp/smoke-result.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
