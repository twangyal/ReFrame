"""Create a PRIVATE source-run desktop demo archive, never a public release."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from reframe.desktop_scene import SceneBundle
from reframe.media import sha256_file

MODULES=('__init__','desktop','desktop_scene','desktop_session','desktop_export',
         'edit_export','edit_render','editing','evaluation','media','reconstruction',
         'selection','selection_render','visualization','association')


def package(scene_path,out):
    scene=SceneBundle.load(scene_path);out=Path(out)
    if out.exists():raise FileExistsError(out)
    payload={f'reframe/{name}.py':(ROOT/'reframe'/f'{name}.py').read_bytes() for name in MODULES}
    payload['requirements-desktop.txt']=(ROOT/'requirements-desktop.txt').read_bytes()
    payload['sample/scene.json']=scene.manifest_path.read_bytes()
    for name in scene.data['files']:payload['sample/'+name]=scene._verified(name).read_bytes()
    payload['launch.sh']=b'#!/usr/bin/env sh\nset -eu\ncd "$(dirname "$0")"\npython -m reframe.desktop --scene sample/scene.json\n'
    payload['launch.bat']=b'@echo off\r\ncd /d "%~dp0"\r\npython -m reframe.desktop --scene sample/scene.json\r\npause\r\n'
    payload['README.txt']=('ReFrame private desktop demo\n\n'
      'Requires Python 3.11+ with Tk, and FFmpeg/ffprobe on PATH.\n'
      'Install: python -m pip install -r requirements-desktop.txt\n'
      'Launch: python -m reframe.desktop --scene sample/scene.json\n'
      'Or use launch.bat on Windows / sh launch.sh on Linux or macOS.\n\n'
      'This is a Python desktop application, not a standalone executable.\n'
      'Linux native GUI tested; Windows/macOS packaging unverified.\n'
      'Saved GPU renders only. No live reconstruction or arbitrary edits.\n'
      'Select the balloon, play/scrub, choose source/nearby view and original,\n'
      'highlight, isolated or orange recolor. Export writes all states to a new MP4.\n'
      'Clean translation failed visual review and is unavailable.\n\n'
      +scene.data['license_notice']+'\n'
      'Do not publish this archive, its screenshots or exported footage.\n'
      'No third-party binaries, credentials, model weights or cloud access included.\n').encode()
    inventory={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()}
    payload['ARCHIVE_MANIFEST.json']=json.dumps({'schema':1,'private_only':True,'scene_sha256':scene.digest,
        'files':inventory},indent=2).encode()
    # Exclusive archive creation; its inputs are an explicit allowlist.
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):archive.writestr(name,data)
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None:raise ValueError('Archive verification failed')
    return {'file':str(out),'bytes':out.stat().st_size,'sha256':sha256_file(out),'files':len(payload),'private_only':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(a.scene,a.out),indent=2))


if __name__=='__main__':main()
