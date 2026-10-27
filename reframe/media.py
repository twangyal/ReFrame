"""Exact decode order, dimensions, presentation timestamps and content hashes."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def scene_images(source):
    """Yield RGB arrays in upstream Saved2D order; images.npz takes precedence."""
    root=Path(source)
    if (root/'images.npz').exists():
        with np.load(root/'images.npz',allow_pickle=False) as packed:
            images=packed['images']
        for img in images:
            yield img[...,:3]
    else:
        paths=sorted(p for p in (root/'images').iterdir() if p.suffix in {'.jpg','.png'})
        for p in paths:
            with Image.open(p) as image:
                yield np.asarray(image.convert('RGB'))


def prepare_scene_video(source,directory,*,fps=30):
    """Encode exact reconstruction images as lossless H.264 RGB, then verify them.

    The playback time base is chosen here; original movie timestamps are unknown.
    Hosted SAM accepted this lossless RGB H.264 format on the Balloon2 experiment.
    """
    directory=Path(directory).resolve();source=Path(source).resolve()
    if directory==source or source in directory.parents:
        raise ValueError('prepared files must stay outside the original scene directory')
    if not np.isfinite(fps) or fps<=0: raise ValueError('fps must be positive')
    images=iter(scene_images(source))
    first=next(images,None)
    if first is None: raise ValueError('scene contains no input frames')
    h,w=first.shape[:2]
    directory.mkdir(parents=True,exist_ok=False)
    video=directory/'scene.mp4'
    process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264rgb','-crf','0',
             '-preset','fast','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    import itertools
    try:
        for img in itertools.chain([first],images):
            if img.dtype!=np.uint8 or img.shape!=(h,w,3):
                raise ValueError('scene frames must be same-size uint8 RGB')
            process.stdin.write(np.ascontiguousarray(img).tobytes())
    finally:
        process.stdin.close()
        code=process.wait()
    if code: raise RuntimeError('lossless frame encoding failed')
    manifest=prepare_video(video,directory/'frames')
    originals=list(scene_images(source))
    if len(originals)!=manifest['frame_count']: raise ValueError('lossless encode changed frame count')
    for img,frame in zip(originals,manifest['frames']):
        with Image.open(directory/'frames'/frame['file']) as decoded:
            if not np.array_equal(img,np.asarray(decoded.convert('RGB'))):
                raise ValueError('lossless encode changed source pixels')
    manifest['original_timestamps_known']=False
    manifest['playback_fps']=fps
    manifest['reconstruction_source']=str(source)
    (directory/'frames'/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def prepare_video(video, directory):
    video=Path(video).resolve(); directory=Path(directory)
    if directory.exists():
        raise FileExistsError(f'refusing to overwrite existing frame directory: {directory}')
    if video.suffix.lower() not in {'.mp4','.mov'}:
        raise ValueError('SAM accepts MP4 or MOV; transcode explicitly before preparing')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                    '-show_streams','-show_frames','-of','json',str(video)]))
    stream=probe['streams'][0]
    if stream['codec_name'] not in {'h264','hevc','mpeg4'}:
        raise ValueError('unsupported SAM video codec')
    if any(float(s.get('rotation',0)) != 0 for s in stream.get('side_data_list',[])) or float(stream.get('tags',{}).get('rotate',0)) != 0:
        raise ValueError('normalize video rotation explicitly before frame preparation')
    pts=[float(f['best_effort_timestamp_time']) for f in probe['frames'] if f.get('media_type')=='video']
    if not 1 <= len(pts) <= 15000 or not np.isfinite(pts).all() or (np.diff(pts)<=0).any():
        raise ValueError('invalid frame count or non-monotonic presentation timestamps')
    directory.mkdir(parents=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-map','0:v:0',
                    '-vsync','0',str(directory/'%06d.png')],check=True)
    paths=sorted(directory.glob('*.png'))
    if len(paths)!=len(pts):
        raise ValueError('decode/probe count mismatch; partial directory retained for diagnosis')
    size=(stream['width'],stream['height'])
    for p in paths:
        with Image.open(p) as img:
            if img.size != size: raise ValueError('decoded frame resolution changed')
    manifest={'schema':1,'video_sha256':sha256_file(video),'width':size[0],'height':size[1],
              'frame_count':len(paths),'resize':'none','crop':'none','source_video':str(video),
              'frames':[{'index':i,'file':p.name,'pts_seconds':pts[i],'sha256':sha256_file(p)} for i,p in enumerate(paths)]}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def validate_manifest(directory):
    directory=Path(directory).resolve()
    m=json.loads((directory/'manifest.json').read_text())
    if m['schema']!=1 or m['frame_count']!=len(m['frames']) or m['frame_count']<1:
        raise ValueError('invalid frame manifest')
    times=[]
    for i,f in enumerate(m['frames']):
        p=(directory/f['file']).resolve()
        if p.parent!=directory or f['index']!=i or sha256_file(p)!=f['sha256']:
            raise ValueError('frame content/order differs from manifest')
        with Image.open(p) as image:
            if image.size!=(m['width'],m['height']): raise ValueError('frame dimensions differ from manifest')
        times.append(f['pts_seconds'])
    if not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('invalid presentation timestamps')
    return m
