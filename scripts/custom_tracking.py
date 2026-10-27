"""Explicit staging and result validation for the custom-clip tracking experiment."""
import hashlib
import json
from pathlib import Path
import shutil

WEIGHT_SHA256='8493c7a69e02c85b9382fbb3c7b8b539b36bc08ede744b9e99feb739a0129f4b'

def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):sha.update(b)
    return sha.hexdigest()

def stage_priors(source,destination):
    report=json.loads((source/'report.json').read_text())
    rows=report['rows']
    if not report['all_prepared_frames'] or [r['prepared_index'] for r in rows]!=list(range(48)):
        raise ValueError('expected all 48 prepared frames in order')
    files=[('report.json',digest(source/'report.json'))]
    for row in rows:
        name=row['file']
        if Path(name).name!=name or name!=f"{row['prepared_index']:06d}.png":
            raise ValueError('unsafe or unordered frame path')
        files.extend([(f'images/{name}',row['source_image_sha256']),
                      (f'external_depth/{Path(name).stem}.npz',row['depth_sha256'])])
    # Validate everything before copying anything.
    if sum((source/name).stat().st_size for name,_ in files)>128*1024**2:
        raise ValueError('priors exceed bounded input size')
    for name,sha in files:
        if (source/name).is_symlink() or digest(source/name)!=sha:raise ValueError('input hash mismatch or symlink')
    destination.mkdir(parents=True,exist_ok=False)
    for name,_ in files:
        out=destination/name;out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,out)
    return dict(files)

def stage_weight(source,destination):
    if source.is_symlink() or source.stat().st_size!=218886140 or digest(source)!=WEIGHT_SHA256:
        raise ValueError('official BootsTAPIR weight digest/size mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    return WEIGHT_SHA256

def validate_tracks(tracks,visibility):
    import numpy as np
    if (tracks.ndim!=3 or tracks.shape[0]!=48 or tracks.shape[2]!=2 or tracks.shape[1]<128
        or visibility.shape!=tracks.shape[:2] or visibility.dtype!=np.bool_ or not np.isfinite(tracks).all()):
        raise ValueError('invalid track time/shape/visibility/coordinates')
    valid=(visibility.sum(axis=0)>=4)
    if valid.sum()<128:raise ValueError('insufficient visible tracks for camera solve')
    return dict(frame_count=48,track_count=int(tracks.shape[1]),
                tracks_visible_at_least_four_frames=int(valid.sum()),
                visible_fraction=float(visibility.mean()))
