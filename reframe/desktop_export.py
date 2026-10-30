"""CPU export of the chosen verified saved-render sequence, with no overwrite."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from .edit_export import _encode
from .media import sha256_file


def export_session(session,target,*,progress=None):
    scene=session.scene;mode=session.mode;view=session.view
    if mode!='baseline' and not session.selected:raise ValueError('No selected object for this edit')
    target=Path(target).resolve();sidecar=target.with_suffix('.export.json')
    if target.suffix.lower()!='.mp4':raise ValueError('Choose an MP4 export filename')
    if target.exists() or sidecar.exists():raise FileExistsError('Choose a new filename; exports never overwrite files')
    if not target.parent.is_dir():raise ValueError('Export directory does not exist')
    if sha256_file(scene.manifest_path)!=scene.digest:raise ValueError('Scene manifest changed since opening')
    # A private snapshot avoids changing source frames during FFmpeg encoding.
    with tempfile.TemporaryDirectory(prefix='.reframe-export-',dir=target.parent) as directory:
        tmp=Path(directory);hashes={}
        for t in range(scene.frames):
            source=scene.image_path(t,view,mode);copy=tmp/f'{t:06d}.png';shutil.copyfile(source,copy)
            digest=sha256_file(copy)
            if digest!=scene.data['files'][source.name]:raise ValueError('Scene frame changed during export')
            hashes[source.name]=digest
            if progress:progress(t+1,scene.frames)
        video=_encode(tmp/'%06d.png',tmp/'video.mp4',scene.fps,scene.frames)
        video['file']=target.name
        report={'schema':1,'kind':'reframe-desktop-export','scene_sha256':scene.digest,
                'object_id':scene.object_id,'selected':session.selected,'mode':mode,'view':view,
                'frames':scene.frames,'playback_fps':scene.fps,'interpolated':False,
                'source':'verified saved GPU renders; no new rendering or inference',
                'private_only':True,'license_notice':scene.data['license_notice'],
                'source_frames_sha256':hashes,'video':video}
        meta=tmp/'export.json';meta.write_text(json.dumps(report,indent=2)+'\n')
        # Same-directory hard links publish completed files exclusively. A race
        # or unsupported filesystem fails safely; never replace an existing file.
        published=False
        try:
            os.link(tmp/'video.mp4',target);published=True
            os.link(meta,sidecar)
        except OSError:
            if published:target.unlink()
            raise
    return report
