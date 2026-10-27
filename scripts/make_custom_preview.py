#!/usr/bin/env python3
"""Create a private custom-reconstruction comparison from hash-bound RGB captures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image,ImageDraw

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reframe.media import sha256_file


def held_frame_indices(times,duration,fps):
    times=np.asarray(times,dtype=float)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all()
        or times[0]!=0 or (np.diff(times)<=0).any() or not np.isfinite(duration)
        or duration<=times[-1] or type(fps) is not int or not 1<=fps<=60):
        raise ValueError('invalid source sample timing or playback rate')
    # Output is explicitly sample-and-hold; no new model timestamps are invented.
    # prepare_video records ffprobe's six-decimal seconds; cover its rounding
    # error, not a whole output frame. Without this, .166667 misses 4/24.
    return np.searchsorted(times,np.arange(int(np.ceil(duration*fps)))/fps+5.1e-7,side='right')-1


def make_preview(scene,stages,out,fps=24,*,priors):
    scene=Path(scene);stages=Path(stages);out=Path(out);priors=Path(priors)
    if out.exists():raise FileExistsError(out)
    source=json.loads((scene/'custom-scene.json').read_text())
    fit=json.loads((stages/'photo-report.json').read_text())
    if fit.get('passed') is not True or fit.get('original_inputs_preserved') is not True:
        raise ValueError('requires a successful preserved-input photometric stage')
    prior_report=json.loads((priors/'report.json').read_text())
    if (sha256_file(priors/'report.json')!=fit['input_sha256'].get('report.json')
        or sha256_file(scene/'custom-scene.json')!=prior_report.get('prepared_manifest_sha256')):
        raise ValueError('source timing manifest binding mismatch')
    render_report_path=stages/'playback-report.json'
    if render_report_path.exists():
        rendered=json.loads(render_report_path.read_text())
        if (rendered.get('stage')!='playback' or rendered.get('passed') is not True
            or rendered.get('training_performed') is not False
            or rendered.get('original_inputs_preserved') is not True
            or rendered.get('original_trained_checkpoints_preserved') is not True
            or rendered.get('input_sha256')!=fit['input_sha256']):
            raise ValueError('playback evidence binding mismatch')
        for name in ('photometric_cam.pth','photometric_s_model_native_add3.pth','photometric_d_model_native_add3.pth'):
            sha=sha256_file(stages/name)
            if sha!=fit['output_sha256'].get(name) or sha!=rendered['output_sha256'].get(name):
                raise ValueError('playback checkpoint binding mismatch')
        if rendered['output_sha256'].get('photo-report.json')!=sha256_file(stages/'photo-report.json'):
            raise ValueError('playback predecessor report binding mismatch')
    else:
        render_report_path=stages/'photo-report.json';rendered=fit
    frames=source['frames'];times=[r['source_pts_seconds'] for r in frames]
    mapping=held_frame_indices(np.asarray(times)-times[0],source['source_duration_seconds'],fps)
    width,height=source['width'],source['height'];inputs={}
    for i,row in enumerate(frames):
        if row['reconstruction_index']!=i:raise ValueError('source index mismatch')
        path=scene/'images'/row['file'];key='images/'+row['file'];sha=sha256_file(path)
        if sha!=row['sha256'] or sha!=fit['input_sha256'].get(key):raise ValueError('source frame binding mismatch')
        inputs[key]=sha
        for view in ('source','nearby'):
            key=f'rgb-diagnostics/{i:06d}_{view}.png';path=stages/key;sha=sha256_file(path)
            if sha!=rendered['output_sha256'].get(key):raise ValueError('render binding mismatch')
            with Image.open(path) as im:
                if im.size!=(width,height):raise ValueError('native render dimensions differ')
            inputs[key]=sha
    out.mkdir(parents=True)
    video=out/'custom-cat-reconstruction-comparison.mp4'
    canvas_size=(width*3,height+48)
    proc=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{canvas_size[0]}x{canvas_size[1]}','-r',str(fps),'-i','pipe:0','-an',
        '-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    previous=None;pixels=None
    try:
        for i in mapping:
            if int(i)!=previous:
                canvas=Image.new('RGB',canvas_size,(20,20,20));draw=ImageDraw.Draw(canvas)
                paths=[scene/'images'/frames[i]['file'],stages/f'rgb-diagnostics/{i:06d}_source.png',stages/f'rgb-diagnostics/{i:06d}_nearby.png']
                labels=['Source: sampled frames','MoSca: source camera','MoSca: moving nearby camera']
                for column,(path,label) in enumerate(zip(paths,labels)):
                    with Image.open(path) as im:canvas.paste(im.convert('RGB'),(column*width,24))
                    draw.text((column*width+8,6),label,fill='white')
                draw.text((8,height+30),f'Frame {i:02d} | Reconstructed timestamps held at source timing. No SAM selection or edits.',fill='white')
                pixels=canvas.tobytes();previous=int(i)
            proc.stdin.write(pixels)
    finally:
        proc.stdin.close();code=proc.wait()
    if code:raise RuntimeError('preview encoding failed; partial output retained')
    record=dict(kind='actual-custom-reconstruction-comparison',video_sha256=sha256_file(video),
        photo_report_sha256=sha256_file(stages/'photo-report.json'),source_manifest_sha256=sha256_file(scene/'custom-scene.json'),
        render_report_sha256=sha256_file(render_report_path),
        prior_report_sha256=sha256_file(priors/'report.json'),timestamp_rounding_tolerance_seconds=5.1e-7,
        frame_sha256=inputs,output_fps=fps,output_to_reconstruction_index=mapping.tolist(),
        timing='Sample-and-hold using original PTS; no temporal interpolation or additional rendered timestamps.',
        selection_applied=False,transformation_applied=False,visual_review='pending')
    (out/'preview.json').write_text(json.dumps(record,indent=2)+'\n')
    return video


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','stages','out','priors'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fps',type=int,default=24)
    a=p.parse_args();print(make_preview(a.scene,a.stages,a.out,a.fps,priors=a.priors))

if __name__=='__main__':main()
