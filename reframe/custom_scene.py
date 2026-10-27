"""CPU preparation of a supplied short clip for MoSca's real images/ loader."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from .media import prepare_video,sha256_file


def prepare_custom_scene(video,out,*,max_frames=48,max_side=480):
    video=Path(video).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if type(max_frames) is not int or not 2<=max_frames<=120 or type(max_side) is not int or not 32<=max_side<=1024:
        raise ValueError('require 2..120 selected frames and 32..1024 maximum image side')
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames,avg_frame_rate,duration','-of','json',str(video)],text=True))
    stream=info['streams'][0]
    if stream.get('duration') in (None,'N/A'):raise ValueError('video stream duration is unavailable; provide a timed MP4')
    duration=float(stream['duration'])
    if not np.isfinite(duration) or not 3<=duration<=5:raise ValueError('supply a preselected 3–5 second clip; no silent trim')
    parts=stream['avg_frame_rate'].split('/')
    rate=float(parts[0])/float(parts[1]) if float(parts[1]) else 0
    estimated_frames=max(float(stream.get('nb_frames','0')) if stream.get('nb_frames','').isdigit() else 0,rate*duration)
    if not 0<estimated_frames<=600 or stream['width']*stream['height']*estimated_frames*3>4*1024**3:
        raise ValueError('input exceeds bounded CPU decode estimate; provide a lower-resolution/frame-rate clip')
    before=sha256_file(video);out.mkdir(parents=True)
    decoded=prepare_video(video,out/'decoded')
    indices=np.linspace(0,decoded['frame_count']-1,min(max_frames,decoded['frame_count'])).astype(int).tolist()
    scale=min(1.,max_side/max(decoded['width'],decoded['height']))
    width=max(2,int(decoded['width']*scale)//2*2);height=max(2,int(decoded['height']*scale)//2*2)
    images=out/'images';images.mkdir();rows=[]
    for i,source_index in enumerate(indices):
        source=decoded['frames'][source_index];target=images/f'{i:06d}.png'
        with Image.open(out/'decoded'/source['file']) as image:
            image.convert('RGB').resize((width,height),Image.Resampling.LANCZOS).save(target)
        rows.append({'reconstruction_index':i,'file':target.name,'sha256':sha256_file(target),
                     'source_decode_index':source_index,'source_pts_seconds':source['pts_seconds'],
                     'source_decoded_sha256':source['sha256']})
    if sha256_file(video)!=before:raise ValueError('source changed during preparation')
    report={'schema':1,'source_video_sha256':before,'source_duration_seconds':duration,
            'source_frame_count':decoded['frame_count'],'selected_decode_indices':indices,
            'width':width,'height':height,'resize':'Pillow LANCZOS, aspect preserved to nearest even dimensions',
            'original_timestamps_known':True,'frames':rows,'reconstruction_completed':False,
            'rights_status':'not inferred; provider/user must establish rights before training/publication',
            'timing_note':'Source PTS retained; MoSca uses discrete indices. Subsampling may have unequal time intervals.'}
    (out/'custom-scene.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',required=True);p.add_argument('--out',required=True)
    p.add_argument('--max-frames',type=int,default=48);p.add_argument('--max-side',type=int,default=480)
    args=p.parse_args()
    try:result=prepare_custom_scene(args.video,args.out,max_frames=args.max_frames,max_side=args.max_side)
    except (ValueError,FileNotFoundError,FileExistsError) as e:p.exit(2,f'{e}\n')
    print(f'Prepared {len(result["frames"])} source images; reconstruction has NOT run: {args.out}')


if __name__=='__main__':main()
