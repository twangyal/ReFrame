"""Offline verified video export from native edit renders; no model calls."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw
from .edit_render import MODES,VIEWS
from .media import sha256_file


def validate_render_set(directory):
    directory=Path(directory);report=json.loads((directory/'report.json').read_text())
    if (report.get('kind')!='gaussian-object-edits' or report.get('completed') is not True
        or report.get('status')!='rendered_pending_visual_review'
        or report.get('modes')!=list(MODES) or report.get('views')!=list(VIEWS)
        or type(report.get('frames')) is not int or report['frames']<1
        or report.get('rendered_frames')!=list(range(report['frames']))):
        raise ValueError('incomplete or unsupported rendered edit report')
    expected={f'{t:06d}_{view}_{mode}.png' for t in report['rendered_frames'] for view in VIEWS for mode in MODES}
    if set(report['files'])!=expected:raise ValueError('incomplete render inventory')
    if sha256_file(directory/'selection.npz')!=report['selection_sha256']:raise ValueError('selection sidecar hash differs')
    for name in sorted(expected):
        path=directory/name
        if sha256_file(path)!=report['files'][name]:raise ValueError('native render hash differs')
        with Image.open(path) as image:
            image.load()
            if image.size!=(report['width'],report['height']) or image.mode!='RGB':raise ValueError('render dimensions or color mode differ')
    return report


def _encode(pattern,target,fps,count):
    subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate',str(fps),'-i',str(pattern),
                    '-frames:v',str(count),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(target)],check=True)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(target),'-f','null','-'],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
         '-show_entries','stream=codec_name,width,height,nb_read_frames,duration','-of','json',str(target)],text=True))['streams'][0]
    if (probe['codec_name']!='h264' or int(probe['nb_read_frames'])!=count
        or abs(float(probe['duration'])-count/fps)>1/fps):raise ValueError('encoded video differs from requested timeline')
    return {'file':target.name,'sha256':sha256_file(target),'bytes':target.stat().st_size,
            'decoded_frames':int(probe['nb_read_frames']),'duration_seconds':float(probe['duration']),
            'width':probe['width'],'height':probe['height'],'codec':probe['codec_name']}


def export_edits(render_dir,out,*,fps=3):
    if isinstance(fps,bool) or not isinstance(fps,(float,int)) or not math.isfinite(fps) or not 0<fps<=60:
        raise ValueError('invalid playback rate')
    render_dir=Path(render_dir).resolve();out=Path(out).resolve()
    if out==render_dir or render_dir in out.parents:raise ValueError('export must be outside native render directory')
    report=validate_render_set(render_dir)
    w,h=report['width'],report['height']
    if w%2 or h%2:raise ValueError('H264 export requires even native dimensions')
    out.mkdir(parents=True,exist_ok=False)
    manifest={'kind':'gaussian-edits-video-export','status':'exporting','phase_two_complete':False,
              'visual_review':'pending inspection of exported videos','object_id':report['object_id'],
              'selection_sha256':report['selection_sha256'],'render_report_sha256':sha256_file(render_dir/'report.json'),
              'reconstruction_states':report['frames'],'playback_fps':fps,'interpolated':False,
              'timing':'chosen playback rate of saved discrete states; acquisition cadence is not established',
              'translation_world':report['translation_world'],'private_only':True,'videos':{}}
    def write():(out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    write()
    for mode in ('baseline','recolored','translated'):
        manifest['videos'][mode]=_encode(render_dir/f'%06d_source_{mode}.png',out/f'balloon2-{mode}.mp4',fps,report['frames']);write()
    for label,modes in [('edits',('baseline','recolored','translated')),('selection',('baseline','highlight','isolated'))]:
        frames=out/(label+'-frames');frames.mkdir()
        for t in report['rendered_frames']:
            canvas=Image.new('RGB',(3*w,2*h+100),'#171717');draw=ImageDraw.Draw(canvas)
            draw.text((8,6),f'ReFrame | {label} | state {t}/{report["frames"]-1} | chosen {fps:g} states/sec',fill='white')
            for row,view in enumerate(VIEWS):
                for col,mode in enumerate(modes):
                    y=38+row*(h+28)
                    with Image.open(render_dir/f'{t:06d}_{view}_{mode}.png') as image:canvas.paste(image,(col*w,y))
                    draw.text((col*w+6,y-18),f'{view} | {mode}',fill='white')
            draw.text((8,2*h+78),'Exploratory selection | no interpolation, inpainting or shadow correction | nearby view only',fill='white')
            canvas.save(frames/f'{t:06d}.png')
        manifest['videos'][label+'-comparison']=_encode(frames/'%06d.png',out/f'balloon2-{label}-comparison.mp4',fps,report['frames']);write()
    manifest['status']='encoded_and_decoded_pending_visual_review';write()
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renders',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fps',type=float,default=3.)
    args=parser.parse_args();result=export_edits(args.renders,args.out,fps=args.fps)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
