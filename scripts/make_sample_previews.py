"""Private browser-playable previews; never use their lossy pixels for alignment.

The reconstruction preview holds three already-rendered frames. It does not
interpolate scene motion or imply object selection. Originals remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image,ImageDraw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--sam',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--sam-object-id',default='0')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.prepared/'frames/manifest.json').read_text())
    width,height=manifest['width'],manifest['height'];count=manifest['frame_count']
    fps=manifest['playback_fps'];duration=count/fps
    if width%2 or height%2:raise ValueError('preview requires even dimensions for yuv420p')
    for kind in ['source','sam']:
        folder=args.out/(kind+'-frames');folder.mkdir()
        for i,frame in enumerate(manifest['frames']):
            source=args.prepared/'frames'/frame['file'] if kind=='source' else args.sam/f'overlay_{i:06d}.png'
            canvas=Image.new('RGB',(width,height+26),'#151515')
            with Image.open(source) as image:canvas.paste(image.convert('RGB'),(0,26))
            label=f'SOURCE frames | chosen {fps:g} fps playback | frame {i}' if kind=='source' else f'Hosted SAM 3.1 mask | object {args.sam_object_id} | source frame {i}'
            ImageDraw.Draw(canvas).text((8,7),label,fill='white');canvas.save(folder/f'{i:03d}.png')
    folder=args.out/'baseline-frames';folder.mkdir()
    times=sorted({0,count//2,count-1})
    for i,t in enumerate(times):
        canvas=Image.new('RGB',(width*2,height+44),'#151515');draw=ImageDraw.Draw(canvas)
        draw.text((8,5),f'MoSca reconstruction | sampled frame {t} | held stills, no object selection',fill='white')
        for c,view in enumerate(['source','nearby']):
            draw.text((c*width+8,25),f'{view} camera',fill='white')
            stem=args.baseline/f'{t:06d}_{view}_baseline'
            image_path=stem.with_suffix('.png') if stem.with_suffix('.png').exists() else stem.with_suffix('.jpg')
            with Image.open(image_path) as image:canvas.paste(image.convert('RGB'),(width*c,44))
        canvas.save(folder/f'{i:03d}.png')
    outputs=[]
    for kind,rate,name in [('source',fps,'balloon2-source-preview.mp4'),('baseline',len(times)/duration,'balloon2-reconstruction-sampled-preview.mp4'),('sam',fps,'balloon2-sam-overlay.mp4')]:
        target=args.out/name
        subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate',str(rate),'-i',str(args.out/(kind+'-frames')/'%03d.png'),
                        '-t',str(duration),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-r','24','-movflags','+faststart',str(target)],check=True)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(target),'-f','null','-'],check=True)
        outputs.append({'path':str(target.resolve()),'bytes':target.stat().st_size,'mime_type':'video/mp4',
                        'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'duration_seconds':duration,'private_only':True})
    (args.out/'transfer-manifest.json').write_text(json.dumps({'files':outputs,'attachment_created':False},indent=2))


if __name__=='__main__':main()
