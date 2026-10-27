"""Encode existing source/nearby selection renders as explicitly held stills.

No model inference, interpolation, rendering, transformation, or upload occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.selection / 'report.json').read_text())
    dynamic = report['selected_dynamic_count']
    static = report['selected_static_count']
    args.out.mkdir(parents=True, exist_ok=False)
    frames = args.out / 'frames'
    frames.mkdir()
    inputs = []
    times = [0, 6, 11]
    for i, t in enumerate(times):
        canvas = Image.new('RGB', (1440, 660), '#151515')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f'ReFrame | real MoSca Gaussian selection | reconstruction frame {t}/11', fill='white')
        draw.text((12, 25), 'Three sampled timestamps held for 2 seconds each; no interpolation or object translation', fill='white')
        for row, view in enumerate(['source', 'nearby']):
            for col, mode in enumerate(['baseline', 'highlight', 'isolated']):
                path = args.selection / f'{t:06d}_{view}_{mode}.png'
                inputs.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                with Image.open(path) as image:
                    if image.size != (480, 270):
                        raise ValueError(f'Unexpected input dimensions: {path}')
                    canvas.paste(image.convert('RGB'), (col * 480, 66 + row * 292))
                draw.text((col * 480 + 12, 48 + row * 292), f'{view} camera | {mode}', fill='white')
        draw.text((12, 642), f'Exploratory refinement | {dynamic:,} dynamic + {static:,} static rows | boundaries/holes/outliers remain | nearby view only', fill='white')
        canvas.save(frames / f'{i:03d}.png')
    target = args.out / 'balloon2-selection-sampled-comparison.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-framerate', '1/2',
                    '-i', str(frames / '%03d.png'), '-t', '6', '-c:v', 'libx264',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-r', '24',
                    '-movflags', '+faststart', str(target)], check=True)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(target),
                    '-f', 'null', '-'], check=True)
    manifest = {'path': str(target.resolve()), 'bytes': target.stat().st_size,
                'mime_type': 'video/mp4', 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                'duration_seconds': 6, 'sampled_frames': times,
                'seconds_per_held_render': 2, 'continuous_motion': False,
                'translation_applied': False, 'private_only': True,
                'attachment_created': False, 'input_renders': inputs}
    (args.out / 'transfer-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'input_renders'}, indent=2))


if __name__ == '__main__':
    main()
