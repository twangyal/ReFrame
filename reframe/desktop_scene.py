"""Portable, verified saved-render scenes for the offline desktop workflow."""
import argparse
import json
from pathlib import Path
import shutil
from PIL import Image
from .media import sha256_file

MODES=('baseline','highlight','isolated','recolored')
VIEWS=('source','nearby')


class SceneBundle:
    def __init__(self,path,manifest):
        self.manifest_path=path;self.root=path.parent;self.data=manifest
        self.digest=sha256_file(path);self.frames=manifest['frames']
        self.width=manifest['width'];self.height=manifest['height']
        self.object_id=manifest['object']['id'];self.fps=manifest['playback_fps']

    @classmethod
    def load(cls,path):
        path=Path(path).resolve()
        if path.stat().st_size>1024**2:raise ValueError('Scene manifest is too large')
        m=json.loads(path.read_text())
        if type(m.get('schema')) is not int or m['schema']!=1 or m.get('kind')!='reframe-desktop-scene':raise ValueError('Unsupported scene bundle')
        if (type(m.get('frames')) is not int or not 1<=m['frames']<=120
            or any(type(m.get(k)) is not int or not 1<=m[k]<=4096 for k in ('width','height'))
            or type(m.get('playback_fps')) not in (int,float) or not 0<m['playback_fps']<=60):
            raise ValueError('Invalid saved timeline or dimensions')
        if (not isinstance(m.get('object'),dict) or not isinstance(m['object'].get('id'),str)
            or not m['object']['id'] or not isinstance(m['object'].get('label'),str)
            or not isinstance(m.get('title'),str) or not isinstance(m.get('license_notice'),str)
            or m.get('private_only') is not True):raise ValueError('Missing identity or private asset notice')
        for key in ('static_count','dynamic_count'):
            if type(m['object'].get(key)) is not int or m['object'][key]<0:raise ValueError('Invalid Gaussian count')
        expected={f'{t:06d}_{view}_{mode}.png' for t in range(m['frames']) for view in VIEWS for mode in (*MODES,'coverage')}
        if not isinstance(m.get('files'),dict) or set(m['files'])!=expected:raise ValueError('Incomplete or unexpected scene files')
        scene=cls(path,m)
        for name in expected:
            p=scene._verified(name)
            with Image.open(p) as image:
                image.load()
                if image.size!=(scene.width,scene.height):raise ValueError('Scene image dimensions differ')
                if image.mode!=('L' if name.endswith('_coverage.png') else 'RGB'):raise ValueError('Unexpected scene image mode')
        return scene

    def _verified(self,name):
        path=(self.root/name).resolve()
        if path.parent!=self.root:raise ValueError('Scene path escapes its directory')
        if sha256_file(path)!=self.data['files'][name]:raise ValueError(f'Scene file changed: {name}')
        return path

    def image_path(self,frame,view,mode):
        if type(frame) is not int or not 0<=frame<self.frames or view not in VIEWS or mode not in MODES:
            raise ValueError('Unsupported frame, camera or edit')
        return self._verified(f'{frame:06d}_{view}_{mode}.png')

    def hit_test(self,frame,view,x,y):
        self.image_path(frame,view,'baseline')
        if not 0<=x<self.width or not 0<=y<self.height:return False
        with Image.open(self._verified(f'{frame:06d}_{view}_coverage.png')) as image:
            return image.getpixel((int(x),int(y)))>=128


def build_bundle(renders,coverage,out):
    """Package matched existing GPU outputs, never render or infer new pixels."""
    from .edit_export import validate_render_set
    from .selection import load_selection,validate_scene_binding
    renders=Path(renders).resolve();coverage=Path(coverage).resolve();out=Path(out).resolve()
    r=validate_render_set(renders);c=json.loads((coverage/'report.json').read_text())
    s=load_selection(renders/'selection.npz');validate_scene_binding(s,c);validate_scene_binding(s,r)
    for report in (r,c):
        for key in ('object_id','selected_static_count','selected_dynamic_count','aggregation'):
            if report.get(key)!=s['metadata'].get(key):raise ValueError(f'Render report contradicts saved selection: {key}')
    if (c.get('selection_sha256')!=s['sha256'] or c.get('completed') is not True
        or c.get('rendered_frames')!=r['rendered_frames']):raise ValueError('Coverage belongs to a different selection/timeline')
    if out==renders or renders in out.parents or out==coverage or coverage in out.parents:
        raise ValueError('Bundle must be separate from original outputs')
    sources={}
    for t in r['rendered_frames']:
        for view in VIEWS:
            # Check exact old/new baseline pixels before reusing its coverage.
            name=f'{t:06d}_{view}_baseline.png'
            with Image.open(renders/name) as a,Image.open(coverage/name) as b:
                if a.size!=b.size or a.tobytes()!=b.tobytes():raise ValueError('Coverage camera or baseline differs')
            for mode in MODES:
                name=f'{t:06d}_{view}_{mode}.png';sources[name]=renders/name
            sources[f'{t:06d}_{view}_coverage.png']=coverage/f'{t:06d}_{view}_visible_mask.png'
    # Verify inputs before creating output. Bundle failure never alters originals.
    for name,path in sources.items():
        with Image.open(path) as image:
            image.load()
            if image.size!=(r['width'],r['height']):raise ValueError('Coverage or render dimensions differ')
    out.mkdir(parents=True,exist_ok=False)
    files={}
    for name,path in sources.items():
        shutil.copyfile(path,out/name);files[name]=sha256_file(out/name)
    manifest={'schema':1,'kind':'reframe-desktop-scene','title':'Balloon2',
              'width':r['width'],'height':r['height'],'frames':r['frames'],'playback_fps':3,
              'object':{'id':r['object_id'],'label':'Dinosaur balloon','static_count':r['selected_static_count'],
                        'dynamic_count':r['selected_dynamic_count'],'membership_threshold':r['aggregation']['foreground_threshold']},
              'private_only':True,'license_notice':'Private research sample. Benchmark asset redistribution rights are unverified; do not publish footage or exports.',
              'provenance':{'selection_sha256':s['sha256'],'render_report_sha256':sha256_file(renders/'report.json'),
                            'coverage_report_sha256':sha256_file(coverage/'report.json'),'checkpoint_sha256':r['checkpoint_sha256']},
              'limitations':['Saved GPU renders, not live reconstruction or arbitrary editing.',
                             'Twelve discrete states; chosen 3 fps playback, not verified acquisition timing.',
                             'Only verified orange recolor and two saved views are available.',
                             'Clean translation failed visual acceptance and is not enabled.'],
              'files':files}
    (out/'scene.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return SceneBundle.load(out/'scene.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renders',type=Path,required=True);parser.add_argument('--coverage',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    scene=build_bundle(a.renders,a.coverage,a.out);print(scene.manifest_path)


if __name__=='__main__':main()
