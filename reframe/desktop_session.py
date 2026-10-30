"""Desktop interaction state, independent of Tk and remote services."""
import json
from pathlib import Path
from .desktop_scene import SceneBundle,MODES,VIEWS


def viewport_pixel(x,y,rect,size):
    left,top,width,height=rect
    if width<=0 or height<=0 or not left<=x<left+width or not top<=y<top+height:return None
    return (int((x-left)*size[0]/width),int((y-top)*size[1]/height))


class DesktopSession:
    def __init__(self,scene):
        self.scene=scene;self.frame=0;self.view='source';self.mode='baseline';self.selected=False;self.history=[]

    def _remember(self):self.history.append((self.selected,self.mode))

    def select(self,value):
        if type(value) is not bool:raise ValueError('Selection must be boolean')
        if value==self.selected:return
        self._remember();self.selected=value;self.mode='highlight' if value else 'baseline'

    def set_mode(self,mode):
        if mode not in MODES or (mode!='baseline' and not self.selected):raise ValueError('Select an object before editing')
        if mode!=self.mode:self._remember();self.mode=mode

    def set_view(self,view):
        if view not in VIEWS:raise ValueError('Unknown saved camera')
        self.view=view

    def seek(self,frame):
        if type(frame) is not int or not 0<=frame<self.scene.frames:raise ValueError('Frame is outside the saved timeline')
        self.frame=frame

    def tick(self):self.frame=(self.frame+1)%self.scene.frames

    def undo(self):
        if self.history:self.selected,self.mode=self.history.pop()

    def reset(self):
        if self.selected or self.mode!='baseline':self._remember()
        self.selected=False;self.mode='baseline'

    def save(self,path):
        # Exclusive creation preserves existing session files.
        payload={'schema':1,'scene_path':str(self.scene.manifest_path),'scene_sha256':self.scene.digest,
                 'object_id':self.scene.object_id,'frame':self.frame,'view':self.view,'mode':self.mode,'selected':self.selected}
        with Path(path).open('x') as handle:json.dump(payload,handle,indent=2)

    @classmethod
    def open(cls,path):
        path=Path(path)
        if path.stat().st_size>16384:raise ValueError('Session file is too large')
        p=json.loads(path.read_text())
        if p.get('schema')!=1:raise ValueError('Unsupported session')
        scene=SceneBundle.load(p['scene_path'])
        if scene.digest!=p['scene_sha256'] or scene.object_id!=p['object_id']:raise ValueError('Session belongs to another scene revision')
        s=cls(scene);s.select(p['selected']);s.set_mode(p['mode']);s.seek(p['frame']);s.set_view(p['view']);s.history=[]
        return s
