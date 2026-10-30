import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from reframe.media import sha256_file
from reframe.desktop_scene import SceneBundle


def bundle_fixture(root):
    root.mkdir();files={}
    for t in range(3):
        for view in ['source','nearby']:
            for mode in ['baseline','highlight','isolated','recolored','coverage']:
                name=f'{t:06d}_{view}_{mode}.png';p=root/name
                image=Image.new('RGB' if mode!='coverage' else 'L',(16,12),0)
                image.paste(255 if mode=='coverage' else (50+t*30,120,30),(4,3,10,9));image.save(p)
                files[name]=sha256_file(p)
    m={'schema':1,'kind':'reframe-desktop-scene','title':'Synthetic fixture','width':16,'height':12,
       'frames':3,'playback_fps':3,'object':{'id':'object-1','label':'Fixture object','dynamic_count':2,'static_count':0},
       'files':files,'private_only':True,'license_notice':'Synthetic test only',
       'provenance':{'selection_sha256':'a'*64,'render_report_sha256':'b'*64}}
    (root/'scene.json').write_text(json.dumps(m));return root/'scene.json'


class DesktopSceneTests(unittest.TestCase):
    def test_verified_scene_and_hit_testing(self):
        with tempfile.TemporaryDirectory() as d:
            s=SceneBundle.load(bundle_fixture(Path(d)/'scene'))
            self.assertEqual(s.frames,3);self.assertEqual(s.object_id,'object-1')
            self.assertTrue(s.hit_test(0,'source',5,4));self.assertFalse(s.hit_test(0,'source',0,0))
            self.assertFalse(s.hit_test(0,'source',-1,3));self.assertFalse(s.hit_test(0,'source',16,3))
            self.assertTrue(s.image_path(1,'nearby','recolored').exists())
            with self.assertRaises(ValueError):s.image_path(3,'nearby','baseline')
            with self.assertRaises(ValueError):s.image_path(0,'source','translated')

    def test_corruption_missing_traversal_and_wrong_size_rejected(self):
        for fault in ['corrupt','missing','traversal','dimensions','extra','boolean_schema','boolean_fps']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                p=bundle_fixture(Path(d)/'scene');m=json.loads(p.read_text())
                name='000000_source_baseline.png'
                if fault=='corrupt':(p.parent/name).write_bytes(b'broken')
                elif fault=='missing':(p.parent/name).unlink()
                elif fault=='traversal':m['files']['../outside.png']=m['files'].pop(name)
                elif fault=='dimensions':m['width']=18
                elif fault=='boolean_schema':m['schema']=True
                elif fault=='boolean_fps':m['playback_fps']=True
                else:m['files']['unused.png']='a'*64
                p.write_text(json.dumps(m))
                with self.assertRaises((ValueError,FileNotFoundError)):SceneBundle.load(p)

    def test_loaded_bundle_rechecks_changed_frame_before_use(self):
        with tempfile.TemporaryDirectory() as d:
            s=SceneBundle.load(bundle_fixture(Path(d)/'scene'))
            path=s.image_path(0,'source','baseline');path.write_bytes(b'changed')
            with self.assertRaises(ValueError):s.image_path(0,'source','baseline')

    def test_builder_rejects_render_report_identity_or_count_mismatch(self):
        import shutil
        from test_selection import fixture
        from reframe.selection import revise_selection,load_selection
        from reframe.desktop_scene import build_bundle
        from reframe.edit_render import MODES as RENDER_MODES
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);original=fixture(root);revise_selection(original,root/'candidate')
            selection=root/'candidate/render-selection.npz';s=load_selection(selection);meta=s['metadata']
            renders=root/'renders';renders.mkdir();coverage=root/'coverage';coverage.mkdir();files={}
            for t in range(meta['frames']):
                for view in ['source','nearby']:
                    for mode in RENDER_MODES:
                        name=f'{t:06d}_{view}_{mode}.png';Image.new('RGB',(2,2),'green').save(renders/name)
                        files[name]=sha256_file(renders/name)
                    shutil.copyfile(renders/f'{t:06d}_{view}_baseline.png',coverage/f'{t:06d}_{view}_baseline.png')
                    Image.new('L',(2,2),255).save(coverage/f'{t:06d}_{view}_visible_mask.png')
            shutil.copyfile(selection,renders/'selection.npz')
            report={**meta,'kind':'gaussian-object-edits','completed':True,'status':'rendered_pending_visual_review',
                    'modes':list(RENDER_MODES),'views':['source','nearby'],'files':files,
                    'selection_sha256':s['sha256'],'rendered_frames':list(range(meta['frames']))}
            (coverage/'report.json').write_text(json.dumps(report))
            (renders/'report.json').write_text(json.dumps(report))
            valid=build_bundle(renders,coverage,root/'valid');self.assertEqual(valid.object_id,meta['object_id'])
            for i,change in enumerate([{'object_id':'wrong-object'},{'selected_dynamic_count':123},{'selected_static_count':123}]):
                (renders/'report.json').write_text(json.dumps({**report,**change}))
                with self.subTest(change=change),self.assertRaises(ValueError):build_bundle(renders,coverage,root/f'bad-{i}')
