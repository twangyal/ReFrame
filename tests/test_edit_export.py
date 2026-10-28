import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from reframe.media import sha256_file
from reframe.edit_render import MODES,VIEWS
from reframe.edit_export import export_edits


def fixture(root):
    root.mkdir();files={}
    for t in range(3):
        for view in VIEWS:
            for mode in MODES:
                p=root/f'{t:06d}_{view}_{mode}.png';Image.new('RGB',(16,12),(t*30,80,110)).save(p);files[p.name]=sha256_file(p)
    (root/'selection.npz').write_bytes(b'fixture')
    r={'kind':'gaussian-object-edits','completed':True,'status':'rendered_pending_visual_review',
       'phase_two_complete':False,'files':files,'rendered_frames':[0,1,2],'frames':3,'height':12,'width':16,
       'object_id':'fixture','selection_sha256':sha256_file(root/'selection.npz'),'translation_world':[.1,0,0],
       'modes':list(MODES),'views':list(VIEWS),'visual_review':'pending'}
    (root/'report.json').write_text(json.dumps(r));return r


class EditExportTests(unittest.TestCase):
    def test_export_playable_and_preserves_explicit_chosen_playback_rate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root/'renders')
            r=export_edits(root/'renders',root/'video',fps=3)
            self.assertFalse(r['phase_two_complete']);self.assertEqual(len(r['videos']),5)
            self.assertEqual(r['reconstruction_states'],3);self.assertFalse(r['interpolated'])
            for item in r['videos'].values():
                self.assertEqual(item['decoded_frames'],3)
                self.assertAlmostEqual(item['duration_seconds'],1.,places=2)
                self.assertEqual(item['sha256'],sha256_file(root/'video'/item['file']))
            with self.assertRaises(FileExistsError):export_edits(root/'renders',root/'video')

    def test_altered_missing_or_incomplete_renders_cannot_export(self):
        for broken in ['hash','missing','status','timeline']:
            with self.subTest(broken=broken),tempfile.TemporaryDirectory() as d:
                root=Path(d);r=fixture(root/'renders')
                if broken=='hash':(root/'renders/000000_source_recolored.png').write_bytes(b'altered')
                if broken=='missing':(root/'renders/000002_nearby_translated.png').unlink()
                if broken=='status':r['completed']=False
                if broken=='timeline':r['rendered_frames']=[0,2]
                (root/'renders/report.json').write_text(json.dumps(r))
                with self.assertRaises((ValueError,FileNotFoundError)):export_edits(root/'renders',root/'video')
                self.assertFalse((root/'video').exists())
