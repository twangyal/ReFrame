from pathlib import Path
import tempfile
import unittest
from test_desktop_scene import bundle_fixture
from reframe.desktop_scene import SceneBundle
from reframe.desktop_session import DesktopSession,viewport_pixel


class DesktopSessionTests(unittest.TestCase):
    def test_selection_edits_undo_and_timeline_identity(self):
        with tempfile.TemporaryDirectory() as d:
            s=DesktopSession(SceneBundle.load(bundle_fixture(Path(d)/'scene')))
            self.assertEqual(s.mode,'baseline')
            with self.assertRaises(ValueError):s.set_mode('recolored')
            s.select(True);s.set_mode('recolored');s.seek(2);s.set_view('nearby')
            self.assertEqual(s.scene.object_id,'object-1');s.undo()
            self.assertTrue(s.selected);self.assertEqual(s.mode,'highlight');self.assertEqual(s.frame,2)
            s.select(False);self.assertEqual(s.mode,'baseline')
            s.tick();self.assertEqual(s.frame,0)
            with self.assertRaises(ValueError):s.seek(-1)
            with self.assertRaises(ValueError):s.set_view('invented')

    def test_project_roundtrip_and_wrong_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scene=SceneBundle.load(bundle_fixture(root/'scene'));s=DesktopSession(scene)
            s.select(True);s.set_mode('isolated');s.seek(2)
            p=root/'project.reframe.json';s.save(p)
            t=DesktopSession.open(p);self.assertEqual((t.mode,t.frame,t.selected),('isolated',2,True))
            with self.assertRaises(FileExistsError):s.save(p)
            scene.manifest_path.write_text(scene.manifest_path.read_text()+' ')
            with self.assertRaises(ValueError):DesktopSession.open(p)

    def test_letterbox_click_coordinates(self):
        self.assertEqual(viewport_pixel(200,150,(100,100,400,200),(16,8)),(4,2))
        self.assertIsNone(viewport_pixel(99,150,(100,100,400,200),(16,8)))
        self.assertIsNone(viewport_pixel(500,150,(100,100,400,200),(16,8)))
