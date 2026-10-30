import json
from pathlib import Path
import tempfile
import unittest
from test_desktop_scene import bundle_fixture
from reframe.desktop_scene import SceneBundle
from reframe.desktop_session import DesktopSession
from reframe.desktop_export import export_session


class DesktopExportTests(unittest.TestCase):
    def test_selected_view_mode_produces_verified_independent_mp4(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s=DesktopSession(SceneBundle.load(bundle_fixture(root/'scene')))
            s.select(True);s.set_mode('recolored');s.set_view('nearby')
            progress=[];p=root/'edited.mp4';r=export_session(s,p,progress=lambda a,b:progress.append((a,b)))
            self.assertTrue(p.exists());self.assertEqual(r['mode'],'recolored');self.assertEqual(r['view'],'nearby')
            self.assertEqual(r['video']['decoded_frames'],3);self.assertTrue(r['private_only'])
            self.assertEqual(progress[-1],(3,3));self.assertTrue(p.with_suffix('.export.json').exists())
            with self.assertRaises(FileExistsError):export_session(s,p)

    def test_changed_input_and_encoder_failure_preserve_existing_files(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s=DesktopSession(SceneBundle.load(bundle_fixture(root/'scene')))
            p=root/'edited.mp4'
            with patch('reframe.desktop_export._encode',side_effect=RuntimeError('encoder failed')):
                with self.assertRaises(RuntimeError):export_session(s,p)
            self.assertFalse(p.exists());self.assertFalse(p.with_suffix('.export.json').exists())
            s.scene.image_path(0,'source','baseline').write_bytes(b'changed')
            with self.assertRaises(ValueError):export_session(s,p)
            self.assertFalse(p.exists())
