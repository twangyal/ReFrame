import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from reframe.media import sha256_file


class CustomSceneTests(unittest.TestCase):
    def test_prepare_keeps_capture_indices_and_preserves_source(self):
        from reframe.custom_scene import prepare_custom_scene
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);video=root/'clip.mp4'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=64x48:rate=4:duration=4','-c:v','libx264',str(video)],check=True)
            before=sha256_file(video)
            report=prepare_custom_scene(video,root/'scene',max_frames=8,max_side=32)
            self.assertEqual(report['selected_decode_indices'],[0,2,4,6,8,10,12,15])
            self.assertEqual(report['width'],32);self.assertEqual(report['height'],24)
            self.assertFalse(report['reconstruction_completed'])
            self.assertEqual(sha256_file(video),before)
            self.assertEqual(len(list((root/'scene/images').glob('*.png'))),8)
            self.assertEqual(report['frames'][-1]['source_pts_seconds'],3.75)
            with self.assertRaises(FileExistsError):prepare_custom_scene(video,root/'scene')

    def test_wrong_duration_is_not_silently_trimmed(self):
        from reframe.custom_scene import prepare_custom_scene
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);video=root/'short.mp4'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=blue:size=32x24:rate=4:duration=1','-c:v','libx264',str(video)],check=True)
            with self.assertRaises(ValueError):prepare_custom_scene(video,root/'scene')
            self.assertFalse((root/'scene').exists())
