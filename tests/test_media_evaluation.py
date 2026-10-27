import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from reframe.media import prepare_video, validate_manifest, prepare_scene_video
from reframe.evaluation import mask_metrics


class MediaTests(unittest.TestCase):
    def test_scene_images_roundtrip_without_pixel_drift(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'scene';(source/'images').mkdir(parents=True)
            original=np.random.default_rng(7).integers(0,256,(24,32,3),dtype=np.uint8)
            Image.fromarray(original).save(source/'images'/'00000.png')
            Image.fromarray(original[::-1].copy()).save(source/'images'/'00001.png')
            manifest=prepare_scene_video(source,root/'prepared',fps=2)
            self.assertFalse(manifest['original_timestamps_known'])
            decoded=np.asarray(Image.open(root/'prepared'/'frames'/manifest['frames'][0]['file']))
            np.testing.assert_array_equal(decoded,original)
            decoded=np.asarray(Image.open(root/'prepared'/'frames'/manifest['frames'][1]['file']))
            np.testing.assert_array_equal(decoded,original[::-1])

    def test_real_decode_manifest_detects_changed_frame(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); video=root/'tiny.mp4'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                            'testsrc2=size=32x24:rate=3:duration=1','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
            manifest=prepare_video(video,root/'frames')
            self.assertEqual(manifest['frame_count'],3)
            self.assertEqual((manifest['height'],manifest['width']),(24,32))
            self.assertEqual([f['index'] for f in manifest['frames']],[0,1,2])
            self.assertAlmostEqual(manifest['frames'][1]['pts_seconds'],1/3,places=5)
            validate_manifest(root/'frames')
            with self.assertRaises(FileExistsError): prepare_video(video,root/'frames')
            (root/'frames'/manifest['frames'][0]['file']).write_bytes(b'changed')
            with self.assertRaises(ValueError): validate_manifest(root/'frames')

class EvaluationTests(unittest.TestCase):
    def test_empty_union_is_unscored_not_perfect_geometry(self):
        r=mask_metrics(np.zeros((2,2),bool),np.zeros((2,2),bool))
        self.assertIsNone(r['iou'])
    def test_iou_and_contamination_hand_calculated(self):
        r=mask_metrics(np.array([[1,1],[0,0]],bool),np.array([[1,0],[1,0]],bool))
        self.assertAlmostEqual(r['iou'],1/3)
        self.assertAlmostEqual(r['false_positive_fraction'],.5)
        self.assertEqual(r['false_negative_pixels'],1)
    def test_shape_and_nonbinary_rejected(self):
        with self.assertRaises(ValueError): mask_metrics(np.ones((1,2)),np.ones((2,1)))
        with self.assertRaises(ValueError): mask_metrics(np.array([[.4]]),np.ones((1,1)))

if __name__ == '__main__': unittest.main()
