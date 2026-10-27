import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

import numpy as np
from PIL import Image
from scripts.modal_scene_worker import collect_artifacts, remote_entrypoint


class SceneArtifactTests(unittest.TestCase):
    def test_timeout_preserves_partial_diagnostics(self):
        error = subprocess.TimeoutExpired('probe',850,output=b'checkpoint verified',stderr=b'last stage')
        with patch('scripts.modal_scene_worker.subprocess.run',side_effect=error):
            result = remote_entrypoint()
        self.assertEqual(result['exit_code'],124)
        self.assertIn('checkpoint verified',result['stdout'])
        self.assertIn('last stage',result['stderr'])
        self.assertTrue(result['timed_out'])

    def test_large_png_results_keep_all_views_with_explicit_jpeg_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'baseline.json').write_text('{"completed": true}')
            rng = np.random.default_rng(20)
            for i in range(6):
                Image.fromarray(rng.integers(0,256,(270,480,3),dtype=np.uint8)).save(root/f'{i}.png')
            result, encoding = collect_artifacts(root)
            self.assertEqual(encoding, 'JPEG quality 95 previews; original render dimensions')
            self.assertLessEqual(len(json.dumps(result)), 1500000)
            self.assertEqual(len(result), 7)
            self.assertEqual(json.loads(base64.b64decode(result['baseline.json'])), {'completed':True})
            for name, value in result.items():
                if name.endswith('.jpg'):
                    self.assertEqual(Image.open(io.BytesIO(base64.b64decode(value))).size,(480,270))
            self.assertEqual(len(list(root.glob('*.png'))),6)
