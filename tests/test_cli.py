import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from test_segmentation import LINE,response


class CliTests(unittest.TestCase):
    def test_offline_import_and_overlay_without_credentials_or_network(self):
        # Synthetic video + synthetic encoded parser fixture: NOT real SAM inference.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);video=root/'synthetic.mp4';frames=root/'frames';out=root/'masks'
            subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i',
                            'color=blue:size=320x334:rate=3:duration=1','-c:v','libx264',str(video)],check=True)
            subprocess.run([sys.executable,'-m','reframe.cli','prepare-video',str(video),'--out',str(frames)],
                           check=True,stdout=subprocess.DEVNULL)
            from reframe.media import sha256_file
            manifest=json.loads((frames/'manifest.json').read_text())
            raw=root/'fixture.json'
            payload={'provenance':{'video_sha256':manifest['video_sha256'],
                                  'frame_manifest_sha256':sha256_file(frames/'manifest.json'),
                                  'model':'sam-3.1','prompt':'fixture-only'},
                     'response':response(f'<2f>{LINE}\n<0f>{LINE}')}
            raw.write_text(json.dumps(payload))
            command=[sys.executable,'-m','reframe.cli','segment','--frames',str(frames),
                     '--prompt','fixture-only','--object-id','2','--response',str(raw),'--out',str(out)]
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            with np.load(out/'masks.npz',allow_pickle=False) as z:
                self.assertEqual(z['observed'].tolist(),[True,False,True])
                self.assertEqual(z['masks'].shape,(3,334,320))
                self.assertEqual(json.loads(str(z['metadata']))['response_origin'],'imported')
                self.assertFalse(z['reliable'].any()) # fixture has no confidence
            self.assertEqual(len(list(out.glob('overlay_*.png'))),3)
            completed=subprocess.run(command,capture_output=True)
            self.assertEqual(completed.returncode,2)
            self.assertTrue((out/'masks.npz').exists())

if __name__=='__main__': unittest.main()
