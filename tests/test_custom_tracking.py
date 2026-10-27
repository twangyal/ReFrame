import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts import custom_tracking

class TrackingStagingTests(unittest.TestCase):
    def test_hash_bound_explicit_inputs_exclude_unlisted_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'priors';(source/'images').mkdir(parents=True);(source/'external_depth').mkdir()
            rows=[]
            for i in range(48):
                name=f'{i:06d}.png';image=b'image';depth=b'depth'
                (source/'images'/name).write_bytes(image)
                (source/'external_depth'/Path(name).with_suffix('.npz')).write_bytes(depth)
                rows.append(dict(prepared_index=i,file=name,source_image_sha256=hashlib.sha256(image).hexdigest(),depth_sha256=hashlib.sha256(depth).hexdigest()))
            report=dict(rows=rows,all_prepared_frames=True)
            (source/'report.json').write_text(json.dumps(report))
            (source/'private.env').write_text('must not upload')
            hashes=custom_tracking.stage_priors(source,root/'stage')
            self.assertEqual(len(hashes),97)
            self.assertFalse((root/'stage/private.env').exists())
            (source/'images/000000.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):
                custom_tracking.stage_priors(source,root/'bad')
            report['rows'][0]['file']='../private.env'
            (source/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'path'):
                custom_tracking.stage_priors(source,root/'unsafe')

    def test_weight_digest_checked_before_staging(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);weight=root/'weight.pt';weight.write_bytes(b'incorrect')
            with self.assertRaisesRegex(ValueError,'digest'):
                custom_tracking.stage_weight(weight,root/'out.pt')
            self.assertFalse((root/'out.pt').exists())

    def test_tracks_require_time_identity_visibility_and_finite_coordinates(self):
        import numpy as np
        tracks=np.zeros((48,128,2),dtype=np.float32);visible=np.ones((48,128),dtype=bool)
        stats=custom_tracking.validate_tracks(tracks,visible)
        self.assertEqual(stats['tracks_visible_at_least_four_frames'],128)
        with self.assertRaises(ValueError):custom_tracking.validate_tracks(tracks[:-1],visible[:-1])
        tracks[0,0,0]=np.nan
        with self.assertRaises(ValueError):custom_tracking.validate_tracks(tracks,visible)
        tracks[0,0,0]=0
        with self.assertRaises(ValueError):custom_tracking.validate_tracks(tracks,visible.astype(float))
