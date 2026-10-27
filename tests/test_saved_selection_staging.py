import json
from pathlib import Path
import tempfile
import unittest
from reframe.media import sha256_file
from scripts.modal_smoke import stage_association


class SavedSelectionStagingTests(unittest.TestCase):
    def test_full_twelve_frame_render_transfers_with_finite_file_bound(self):
        from scripts.artifact_stream import emit_artifacts,receive_artifacts
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir()
            for i in range(110):(source/f'{i:03d}.png').write_bytes(b'fixture')
            receive_artifacts(emit_artifacts(source),root/'received')
            self.assertEqual(len(list((root/'received').iterdir())),110)
            for i in range(110,129):(source/f'{i:03d}.png').write_bytes(b'fixture')
            with self.assertRaises(ValueError):list(emit_artifacts(source))

    def test_compact_sidecar_stages_with_explicit_render_mode(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frames=root/'frames';frames.mkdir();rows=[]
            for i in range(12):
                p=frames/f'{i:03d}.png';p.write_bytes(b'frame fixture')
                rows.append({'file':p.name,'sha256':sha256_file(p)})
            (frames/'manifest.json').write_text(json.dumps({'frame_count':12,'width':480,'height':270,'frames':rows}))
            for name in ['masks.npz','baseline.json','compact.npz']:(root/name).write_bytes(b'fixture')
            hashes=stage_association(frames,root/'masks.npz',root/'baseline.json',root/'stage',
                                     selection=root/'compact.npz',all_frames=True)
            self.assertEqual(hashes['render-selection.npz'],sha256_file(root/'compact.npz'))
            settings=json.loads((root/'stage/association-input/settings.json').read_text())
            self.assertEqual(settings['mode'],'render-selection')
            self.assertTrue(settings['all_frames'])
            self.assertEqual((root/'stage/association-input/render-selection.npz').read_bytes(),b'fixture')

    def test_worker_routes_saved_selection_without_association_or_sam(self):
        from scripts.modal_scene_worker import selection_command
        cmd=selection_command({'mode':'render-selection','all_frames':True},Path('/checkpoint'))
        self.assertIn('render-selection',cmd)
        self.assertNotIn('associate',cmd)
        self.assertNotIn('segment',cmd)
        self.assertIn('--all-frames',cmd)
        self.assertIn('/smoke/association-input/render-selection.npz',cmd)
        with self.assertRaises(ValueError):selection_command({'mode':'unknown'},Path('/checkpoint'))
