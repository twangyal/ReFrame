import tempfile
import unittest
from pathlib import Path
import zipfile
from scripts import training_artifacts
from scripts.artifact_stream import emit_artifacts,receive_artifacts

class TrainingArtifactsTests(unittest.TestCase):
    def test_checkpoint_archive_exact_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';(source/'bundle').mkdir(parents=True)
            (source/'bundle/camera.pth').write_bytes(b'checkpoint')
            (source/'report.json').write_text('{}')
            package=root/'package';package.mkdir()
            training_artifacts.pack(source,package/'stages.zip')
            receive_artifacts(emit_artifacts(package,suffixes={'.zip'},max_bytes=1024**2),root/'transfer',suffixes={'.zip'},max_bytes=1024**2)
            training_artifacts.unpack(root/'transfer/stages.zip',root/'restored')
            self.assertEqual((root/'restored/bundle/camera.pth').read_bytes(),b'checkpoint')

    def test_reject_traversal_symlink_and_decompression_overflow_before_extracting(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for case in ('traversal','symlink','size'):
                z=root/(case+'.zip')
                with zipfile.ZipFile(z,'w') as f:
                    if case=='traversal':f.writestr('../escape',b'x')
                    elif case=='symlink':
                        info=zipfile.ZipInfo('link');info.create_system=3;info.external_attr=(0o120777<<16);f.writestr(info,'/private')
                    else:f.writestr('large',b'12345')
                with self.subTest(case=case),self.assertRaises(ValueError):training_artifacts.unpack(z,root/case,max_bytes=4)
                self.assertFalse((root/case).exists())
