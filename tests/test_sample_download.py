import io
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.fetch_mosca_sample import extract_verified


class SampleDownloadTests(unittest.TestCase):
    def archive(self):
        b = io.BytesIO()
        with zipfile.ZipFile(b, 'w') as z:
            z.writestr('scene/model.pth', b'original')
        b.seek(0)
        return zipfile.ZipFile(b)

    def test_hash_validation_and_original_preservation(self):
        m = {'archive_path':'scene/model.pth','bytes':8,
             'sha256':hashlib.sha256(b'original').hexdigest()}
        with tempfile.TemporaryDirectory() as d, self.archive() as z:
            extract_verified(z, [m], Path(d))
            extract_verified(z, [m], Path(d))
            p = Path(d)/m['archive_path']
            p.write_bytes(b'changed!')
            with self.assertRaises(ValueError): extract_verified(z, [m], Path(d))
            self.assertEqual(p.read_bytes(), b'changed!')

    def test_mismatched_download_never_becomes_checkpoint(self):
        m = {'archive_path':'scene/model.pth','bytes':8,'sha256':'0'*64}
        with tempfile.TemporaryDirectory() as d, self.archive() as z:
            with self.assertRaises(ValueError): extract_verified(z, [m], Path(d))
            self.assertFalse((Path(d)/m['archive_path']).exists())

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d, self.archive() as z:
            with self.assertRaises(ValueError):
                extract_verified(z, [{'archive_path':'../escape'}], Path(d))
