import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from test_desktop_scene import bundle_fixture
from scripts.package_desktop_demo import package


class DesktopPackageTests(unittest.TestCase):
    def test_archive_contains_only_allowlisted_source_and_verified_scene(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=bundle_fixture(root/'scene');out=root/'demo.zip'
            r=package(p,out);self.assertTrue(r['private_only'])
            with zipfile.ZipFile(out) as z:
                names=z.namelist();self.assertIn('reframe/desktop.py',names);self.assertIn('sample/scene.json',names)
                self.assertFalse(any('.cache' in n or n.endswith(('.pth','.toml','.env')) for n in names))
                self.assertTrue(json.loads(z.read('ARCHIVE_MANIFEST.json'))['private_only'])
                self.assertIsNone(z.testzip())
            with self.assertRaises(FileExistsError):package(p,out)
