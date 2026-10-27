import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.artifact_stream import emit_artifacts, receive_artifacts


class ArtifactStreamTests(unittest.TestCase):
    def test_large_binary_roundtrip_is_bounded_and_lossless(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir();out=Path(d)/'out'
            data=bytes(range(256))*5000
            (source/'selection.npz').write_bytes(data)
            events=list(emit_artifacts(source))
            self.assertTrue(all(len(e.get('data',b''))<=512*1024 for e in events))
            receive_artifacts(iter(events),out)
            self.assertEqual((out/'selection.npz').read_bytes(),data)

    def test_corrupt_or_incomplete_stream_never_finalizes_file(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';source.mkdir();(source/'selection.npz').write_bytes(b'valid')
            events=list(emit_artifacts(source))
            events[1]['data']=b'wrong'
            with self.assertRaises(ValueError): receive_artifacts(iter(events),Path(d)/'corrupt')
            self.assertFalse((Path(d)/'corrupt/selection.npz').exists())
            with self.assertRaises(ValueError): receive_artifacts(iter(events[:1]),Path(d)/'incomplete')

    def test_unsafe_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            events=iter([{'kind':'file','name':'../escape.npz','bytes':0,'sha256':hashlib.sha256(b'').hexdigest()}])
            with self.assertRaises(ValueError): receive_artifacts(events,Path(d)/'out')
