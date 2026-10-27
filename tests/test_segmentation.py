import tempfile
import unittest
from pathlib import Path

import numpy as np

from reframe.segmentation import decode_response, request_body, cache_key, load_cached_response, validate_import


# Synthetic 60x27 raster: rectangle x=[10,50), y=[5,22), area680.
# Encoded once using meta-sam-parser0.0.6 _encode_raster; no published model output.
MASK = "!!!!!0@HRC3Ourv+X.fSn!!"
LINE = f"2<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,{MASK}|>"

def response(text, status='completed'):
    return {'id':'fixture', 'status':status, 'output':[{'type':'message','content':[{'type':'output_text','text':text}]}]}


class SegmentationTests(unittest.TestCase):
    def test_sparse_ids_reverse_frames_and_box_local_raster(self):
        masks, present, confidence = decode_response(response(f'<2f>{LINE}\n<0f>{LINE}'), '2', 3, 334, 320)
        self.assertEqual(present.tolist(), [True,False,True])
        self.assertEqual(int(masks[0].sum()), 680)
        self.assertEqual(masks[1].sum(), 0)
        np.testing.assert_array_equal(masks[0], masks[2])
        self.assertEqual(masks[0,:228].sum(),0)
        self.assertEqual(masks[0,255:].sum(),0)
        self.assertEqual(masks[0,:,:211].sum(),0)
        self.assertEqual(masks[0,:,271:].sum(),0)

    def test_missing_object_is_not_silently_replaced_by_first_object(self):
        masks,present,_ = decode_response(response(f'<0f>{LINE}'), '0', 1,334,320)
        self.assertFalse(present.any())
        self.assertFalse(masks.any())

    def test_incomplete_malformed_wrong_resolution_and_invalid_frames_rejected(self):
        for r,n,h,w in [(response('', 'incomplete'),1,334,320),
                        (response('<0f>garbage'),1,334,320),
                        (response(f'<0f>{LINE}'),1,335,320),
                        (response(f'<2f>{LINE}'),2,334,320)]:
            with self.subTest(r=r), self.assertRaises(ValueError): decode_response(r,'2',n,h,w)

    def test_empty_completed_response_is_no_matches(self):
        masks,present,_=decode_response(response(''),'2',2,10,20)
        self.assertEqual(masks.shape,(2,10,20))
        self.assertFalse(present.any())

    def test_reordered_box_fields_still_reject_wrong_source_dimensions(self):
        text='<0f>'+LINE.replace(';w=320;h=334',';h=335;w=321')
        with self.assertRaises(ValueError): decode_response(response(text),'2',1,334,320)

    def test_confidence_is_preserved_separately_from_presence(self):
        text='<0f>'+LINE.replace(';h=334',';h=334;c=0').replace(';y=0;data',';y=0;c=0;data')
        _,present,confidence=decode_response(response(text),'2',2,334,320)
        self.assertTrue(present[0]);self.assertEqual(confidence[0],0.)
        self.assertTrue(np.isnan(confidence[1]))
        self.assertFalse((present & (confidence>=.5)).any())

    def test_import_must_record_matching_original_video_and_manifest(self):
        raw=response('')
        with self.assertRaises(ValueError): validate_import(raw,'video','manifest','chair')
        payload={'provenance':{'video_sha256':'video','frame_manifest_sha256':'wrong',
                              'model':'sam-3.1','prompt':'chair'},'response':raw}
        with self.assertRaises(ValueError): validate_import(payload,'video','manifest','chair')
        payload['provenance']['frame_manifest_sha256']='manifest'
        self.assertEqual(validate_import(payload,'video','manifest','chair'),raw)

    def test_request_and_cache_include_exact_input_and_options(self):
        body = request_body('file-fixture','chair')
        self.assertEqual(body['model'],'sam-3.1')
        self.assertEqual(body['input'][0]['content'][1],{'type':'input_video','file_id':'file-fixture'})
        self.assertEqual(body['metadata']['mask_encoding'],'one_bit')
        self.assertNotEqual(cache_key('abc','chair'),cache_key('def','chair'))
        self.assertNotEqual(cache_key('abc','chair'),cache_key('abc','table'))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'response.json'
            p.write_text('{"status":"incomplete"}')
            with self.assertRaises(ValueError): load_cached_response(p)

if __name__ == '__main__': unittest.main()
