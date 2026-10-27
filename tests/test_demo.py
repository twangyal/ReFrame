import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from test_selection import fixture
from reframe.media import sha256_file


def demo_fixture(root):
    selection=fixture(root);folder=selection.parent
    with np.load(selection,allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
    prepared=root/'prepared';frames=prepared/'frames';frames.mkdir(parents=True)
    rows=[]
    for i in range(5):
        p=frames/f'{i:03d}.png';Image.new('RGB',(2,2)).save(p)
        rows.append({'index':i,'file':p.name,'pts_seconds':i/3,'sha256':sha256_file(p)})
    manifest={'schema':1,'frame_count':5,'height':2,'width':2,'frames':rows,'playback_fps':3}
    (frames/'manifest.json').write_text(json.dumps(manifest))
    meta['segmentation']['frame_manifest_sha256']=sha256_file(frames/'manifest.json')
    with np.load(selection,allow_pickle=False) as z:values=dict(z)
    values['metadata']=json.dumps(meta);np.savez_compressed(selection,**values)
    report={**meta,'evaluation':[{'frame':0,'iou':.8},{'frame':2,'iou':.9},{'frame':4,'iou':.7}]}
    (folder/'report.json').write_text(json.dumps(report))
    initial=root/'initial.json';initial.write_text(json.dumps(report))
    review=root/'review.json';review.write_text(json.dumps({'object_id':'fixture-object','selected_dynamic':1,'selected_static':1,'qualified_feasibility_demonstrated':True}))
    sam=root/'sam';sam.mkdir();np.savez_compressed(sam/'masks.npz',masks=np.zeros((5,2,2),bool),reliable=np.ones(5,bool),metadata=json.dumps(meta['segmentation']))
    for t in (0,2,4):
        for view in ('source','nearby'):
            for mode in ('baseline','highlight','isolated'):
                Image.new('RGB',(2,2)).save(folder/f'{t:06d}_{view}_{mode}.png')
        Image.new('RGB',(2,2)).save(folder/f'{t:06d}_mask_comparison.png')
    config={'prepared':'prepared','sam':'sam','selection_dir':'original','initial_report':'initial.json','visual_review':'review.json',
            'legacy_render_binding':{'selection_sha256':sha256_file(selection),'report_sha256':sha256_file(folder/'report.json')},
            'sha256':{str(p.relative_to(root)):sha256_file(p) for p in root.rglob('*') if p.is_file()}}
    return config


class DemoTests(unittest.TestCase):
    def test_completed_render_can_bind_a_previously_unrendered_revision(self):
        from reframe.demo import validate_inputs
        from reframe.selection import revise_selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=demo_fixture(root)
            revise_selection(root/'original/selection.npz',root/'candidate')
            (root/'original/selection.npz').write_bytes((root/'candidate/selection.npz').read_bytes())
            p=root/'original/report.json';report=json.loads(p.read_text())
            report.update(selection_sha256=sha256_file(root/'original/selection.npz'),
                          completed=True,kind='saved-selection-render',status='rendered_pending_visual_review')
            p.write_text(json.dumps(report))
            for p in [p,root/'original/selection.npz']:config['sha256'][str(p.relative_to(root))]=sha256_file(p)
            self.assertTrue(validate_inputs(config,root)['report']['completed'])

    def test_unrendered_revision_cannot_be_presented_with_legacy_report(self):
        from reframe.demo import validate_inputs
        from reframe.selection import revise_selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=demo_fixture(root)
            revise_selection(root/'original/selection.npz',root/'candidate')
            (root/'original/selection.npz').write_bytes((root/'candidate/selection.npz').read_bytes())
            config['sha256']['original/selection.npz']=sha256_file(root/'original/selection.npz')
            with self.assertRaises(ValueError):validate_inputs(config,root)

    def test_validate_rejects_changed_render_before_assembly(self):
        from reframe.demo import validate_inputs
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=demo_fixture(root)
            result=validate_inputs(config,root)
            self.assertEqual(result['report']['evaluation'][0]['iou'],.8)
            (root/'original/000000_source_highlight.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_inputs(config,root)

    def test_rejects_mismatched_masks_even_if_file_digest_updated(self):
        from reframe.demo import validate_inputs
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=demo_fixture(root);p=root/'sam/masks.npz'
            with np.load(p,allow_pickle=False) as z:arrays=dict(z)
            meta=json.loads(str(arrays['metadata']));meta['cache_key']='another-track';arrays['metadata']=json.dumps(meta)
            np.savez_compressed(p,**arrays);config['sha256']['sam/masks.npz']=sha256_file(p)
            with self.assertRaises(ValueError):validate_inputs(config,root)

    def test_candidate_has_no_old_quality_and_must_belong_to_measured_parent(self):
        from reframe.demo import validate_inputs
        from reframe.selection import revise_selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=demo_fixture(root)
            revise_selection(root/'original/selection.npz',root/'candidate',exclusions=[{'pool':'static','index':0,'reason':'test ablation'}])
            config['candidate_dir']='candidate'
            config['sha256'].update({str(p.relative_to(root)):sha256_file(p) for p in (root/'candidate').glob('*')})
            result=validate_inputs(config,root)
            self.assertEqual(result['candidate']['status'],'unrendered_exploratory_candidate')
            self.assertNotIn('evaluation',result['candidate'])
            p=root/'candidate/revision.json';data=json.loads(p.read_text());data['parent_selection_sha256']='f'*64;p.write_text(json.dumps(data))
            config['sha256']['candidate/revision.json']=sha256_file(p)
            with self.assertRaises(ValueError):validate_inputs(config,root)

    def test_existing_output_is_preserved(self):
        from reframe.demo import build_demo
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=demo_fixture(root);out=root/'demo';out.mkdir();(out/'keep').write_text('original')
            with self.assertRaises(FileExistsError):build_demo(config,root,out)
            self.assertEqual((out/'keep').read_text(),'original')
